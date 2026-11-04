//go:build !js

// Command hang is the hostile guest of the GPU time budget e2e: it submits a
// compute shader that runs far longer than the host's budget, sized from the
// adapter's limits and from a calibration on the adapter itself (every
// workgroup the device allows in X, 16 in Y; the per-invocation loop is
// scaled until the shader needs about TARGET ms), and reports what the host
// answered.
//
//	hang [TARGET_MS [split]]
//
// By default the shader module, the pipeline and the dispatch are in one
// frame, which the host holds to its hard cap (frames that create objects
// may compile). With "split" the objects are created and flushed in a first
// frame and the long dispatch is a frame of its own, which the host holds to
// the steady-state budget.
//
// After the overrun it checks what the sanction promises: the device is
// gone, a new context is refused as revoked.
package main

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/go-gfx/webgpu"
	"github.com/go-wasmcontainers/gpu/wasigpu/batchsink"
)

func shader(iters uint64) string {
	return fmt.Sprintf(`
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  var x = id.x + id.y * 65537u + 1u;
  for (var i = 0u; i < %du; i = i + 1u) {
    x = x * 1664525u + 1013904223u;
    if (x == 0xFFFFFFFFu) { break; }
  }
  out[(id.x + id.y) %% arrayLength(&out)] = x;
}`, iters)
}

type rig struct {
	d      *webgpu.Device
	bgl    *webgpu.BindGroupLayout
	pl     *webgpu.PipelineLayout
	bg     *webgpu.BindGroup
	gx, gy uint32
	split  bool
}

// submit compiles the shader for iters and dispatches it; it returns the
// error of the submit and how long it took. In split mode the objects go in
// a first frame (a one-workgroup dispatch, not timed) and the long dispatch
// is a frame of its own.
func (r *rig) submit(iters uint64) (time.Duration, error) {
	m, err := r.d.CreateShaderModule(webgpu.ShaderModuleDescriptor{Code: shader(iters)})
	if err != nil {
		return 0, err
	}
	pipe, err := r.d.CreateComputePipeline(webgpu.ComputePipelineDescriptor{Layout: r.pl, Compute: webgpu.ProgrammableStage{Module: m, EntryPoint: "main"}})
	if err != nil {
		return 0, err
	}
	dispatch := func(gx, gy uint32) (time.Duration, error) {
		rec := r.d.NewRecorder()
		p := rec.BeginComputePass()
		p.SetPipeline(pipe)
		p.SetBindGroup(0, r.bg)
		p.DispatchWorkgroups(gx, gy, 1)
		if err := p.End(); err != nil {
			return 0, err
		}
		t0 := time.Now()
		_, err := rec.Submit()
		return time.Since(t0), err
	}
	if r.split {
		if _, err := dispatch(1, 1); err != nil {
			return 0, err
		}
	}
	return dispatch(r.gx, r.gy)
}

func main() {
	// 3 s: over the 2 s hard cap, and the GPU still holds about a second of
	// the work when the device is destroyed. destroy() does not cancel work
	// the GPU already holds (go-wasmcontainers/gpu#26): the host waits for
	// it to drain before it serves anyone else.
	target := 3000.0
	split := false
	if len(os.Args) > 1 {
		if n, err := strconv.ParseFloat(os.Args[1], 64); err == nil && n > 0 {
			target = n
		}
	}
	split = len(os.Args) > 2 && os.Args[2] == "split"
	s, err := batchsink.Open()
	var be *batchsink.Error
	if errors.Is(err, batchsink.ErrUnavailable) || errors.As(err, &be) && (be.Errno == batchsink.Denied || be.Errno == batchsink.Unsupported) {
		fmt.Println("no GPU:", err)
		os.Exit(2)
	}
	must(err)
	defer s.Close()
	c := s.Caps()
	d := s.Device("hang")
	bgl, err := d.CreateBindGroupLayout(webgpu.BindGroupLayoutDescriptor{Entries: []webgpu.BindGroupLayoutEntry{
		{Binding: 0, Visibility: webgpu.ShaderStageCompute, Buffer: &webgpu.BufferBindingLayout{Type: webgpu.BufferBindingTypeStorage}}}})
	must(err)
	pl, err := d.CreatePipelineLayout(webgpu.PipelineLayoutDescriptor{BindGroupLayouts: []*webgpu.BindGroupLayout{bgl}})
	must(err)
	buf, err := d.CreateBuffer(webgpu.BufferDescriptor{Size: 4096, Usage: webgpu.BufferUsageStorage | webgpu.BufferUsageCopySrc})
	must(err)
	bg, err := d.CreateBindGroup(webgpu.BindGroupDescriptor{Layout: bgl, Entries: []webgpu.BindGroupEntry{{Binding: 0, Buffer: buf}}})
	must(err)
	r := &rig{d: d, bgl: bgl, pl: pl, bg: bg, gx: 1, gy: 1, split: split}
	maxX := c.Device.MaxComputeWorkgroupsPerDimension
	if len(os.Args) > 2 && os.Args[2] == "wide" {
		// A dispatch past the host's compute quota is refused before it
		// reaches the GPU: Refused, not Timeout, and no sanction.
		r.gx, r.gy = maxX, 16
		_, err := r.submit(1)
		ok := errors.As(err, &be) && be.Errno == batchsink.Refused
		fmt.Printf("wide dispatch %d x 16: %v (refused as expected: %v)\n", maxX, err, ok)
		if !ok {
			os.Exit(1)
		}
		return
	}

	// Calibrate: double the work (X workgroups up to the device limit, then
	// 16 in Y, then the loop) until one submit takes `stop`, well under the
	// budget it must not trip, then scale the loop to the target.
	// A submit also has a fixed latency (about 200 ms in Firefox, which
	// quantises read-back): measure it on an empty-ish shader and subtract
	// it, or the calibration would stop at the latency, not at GPU work.
	stop := 250 * time.Millisecond
	if split {
		stop = 40 * time.Millisecond // the doubling may overshoot: stay far under the budget
	}
	lat := time.Hour
	for i := 0; i < 3; i++ {
		l, err := r.submit(1)
		must(err)
		lat = min(lat, l)
	}
	iters := uint64(1 << 10)
	var took time.Duration
	// With the host's loop budget (layer 4) the doubling ends by itself: a
	// loop past the budget is cut and the submit answers Truncated. That
	// submit is the hostile one: it is what the host bounded.
	var cut error
	for {
		took, err = r.submit(iters)
		if truncated(err) {
			cut = err
			break
		}
		must(err)
		if took-lat >= stop || iters >= 1<<31 {
			break
		}
		switch {
		case r.gx < maxX:
			r.gx = min(r.gx*2, maxX)
		case r.gy < 4:
			// 65535 x 4 workgroups, counted at the device's 256-invocation
			// workgroup bound, is 67,107,840 invocations: just under the
			// host's dispatch quota (2^26). The loop makes up the rest of the
			// time: a quota on the size of the dispatch cannot bound a loop.
			r.gy *= 2
		default:
			iters *= 2
		}
	}
	work := max(float64((took - lat).Milliseconds()), 1)
	iters = uint64(float64(iters) * max(target-float64(lat.Milliseconds()), 1) / work)
	if iters == 0 {
		iters = 1
	}
	if iters > 1<<32-1 {
		iters = 1<<32 - 1
	}
	fmt.Printf("dispatch %d x %d workgroups of 64; calibration took %d ms (latency %d ms); %d iterations aim at %.0f ms (split %v)\n",
		r.gx, r.gy, took.Milliseconds(), lat.Milliseconds(), iters, target, split)

	if cut != nil {
		err = cut
	} else {
		took, err = r.submit(iters)
	}
	ms := took.Milliseconds()
	var e *batchsink.Error
	byLoop := errors.As(err, &e) && e.Errno == batchsink.Truncated
	ok := errors.As(err, &e) && e.Errno == batchsink.Timeout || byLoop
	fmt.Printf("long submit: %v after %d ms (cut as expected: %v, by the loop budget: %v)\n", err, ms, ok, byLoop)
	// The device is destroyed: every later call fails, as an error.
	_, err = r.submit(1)
	ok2 := err != nil && (errors.Is(err, webgpu.ErrDestroyed) || errors.As(err, &e) && (e.Errno == batchsink.ContextLost || e.Errno == batchsink.Revoked) || strings.Contains(err.Error(), "lost"))
	fmt.Printf("next submit: %v (device gone as expected: %v)\n", err, ok2)
	// The grant is revoked for the container's life: a new context is refused
	// with the documented errno, and so is a new connection to the module.
	_, err = batchsink.Open()
	ok3 := errors.As(err, &e) && e.Errno == batchsink.Revoked
	fmt.Printf("new context: %v (revoked as expected: %v)\n", err, ok3)
	if !ok || !ok2 || !ok3 {
		os.Exit(1)
	}
}

// truncated reports whether err is the host's loop-budget cut.
func truncated(err error) bool {
	var e *batchsink.Error
	return errors.As(err, &e) && e.Errno == batchsink.Truncated
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}
