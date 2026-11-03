//go:build !js

// Command hang is the guest of the GPU-time deadline e2e: it submits a
// compute shader that runs far longer than the host's 2 s of GPU time, sized
// from the adapter's limits and from a calibration on the adapter itself
// (every workgroup the device allows in X, 16 in Y; the per-invocation loop
// is scaled until the shader needs about TARGET ms), and reports what the
// host answered.
//
//	hang [TARGET_MS]
//
// The target stays a few seconds: the host destroys the device at 2 s, but
// the GPU finishes the work it already holds, and the containers started
// after this one wait for it.
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
}

// submit compiles the shader for iters and dispatches it; it returns the
// error of the submit and how long it took.
func (r *rig) submit(iters uint64) (time.Duration, error) {
	m, err := r.d.CreateShaderModule(webgpu.ShaderModuleDescriptor{Code: shader(iters)})
	if err != nil {
		return 0, err
	}
	pipe, err := r.d.CreateComputePipeline(webgpu.ComputePipelineDescriptor{Layout: r.pl, Compute: webgpu.ProgrammableStage{Module: m, EntryPoint: "main"}})
	if err != nil {
		return 0, err
	}
	rec := r.d.NewRecorder()
	p := rec.BeginComputePass()
	p.SetPipeline(pipe)
	p.SetBindGroup(0, r.bg)
	p.DispatchWorkgroups(r.gx, r.gy, 1)
	if err := p.End(); err != nil {
		return 0, err
	}
	t0 := time.Now()
	_, err = rec.Submit()
	return time.Since(t0), err
}

func main() {
	target := 5000.0
	if len(os.Args) > 1 {
		if n, err := strconv.ParseFloat(os.Args[1], 64); err == nil && n > 0 {
			target = n
		}
	}
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
	r := &rig{d: d, bgl: bgl, pl: pl, bg: bg, gx: c.Device.MaxComputeWorkgroupsPerDimension, gy: 16}

	// Calibrate: double the loop until one submit takes 250 ms (well under
	// the 2 s deadline), then scale to the target.
	iters := uint64(1 << 10)
	var took time.Duration
	for {
		took, err = r.submit(iters)
		must(err)
		if took >= 250*time.Millisecond || iters >= 1<<31 {
			break
		}
		iters *= 2
	}
	iters = uint64(float64(iters) * target / float64(took.Milliseconds()+1))
	if iters > 1<<32-1 {
		iters = 1<<32 - 1
	}
	fmt.Printf("dispatch %d x %d workgroups of 64; calibration took %d ms; %d iterations aim at %.0f ms\n",
		r.gx, r.gy, took.Milliseconds(), iters, target)

	took, err = r.submit(iters)
	ms := took.Milliseconds()
	var e *batchsink.Error
	ok := errors.As(err, &e) && e.Errno == batchsink.Timeout
	fmt.Printf("long submit: %v after %d ms (timeout as expected: %v)\n", err, ms, ok)
	// The device is destroyed: every later call fails, as an error.
	_, err = r.submit(1)
	ok2 := err != nil && (errors.Is(err, webgpu.ErrDestroyed) || errors.As(err, &e) && e.Errno == batchsink.ContextLost || strings.Contains(err.Error(), "lost"))
	fmt.Printf("next submit: %v (device gone as expected: %v)\n", err, ok2)
	if !ok || !ok2 {
		os.Exit(1)
	}
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}
