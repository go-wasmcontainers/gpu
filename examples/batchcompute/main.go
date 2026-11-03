//go:build !js

// Command batchcompute is a WASI container that computes on the GPU through
// the batched command stream (wasi_webgpu_batch) and checks every element
// it reads back.
//
//	GOOS=wasip1 GOARCH=wasm go build -o batchcompute.wasm ./examples/batchcompute
//	wctr run --gpu [--gpu-memory=SIZE] IMAGE [MODE]
//
// It records a compute shader that writes known values into a storage
// buffer, copies the buffer to a MAP_READ buffer and reads it back in the
// same submit: one crossing into the host. MODE:
//
//	(none)     1 MiB and 4 KiB read-backs, every element checked; exit 0
//	refusals   frames the host must refuse (an out-of-range write, a read
//	           outside the window, a malformed frame, an oversized frame,
//	           an allocation past the GPU memory quota); prints one line
//	           each; exit 0 when every one is refused as expected
//	shaders    WGSL the GPU refuses (a reserved word, a syntax error) must fail with
//	           errno InvalidShader and the compiler's message; a valid source is the
//	           control; exit 0 when every case is as expected
//	quota      allocates 64 MiB buffers through the remote device until
//	           the guest-side quota refuses
//	bench N    N interleaved rounds of the per-submit round trip at 4 KiB
//	           and 1 MiB, read-only and compute+copy+read; prints medians
//
// Exit 2 (printing "no GPU") when the container has no GPU: no --gpu, no
// WebGPU where it runs, or only a software adapter.
package main

import (
	"errors"
	"fmt"
	"os"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/go-gfx/webgpu"
	"github.com/go-gfx/webgpu/wire"
	"github.com/go-virtio/gpu/wgpubatch"
	"github.com/go-wasmcontainers/gpu/wasigpu/batchsink"
)

const shader = `
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < arrayLength(&out)) { out[id.x] = id.x * 2654435761u + 7u; }
}`

func expected(i uint32) uint32 { return i*2654435761 + 7 }

func fail(err error) {
	fmt.Fprintln(os.Stderr, "error:", err)
	os.Exit(1)
}

func main() {
	s, err := batchsink.Open()
	var be *batchsink.Error
	if errors.Is(err, batchsink.ErrUnavailable) || errors.As(err, &be) && (be.Errno == batchsink.Denied || be.Errno == batchsink.Unsupported) {
		fmt.Println("no GPU:", err)
		os.Exit(2)
	}
	if err != nil {
		fail(err)
	}
	defer s.Close()
	c := s.Caps()
	fmt.Printf("caps: stream v%d, max_frame %d, max_objects %d, max_memory %d, f16 %v, subgroups %v, workgroups/dim %d\n",
		c.StreamVersion, c.MaxFrame, c.MaxObjects, c.MaxMemory,
		c.Features&wgpubatch.FeatureShaderF16 != 0, c.Features&wgpubatch.FeatureSubgroups != 0, c.Device.MaxComputeWorkgroupsPerDimension)
	mode := ""
	if len(os.Args) > 1 {
		mode = os.Args[1]
	}
	switch mode {
	case "":
		compute(s)
	case "refusals":
		refusals(s)
	case "quota":
		quota(s)
	case "shaders":
		shaders()
	case "bench":
		rounds := 30
		if len(os.Args) > 2 {
			if n, err := strconv.Atoi(os.Args[2]); err == nil && n > 0 {
				rounds = n
			}
		}
		bench(s, rounds)
	default:
		fail(fmt.Errorf("unknown mode %q", mode))
	}
}

// rig is the compute pipeline, a storage buffer and a MAP_READ buffer of n
// words.
type rig struct {
	d        *webgpu.Device
	pipe     *webgpu.ComputePipeline
	bg       *webgpu.BindGroup
	src, dst *webgpu.Buffer
	n        uint64
}

func newRig(d *webgpu.Device, n uint64) *rig {
	m, err := d.CreateShaderModule(webgpu.ShaderModuleDescriptor{Code: shader})
	must(err)
	bgl, err := d.CreateBindGroupLayout(webgpu.BindGroupLayoutDescriptor{Entries: []webgpu.BindGroupLayoutEntry{
		{Binding: 0, Visibility: webgpu.ShaderStageCompute, Buffer: &webgpu.BufferBindingLayout{Type: webgpu.BufferBindingTypeStorage}}}})
	must(err)
	pl, err := d.CreatePipelineLayout(webgpu.PipelineLayoutDescriptor{BindGroupLayouts: []*webgpu.BindGroupLayout{bgl}})
	must(err)
	r := &rig{d: d, n: n}
	r.src, err = d.CreateBuffer(webgpu.BufferDescriptor{Size: 4 * n, Usage: webgpu.BufferUsageStorage | webgpu.BufferUsageCopySrc})
	must(err)
	r.dst, err = d.CreateBuffer(webgpu.BufferDescriptor{Size: 4 * n, Usage: webgpu.BufferUsageMapRead | webgpu.BufferUsageCopyDst})
	must(err)
	r.pipe, err = d.CreateComputePipeline(webgpu.ComputePipelineDescriptor{Layout: pl, Compute: webgpu.ProgrammableStage{Module: m, EntryPoint: "main"}})
	must(err)
	r.bg, err = d.CreateBindGroup(webgpu.BindGroupDescriptor{Layout: bgl, Entries: []webgpu.BindGroupEntry{{Binding: 0, Buffer: r.src}}})
	must(err)
	return r
}

func must(err error) {
	if err != nil {
		fail(err)
	}
}

// run records one submit: optionally the dispatch and the copy, then a read
// of the first size bytes of the MAP_READ buffer.
func (r *rig) run(compute bool, size uint64) []byte {
	rec := r.d.NewRecorder()
	if compute {
		p := rec.BeginComputePass()
		p.SetPipeline(r.pipe)
		p.SetBindGroup(0, r.bg)
		p.DispatchWorkgroups(uint32((r.n+63)/64), 1, 1)
		must(p.End())
		rec.CopyBufferToBuffer(r.src, 0, r.dst, 0, 4*r.n)
	}
	rb := rec.ReadBuffer(r.dst, 0, size)
	_, err := rec.Submit()
	must(err)
	b, err := rb.Bytes()
	must(err)
	return b
}

func compute(s *batchsink.Sink) {
	d := s.Device("batchcompute")
	for _, size := range []uint64{1 << 20, 4 << 10} {
		r := newRig(d, size/4)
		before := s.Submits()
		got := r.run(true, size)
		bad := 0
		for i := 0; i < len(got)/4; i++ {
			v := uint32(got[4*i]) | uint32(got[4*i+1])<<8 | uint32(got[4*i+2])<<16 | uint32(got[4*i+3])<<24
			if v != expected(uint32(i)) {
				if bad < 5 {
					fmt.Printf("element %d: %#x, want %#x\n", i, v, expected(uint32(i)))
				}
				bad++
			}
		}
		if bad > 0 {
			fmt.Printf("%d of %d elements differ\n", bad, len(got)/4)
			os.Exit(1)
		}
		fmt.Printf("%d KiB: %d elements match the CPU, %d host crossing(s) for the compute and the read\n", size>>10, len(got)/4, s.Submits()-before)
	}
}

func quota(s *batchsink.Sink) {
	d := s.Device("batchcompute-quota")
	for i := 0; i < 64; i++ {
		if _, err := d.CreateBuffer(webgpu.BufferDescriptor{Size: 64 << 20, Usage: webgpu.BufferUsageStorage}); err != nil {
			fmt.Printf("buffer %d refused: %v\n", i, err)
			return
		}
		fmt.Printf("buffer %d ok\n", i)
	}
	fmt.Println("no quota")
	os.Exit(1)
}

// frame builds a frame by hand, as a hostile guest would, and sends it
// through the sink's raw path (the recorder would refuse most of these
// itself).
func frame(max uint32, build func(*wgpubatch.Encoder)) []byte {
	e := wgpubatch.NewEncoder(max)
	build(e)
	f, err := e.Frame()
	must(err)
	return slices.Clone(f)
}

func bufferDesc(size uint64, usage webgpu.BufferUsage) (uint32, []byte) {
	kind, b, err := webgpu.EncodeDescriptor(webgpu.BufferDescriptor{Size: size, Usage: usage})
	must(err)
	return uint32(kind), b
}

func refusals(s *batchsink.Sink) {
	c := s.Caps()
	type tc struct {
		name  string
		frame func() []byte
		win   int
		want  batchsink.Errno
	}
	cases := []tc{
		{"out-of-range write", func() []byte {
			k, d := bufferDesc(64, webgpu.BufferUsageCopyDst)
			return frame(c.MaxFrame, func(e *wgpubatch.Encoder) {
				e.Create(1, k, d)
				// WriteBuffer id 1, offset 60, dataOffset 0, 16 bytes: past the end.
				e.Stream([]uint32{uint32(wire.WriteBuffer), 1, 60, 0, 0, 16}, make([]byte, 16))
			})
		}, 0, batchsink.Refused},
		{"read outside the window", func() []byte {
			k, d := bufferDesc(64, webgpu.BufferUsageMapRead|webgpu.BufferUsageCopyDst)
			return frame(c.MaxFrame, func(e *wgpubatch.Encoder) {
				e.Create(1, k, d)
				// ReadBuffer id 1, offset 0, 16 bytes into window offset 8 of a 16-byte window.
				e.Stream([]uint32{uint32(wire.ReadBuffer), 1, 0, 0, 16, 8}, nil)
			})
		}, 16, batchsink.Refused},
		{"dispatch above the limit", func() []byte {
			return frame(c.MaxFrame, func(e *wgpubatch.Encoder) {
				e.Stream([]uint32{uint32(wire.BeginComputePass), uint32(wire.Dispatch), c.Device.MaxComputeWorkgroupsPerDimension + 1, 1, 1, uint32(wire.EndPass)}, nil)
			})
		}, 0, batchsink.Refused},
		{"id of another context", func() []byte {
			return frame(c.MaxFrame, func(e *wgpubatch.Encoder) {
				e.Stream([]uint32{uint32(wire.WriteBuffer), 77, 0, 0, 0, 4}, make([]byte, 4))
			})
		}, 0, batchsink.Refused},
		{"bad magic", func() []byte { return make([]byte, 64) }, 0, batchsink.Refused},
		{"truncated frame", func() []byte {
			k, d := bufferDesc(64, webgpu.BufferUsageCopyDst)
			f := frame(c.MaxFrame, func(e *wgpubatch.Encoder) { e.Create(1, k, d) })
			return f[:len(f)-4]
		}, 0, batchsink.Refused},
		{"frame above max_frame", func() []byte { return make([]byte, c.MaxFrame+4) }, 0, batchsink.TooLarge},
		{"allocation past the quota", func() []byte {
			k, d := bufferDesc(c.MaxMemory+4096, webgpu.BufferUsageStorage)
			return frame(c.MaxFrame, func(e *wgpubatch.Encoder) { e.Create(1, k, d) })
		}, 0, batchsink.QuotaExceeded},
	}
	bad := 0
	for _, t := range cases {
		// Each case on its own context: a refusal loses the context.
		cs, err := batchsink.Open()
		must(err)
		err = cs.Raw(t.frame(), make([]byte, t.win))
		var e *batchsink.Error
		ok := errors.As(err, &e) && e.Errno == t.want
		fmt.Printf("%s: %v (as expected: %v)\n", t.name, err, ok)
		if !ok {
			bad++
		}
		cs.Close()
	}
	if bad > 0 {
		os.Exit(1)
	}
	fmt.Printf("%d frames refused\n", len(cases))
}

func median(v []float64) float64 {
	slices.Sort(v)
	return v[len(v)/2]
}

func bench(s *batchsink.Sink, rounds int) {
	d := s.Device("batchcompute-bench")
	small, large := newRig(d, 1<<10), newRig(d, 1<<18)
	type cfg struct {
		name    string
		r       *rig
		compute bool
		size    uint64
	}
	cfgs := []cfg{
		{"read only 4 KiB", small, false, 4 << 10}, {"read only 1 MiB", large, false, 1 << 20},
		{"compute+copy+read 4 KiB", small, true, 4 << 10}, {"compute+copy+read 1 MiB", large, true, 1 << 20},
	}
	times := make([][]float64, len(cfgs))
	for _, c := range cfgs { // warm up: pipelines, replay.js, the first map
		c.r.run(c.compute, c.size)
	}
	before := s.Submits()
	for i := 0; i < rounds; i++ {
		for j, c := range cfgs { // interleaved: every round runs every configuration
			t0 := time.Now()
			c.r.run(c.compute, c.size)
			times[j] = append(times[j], float64(time.Since(t0).Nanoseconds())/1000)
		}
	}
	for j, c := range cfgs {
		fmt.Printf("bench %s: median %.1f us, min %.1f, max %.1f (%d rounds)\n", c.name, median(slices.Clone(times[j])), slices.Min(times[j]), slices.Max(times[j]), rounds)
	}
	fmt.Printf("bench crossings: %d for %d submits\n", s.Submits()-before, rounds*len(cfgs))
}

// shaders sends invalid WGSL and checks that the failure is loud: the submit
// that carries the module fails with InvalidShader and the error holds the
// compiler's message; the valid control passes.
func shaders() {
	cases := []struct {
		name, code, want string
		ok               bool
	}{
		{"valid control", shader, "", true},
		{"reserved word", "@group(0) @binding(0) var<storage, read> meta: array<u32>;\n@compute @workgroup_size(1) fn main() { _ = meta[0]; }", "meta", false},
		{"syntax error", "@compute @workgroup_size(1) fn main( { }", ")", false},
	}
	bad := 0
	for _, c := range cases {
		s, err := batchsink.Open()
		must(err)
		d := s.Device("shaders")
		_, err = d.CreateShaderModule(webgpu.ShaderModuleDescriptor{Code: c.code})
		must(err) // creation travels with the next submit
		err = s.Flush()
		var be *batchsink.Error
		loud := errors.As(err, &be) && be.Errno == batchsink.InvalidShader && be.Msg != "" && (c.ok || strings.Contains(be.Msg, c.want))
		fmt.Printf("%s: %v (as expected: %v)\n", c.name, err, loud == !c.ok && (err == nil) == c.ok)
		if loud == c.ok || (err == nil) != c.ok {
			bad++
		}
		s.Close()
	}
	if bad > 0 {
		os.Exit(1)
	}
	fmt.Println("3 shader cases as expected")
}
