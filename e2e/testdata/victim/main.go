//go:build !js

// Command victim is the innocent guest of the GPU time budget e2e: it makes
// short submits (a 4 KiB compute, copy and read-back, as batchcompute does)
// for DURATION_MS, a pause between them, and reports the latency each one
// saw from the guest, so a test can tell what another container's runaway
// shader cost it.
//
//	victim [DURATION_MS [PAUSE_MS]]
//
// It prints "victim: N submits, median M ms, max X ms, errors E" and every
// submit slower than 100 ms; exit 1 if any submit failed or read back wrong.
package main

import (
	"errors"
	"fmt"
	"os"
	"slices"
	"strconv"
	"time"

	"github.com/go-gfx/webgpu"
	"github.com/go-wasmcontainers/gpu/wasigpu/batchsink"
)

const shader = `
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < arrayLength(&out)) { out[id.x] = id.x * 2654435761u + 7u; }
}`

func main() {
	dur, pause := 6000*time.Millisecond, 20*time.Millisecond
	if len(os.Args) > 1 {
		if n, err := strconv.Atoi(os.Args[1]); err == nil && n > 0 {
			dur = time.Duration(n) * time.Millisecond
		}
	}
	if len(os.Args) > 2 {
		if n, err := strconv.Atoi(os.Args[2]); err == nil && n >= 0 {
			pause = time.Duration(n) * time.Millisecond
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
	d := s.Device("victim")
	const n = 1024
	m, err := d.CreateShaderModule(webgpu.ShaderModuleDescriptor{Code: shader})
	must(err)
	bgl, err := d.CreateBindGroupLayout(webgpu.BindGroupLayoutDescriptor{Entries: []webgpu.BindGroupLayoutEntry{
		{Binding: 0, Visibility: webgpu.ShaderStageCompute, Buffer: &webgpu.BufferBindingLayout{Type: webgpu.BufferBindingTypeStorage}}}})
	must(err)
	pl, err := d.CreatePipelineLayout(webgpu.PipelineLayoutDescriptor{BindGroupLayouts: []*webgpu.BindGroupLayout{bgl}})
	must(err)
	src, err := d.CreateBuffer(webgpu.BufferDescriptor{Size: 4 * n, Usage: webgpu.BufferUsageStorage | webgpu.BufferUsageCopySrc})
	must(err)
	dst, err := d.CreateBuffer(webgpu.BufferDescriptor{Size: 4 * n, Usage: webgpu.BufferUsageMapRead | webgpu.BufferUsageCopyDst})
	must(err)
	pipe, err := d.CreateComputePipeline(webgpu.ComputePipelineDescriptor{Layout: pl, Compute: webgpu.ProgrammableStage{Module: m, EntryPoint: "main"}})
	must(err)
	bg, err := d.CreateBindGroup(webgpu.BindGroupDescriptor{Layout: bgl, Entries: []webgpu.BindGroupEntry{{Binding: 0, Buffer: src}}})
	must(err)

	var lat []float64
	bad := 0
	for start := time.Now(); time.Since(start) < dur; time.Sleep(pause) {
		rec := d.NewRecorder()
		p := rec.BeginComputePass()
		p.SetPipeline(pipe)
		p.SetBindGroup(0, bg)
		p.DispatchWorkgroups(n/64, 1, 1)
		must(p.End())
		rec.CopyBufferToBuffer(src, 0, dst, 0, 4*n)
		rb := rec.ReadBuffer(dst, 0, 4*n)
		t0 := time.Now()
		_, err := rec.Submit()
		var b []byte
		if err == nil {
			b, err = rb.Bytes()
		}
		ms := float64(time.Since(t0).Microseconds()) / 1000
		lat = append(lat, ms)
		if err != nil {
			fmt.Printf("victim: submit %d failed after %.1f ms: %v\n", len(lat), ms, err)
			bad++
			continue
		}
		for i := uint32(0); i < n; i++ {
			if got := uint32(b[4*i]) | uint32(b[4*i+1])<<8 | uint32(b[4*i+2])<<16 | uint32(b[4*i+3])<<24; got != i*2654435761+7 {
				fmt.Printf("victim: element %d is %d\n", i, got)
				bad++
				break
			}
		}
		if ms > 100 {
			fmt.Printf("victim: submit %d took %.1f ms (%.0f ms into the run)\n", len(lat), ms, float64(time.Since(start).Milliseconds()))
		}
	}
	sorted := slices.Clone(lat)
	slices.Sort(sorted)
	fmt.Printf("victim: %d submits, median %.1f ms, max %.1f ms, errors %d\n", len(lat), sorted[len(sorted)/2], sorted[len(sorted)-1], bad)
	if bad > 0 {
		os.Exit(1)
	}
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}
