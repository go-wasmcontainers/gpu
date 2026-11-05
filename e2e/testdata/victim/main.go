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
	const n = 1024
	var s *batchsink.Sink
	var d *webgpu.Device
	var pipe *webgpu.ComputePipeline
	var bg *webgpu.BindGroup
	var src, dst *webgpu.Buffer
	// open makes the victim's context and objects; it is also what the victim
	// does when the host reset its device (errno ContextLost: another
	// container's device was destroyed by the browser, go-wasmcontainers/gpu#32).
	open := func() error {
		var err error
		if s, err = batchsink.Open(); err != nil {
			return err
		}
		d = s.Device("victim")
		m, err := d.CreateShaderModule(webgpu.ShaderModuleDescriptor{Code: shader})
		if err != nil {
			return err
		}
		bgl, err := d.CreateBindGroupLayout(webgpu.BindGroupLayoutDescriptor{Entries: []webgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: webgpu.ShaderStageCompute, Buffer: &webgpu.BufferBindingLayout{Type: webgpu.BufferBindingTypeStorage}}}})
		if err != nil {
			return err
		}
		pl, err := d.CreatePipelineLayout(webgpu.PipelineLayoutDescriptor{BindGroupLayouts: []*webgpu.BindGroupLayout{bgl}})
		if err != nil {
			return err
		}
		if src, err = d.CreateBuffer(webgpu.BufferDescriptor{Size: 4 * n, Usage: webgpu.BufferUsageStorage | webgpu.BufferUsageCopySrc}); err != nil {
			return err
		}
		if dst, err = d.CreateBuffer(webgpu.BufferDescriptor{Size: 4 * n, Usage: webgpu.BufferUsageMapRead | webgpu.BufferUsageCopyDst}); err != nil {
			return err
		}
		if pipe, err = d.CreateComputePipeline(webgpu.ComputePipelineDescriptor{Layout: pl, Compute: webgpu.ProgrammableStage{Module: m, EntryPoint: "main"}}); err != nil {
			return err
		}
		if bg, err = d.CreateBindGroup(webgpu.BindGroupDescriptor{Layout: bgl, Entries: []webgpu.BindGroupEntry{{Binding: 0, Buffer: src}}}); err != nil {
			return err
		}
		return s.Flush()
	}
	err := open()
	var be *batchsink.Error
	if errors.Is(err, batchsink.ErrUnavailable) || errors.As(err, &be) && (be.Errno == batchsink.Denied || be.Errno == batchsink.Unsupported) {
		fmt.Println("no GPU:", err)
		os.Exit(2)
	}
	must(err)
	defer func() { s.Close() }()

	var lat []float64
	bad, resets := 0, 0
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
		if errors.As(err, &be) && be.Errno == batchsink.ContextLost {
			// the host reset the device (not a sanction): start again
			resets++
			fmt.Printf("victim: submit %d answered context-lost after %.1f ms: opening a new context\n", len(lat), ms)
			s.Close()
			for try := 0; try < 240; try++ { // Safari answers no adapter for a while after its watchdog
				if err = open(); err == nil {
					break
				}
				time.Sleep(200 * time.Millisecond)
			}
			if err != nil {
				fmt.Printf("victim: reopening failed: %v\n", err)
				bad++
			}
			continue
		}
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
	fmt.Printf("victim: %d submits, median %.1f ms, max %.1f ms, errors %d, resets %d\n", len(lat), sorted[len(sorted)/2], sorted[len(sorted)-1], bad, resets)
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
