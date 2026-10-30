//go:build js

package webgpu

import (
	stdctx "context" // this package has a type named context
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"sync"

	wg "github.com/go-gfx/webgpu"
)

// gpuDevice runs the kernels on a WebGPU device (github.com/go-gfx/webgpu).
// The device's wg.Quota bounds the bytes it holds at once, readback
// staging included: the graph's GPU memory quota.
type gpuDevice struct {
	d    *wg.Device
	info wg.AdapterInfo

	mu        sync.Mutex
	pipelines map[string]*wg.ComputePipeline
}

// newGPUDevice requests an adapter and a device whose memory is bounded
// by limit. A fallback adapter is refused unless allowFallback.
func newGPUDevice(allowFallback bool, limit int64) (*gpuDevice, error) {
	g, err := wg.Navigator()
	if err != nil {
		return nil, err
	}
	ctx := stdctx.Background()
	a, err := g.RequestAdapter(ctx, &wg.RequestAdapterOptions{AllowFallback: allowFallback})
	if err != nil {
		return nil, err
	}
	d, err := a.RequestDevice(ctx, &wg.DeviceDescriptor{Label: "wasi-nn", Quota: &wg.Quota{MemoryBytes: limit}})
	if err != nil {
		return nil, err
	}
	return &gpuDevice{d: d, info: a.Info(), pipelines: map[string]*wg.ComputePipeline{}}, nil
}

func (g *gpuDevice) lost() bool {
	select {
	case <-g.d.Lost():
		return true
	default:
		return false
	}
}

type gbuf struct{ b *wg.Buffer }

func f32bytes(f []float32) []byte {
	b := make([]byte, 4*len(f))
	for i, v := range f {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(v))
	}
	return b
}

const storage = wg.BufferUsageStorage | wg.BufferUsageCopySrc | wg.BufferUsageCopyDst

func (g *gpuDevice) buffer(size int64) (*wg.Buffer, error) {
	return g.create(wg.BufferDescriptor{Size: uint64(max(4, size)), Usage: storage})
}

// create makes a buffer; a quota refusal keeps the "GPU memory quota"
// message, whatever the buffer is for.
func (g *gpuDevice) create(desc wg.BufferDescriptor) (*wg.Buffer, error) {
	b, err := g.d.CreateBuffer(desc)
	var qe *wg.QuotaError
	if errors.As(err, &qe) {
		return nil, fmt.Errorf("GPU memory quota of %d bytes exceeded", g.d.Quota().MemoryBytes)
	}
	return b, err
}

func (g *gpuDevice) upload(f []float32) (buf, error) {
	b, err := g.buffer(int64(4 * len(f)))
	if err != nil {
		return nil, err
	}
	if len(f) > 0 {
		if err := g.d.Queue().WriteBuffer(b, 0, f32bytes(f)); err != nil {
			b.Destroy()
			return nil, err
		}
	}
	return &gbuf{b}, nil
}

func (g *gpuDevice) alloc(n int64) (buf, error) {
	b, err := g.buffer(4 * n)
	if err != nil {
		return nil, err
	}
	return &gbuf{b}, nil
}

func (g *gpuDevice) release(b buf) { b.(*gbuf).b.Destroy() }

// read copies n floats out: a copy to a mappable buffer, mapAsync,
// getMappedRange.
func (g *gpuDevice) read(b buf, n int64) ([]float32, error) {
	size := uint64(max(4, 4*n))
	staging, err := g.create(wg.BufferDescriptor{Size: size, Usage: wg.BufferUsageMapRead | wg.BufferUsageCopyDst})
	if err != nil {
		return nil, err
	}
	defer staging.Destroy()
	enc, err := g.d.CreateCommandEncoder("read")
	if err != nil {
		return nil, err
	}
	if err := enc.CopyBufferToBuffer(b.(*gbuf).b, 0, staging, 0, size); err != nil {
		return nil, err
	}
	cb, err := enc.Finish()
	if err != nil {
		return nil, err
	}
	if err := g.d.Queue().Submit(cb); err != nil {
		return nil, err
	}
	if err := staging.MapAsync(stdctx.Background(), wg.MapModeRead, 0, size); err != nil {
		return nil, err
	}
	raw := make([]byte, size)
	if err := staging.ReadMapped(0, raw); err != nil {
		return nil, err
	}
	staging.Unmap()
	f := make([]float32, n)
	for i := range f {
		f[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return f, nil
}

// pipeline is the compute pipeline of a WGSL module (entry point "main",
// layout "auto"), compiled once per source.
func (g *gpuDevice) pipeline(wgsl string) (*wg.ComputePipeline, error) {
	g.mu.Lock()
	p, ok := g.pipelines[wgsl]
	g.mu.Unlock()
	if ok {
		return p, nil
	}
	mod, err := g.d.CreateShaderModule(wg.ShaderModuleDescriptor{Code: wgsl})
	if err != nil {
		return nil, err
	}
	p, err = g.d.CreateComputePipelineAsync(stdctx.Background(), wg.ComputePipelineDescriptor{
		Compute: wg.ProgrammableStage{Module: mod, EntryPoint: "main"},
	})
	if err != nil {
		return nil, fmt.Errorf("webgpu: pipeline: %w", err)
	}
	g.mu.Lock()
	g.pipelines[wgsl] = p
	g.mu.Unlock()
	return p, nil
}

func (g *gpuDevice) run(kernel string, n int64, prm []uint32, bufs ...buf) error {
	src, ok := kernelSource(kernel)
	if !ok {
		return fmt.Errorf("no kernel %q", kernel)
	}
	p, err := g.pipeline(src)
	if err != nil {
		return err
	}
	pb := make([]byte, 4*max(len(prm), 1))
	for i, w := range prm {
		binary.LittleEndian.PutUint32(pb[i*4:], w)
	}
	pbuf, err := g.buffer(int64(len(pb)))
	if err != nil {
		return err
	}
	defer pbuf.Destroy()
	if err := g.d.Queue().WriteBuffer(pbuf, 0, pb); err != nil {
		return err
	}
	if n <= 0 {
		return nil
	}
	layout, err := p.GetBindGroupLayout(0)
	if err != nil {
		return err
	}
	entries := []wg.BindGroupEntry{{Binding: 0, Buffer: pbuf}}
	for i, b := range bufs {
		entries = append(entries, wg.BindGroupEntry{Binding: uint32(i + 1), Buffer: b.(*gbuf).b})
	}
	group, err := g.d.CreateBindGroup(wg.BindGroupDescriptor{Layout: layout, Entries: entries})
	if err != nil {
		return err
	}
	enc, err := g.d.CreateCommandEncoder(kernel)
	if err != nil {
		return err
	}
	pass, err := enc.BeginComputePass(kernel)
	if err != nil {
		return err
	}
	pass.SetPipeline(p)
	pass.SetBindGroup(0, group)
	// 64 invocations per workgroup, laid over x and y when n exceeds one
	// dimension's limit (kernels.go: index is x + y * 65535 * 64).
	groups := (n + 63) / 64
	x, y := groups, int64(1)
	if x > 65535 {
		y = (x + 65534) / 65535
		x = 65535
	}
	pass.DispatchWorkgroups(uint32(x), uint32(y), 1)
	if err := pass.End(); err != nil {
		return err
	}
	cb, err := enc.Finish()
	if err != nil {
		return err
	}
	return g.d.Queue().Submit(cb)
}
