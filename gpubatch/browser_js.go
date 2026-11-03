//go:build js

package gpubatch

import (
	"context"
	"fmt"
	"sync"
	"time"

	wg "github.com/go-gfx/webgpu"
	"github.com/go-gfx/webgpu/batch"
	"github.com/go-virtio/gpu/wgpubatch"
)

// Browser is the Provider in the browser: one WebGPU device per container,
// on the worker's navigator.gpu, bounded by a go-gfx/webgpu Quota (GPU
// memory from the container's grant, 256 pipelines, 1 MiB per WGSL module,
// the read window). The device asks for no raised limit; it enables
// "shader-f16" and "subgroups" when the adapter has them, never
// "timestamp-query". A fallback (software) adapter is refused unless
// AllowFallback.
type Browser struct {
	// AllowFallback accepts a software adapter; off by default, since its
	// work is not GPU work.
	AllowFallback bool
	// ReplayURL is where the page serves replay.js (webgpu.ReplayJS) when
	// the worker's CSP forbids the blob URL InstallReplay otherwise uses.
	ReplayURL string
	// MaxRead is the quota's read window (default DefaultLimits.MaxRead).
	MaxRead int
}

// WebGPU's default limits (the spec's table), which a device that asks for
// no raised limit has; the capset reports the smaller of these and the
// adapter's, so a compatibility-mode adapter is not overstated.
var defaultLimits = map[string]uint64{
	"maxStorageBufferBindingSize":       128 << 20,
	"maxComputeWorkgroupStorageSize":    16384,
	"maxComputeInvocationsPerWorkgroup": 256,
	"maxComputeWorkgroupsPerDimension":  65535,
	"maxStorageBuffersPerShaderStage":   8,
}

// Open implements Provider.
func (b Browser) Open(memory int64) (Device, error) {
	ctx := context.Background()
	g, err := wg.Navigator()
	if err != nil {
		return nil, err
	}
	if err := wg.InstallReplay(ctx, b.ReplayURL); err != nil {
		return nil, fmt.Errorf("gpubatch: %w", err)
	}
	ad, err := g.RequestAdapter(ctx, &wg.RequestAdapterOptions{AllowFallback: b.AllowFallback})
	if err != nil {
		return nil, err // ErrNoAdapter, ErrFallbackAdapter
	}
	var features uint32
	var req []wg.FeatureName
	for _, f := range []struct {
		name wg.FeatureName
		bit  uint32
	}{{wg.FeatureShaderF16, wgpubatch.FeatureShaderF16}, {"subgroups", wgpubatch.FeatureSubgroups}} {
		if ad.HasFeature(f.name) {
			req = append(req, f.name)
			features |= f.bit
		}
	}
	maxRead := b.MaxRead
	if maxRead <= 0 {
		maxRead = int(DefaultLimits.MaxRead)
	}
	q := &wg.Quota{MemoryBytes: memory, Pipelines: 256, ShaderBytes: 1 << 20, ReadBytes: maxRead, Features: req}
	dev, err := ad.RequestDevice(ctx, &wg.DeviceDescriptor{Label: "wasi_webgpu_batch", RequiredFeatures: req, Quota: q})
	if err != nil {
		return nil, err
	}
	lim := func(name string) uint64 {
		v, ok := ad.Limit(name)
		if !ok {
			return defaultLimits[name]
		}
		return min(v, defaultLimits[name])
	}
	return &browserDevice{dev: dev, features: features, limits: wgpubatch.DeviceLimits{
		MaxStorageBufferBindingSize:       lim("maxStorageBufferBindingSize"),
		MaxComputeWorkgroupStorageSize:    uint32(lim("maxComputeWorkgroupStorageSize")),
		MaxComputeInvocationsPerWorkgroup: uint32(lim("maxComputeInvocationsPerWorkgroup")),
		MaxComputeWorkgroupsPerDimension:  uint32(lim("maxComputeWorkgroupsPerDimension")),
		MaxStorageBuffersPerShaderStage:   uint32(lim("maxStorageBuffersPerShaderStage")),
	}}, nil
}

type browserDevice struct {
	dev      *wg.Device
	features uint32
	limits   wgpubatch.DeviceLimits
}

func (d *browserDevice) Caps() (uint32, wgpubatch.DeviceLimits) { return d.features, d.limits }
func (d *browserDevice) Backend() batch.Backend                 { return batch.NewDeviceBackend(d.dev) }
func (d *browserDevice) Destroy()                               { d.dev.Destroy() }

// drainProbe is the device Drain submits to: one per page, opened at the
// first sanction and kept. Its work is a single workgroup; it only has to
// stand in the GPU's queue behind whatever a destroyed device left there.
var drainProbe struct {
	sync.Mutex
	dev *wg.Device
	rig *drainRig
}

type drainRig struct {
	pipe *wg.ComputePipeline
	bg   *wg.BindGroup
}

const drainWGSL = `
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(1) fn main() { out[0] = out[0] + 1u; }`

// Drain implements Drainer: WebGPU's destroy() does not cancel what the GPU
// already holds (the spec's own text: outstanding asynchronous operations
// fail; nothing says submitted work stops), so after a sanction the host
// waits for a trivial submit of its own to complete, which it does once the
// destroyed device's work has left the GPU queue.
func (b Browser) Drain(timeout time.Duration) error {
	drainProbe.Lock()
	defer drainProbe.Unlock()
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	if drainProbe.dev == nil {
		g, err := wg.Navigator()
		if err != nil {
			return err
		}
		ad, err := g.RequestAdapter(ctx, &wg.RequestAdapterOptions{AllowFallback: b.AllowFallback})
		if err != nil {
			return err
		}
		dev, err := ad.RequestDevice(ctx, &wg.DeviceDescriptor{Label: "wasi_webgpu_batch drain"})
		if err != nil {
			return err
		}
		bgl, err := dev.CreateBindGroupLayout(wg.BindGroupLayoutDescriptor{Entries: []wg.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wg.ShaderStageCompute, Buffer: &wg.BufferBindingLayout{Type: wg.BufferBindingTypeStorage}}}})
		if err != nil {
			return err
		}
		pl, err := dev.CreatePipelineLayout(wg.PipelineLayoutDescriptor{BindGroupLayouts: []*wg.BindGroupLayout{bgl}})
		if err != nil {
			return err
		}
		m, err := dev.CreateShaderModule(wg.ShaderModuleDescriptor{Code: drainWGSL})
		if err != nil {
			return err
		}
		pipe, err := dev.CreateComputePipeline(wg.ComputePipelineDescriptor{Layout: pl, Compute: wg.ProgrammableStage{Module: m, EntryPoint: "main"}})
		if err != nil {
			return err
		}
		buf, err := dev.CreateBuffer(wg.BufferDescriptor{Size: 16, Usage: wg.BufferUsageStorage})
		if err != nil {
			return err
		}
		bg, err := dev.CreateBindGroup(wg.BindGroupDescriptor{Layout: bgl, Entries: []wg.BindGroupEntry{{Binding: 0, Buffer: buf}}})
		if err != nil {
			return err
		}
		drainProbe.dev, drainProbe.rig = dev, &drainRig{pipe: pipe, bg: bg}
	}
	rec := drainProbe.dev.NewRecorder()
	p := rec.BeginComputePass()
	p.SetPipeline(drainProbe.rig.pipe)
	p.SetBindGroup(0, drainProbe.rig.bg)
	p.DispatchWorkgroups(1, 1, 1)
	if err := p.End(); err != nil {
		return err
	}
	seq, err := rec.Submit()
	if err != nil {
		return err
	}
	return drainProbe.dev.WaitSubmitted(ctx, seq)
}
