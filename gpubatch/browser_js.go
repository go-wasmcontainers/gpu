//go:build js

package gpubatch

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
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
	// LoopWork is the loop budget's work per dispatch, in invocation-
	// iterations (see batch.LoopBudget): a dispatch's loops get
	// LoopWork/invocations iterations, rounded down to a power of two, at
	// least LoopMin. 0 is DefaultLoopWork.
	LoopWork uint64
	// LoopMin is the smallest per-loop iteration budget (0: 1024).
	LoopMin uint32
	// NoLoopBudget turns layer 4 off: guest WGSL reaches the browser as the
	// guest wrote it, bounded by the time budget only. For measurements.
	NoLoopBudget bool
}

// DefaultLoopWork is Browser.LoopWork's default.
const DefaultLoopWork = 1 << 36

// WebGPU's default limits (the spec's table), which a device that asks for
// no raised limit has; the capset reports the smaller of these and the
// adapter's, so a compatibility-mode adapter is not overstated.
var defaultLimits = map[string]uint64{
	"maxStorageBufferBindingSize":       128 << 20,
	"maxComputeWorkgroupStorageSize":    16384,
	"maxComputeInvocationsPerWorkgroup": 256,
	"maxComputeWorkgroupsPerDimension":  65535,
	"maxStorageBuffersPerShaderStage":   8,
	"maxBindGroups":                     4, // the last one is the host's (layer 4)
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
	// One more storage buffer per stage than the default: the loop budget's
	// flag, so that a shader that binds the 8 the capset reports can still
	// be rewritten.
	if !b.NoLoopBudget {
		q.Limits = batch.RequiredLimits(ad)
	}
	dev, err := ad.RequestDevice(ctx, &wg.DeviceDescriptor{Label: "wasi_webgpu_batch", RequiredFeatures: req, RequiredLimits: q.Limits, Quota: q})
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
	work := b.LoopWork
	if work == 0 {
		work = DefaultLoopWork
	}
	bd := &browserDevice{dev: dev, features: features, loop: batch.LoopBudget{
		Work: work, MinIterations: b.LoopMin, Group: uint32(lim("maxBindGroups")) - 1}, noLoop: b.NoLoopBudget,
		limits: wgpubatch.DeviceLimits{
			MaxStorageBufferBindingSize:       lim("maxStorageBufferBindingSize"),
			MaxComputeWorkgroupStorageSize:    uint32(lim("maxComputeWorkgroupStorageSize")),
			MaxComputeInvocationsPerWorkgroup: uint32(lim("maxComputeInvocationsPerWorkgroup")),
			MaxComputeWorkgroupsPerDimension:  uint32(lim("maxComputeWorkgroupsPerDimension")),
			MaxStorageBuffersPerShaderStage:   uint32(lim("maxStorageBuffersPerShaderStage")),
		}}
	go func() { // a device the browser took, not the host: see LastBrowserLoss
		<-dev.Lost()
		if !bd.ours.Load() {
			browserLoss.Lock()
			browserLoss.last = time.Now()
			browserLoss.Unlock()
		}
	}()
	return bd, nil
}

// browserLoss is when a device of this page was last lost to the browser (its
// own watchdog, a GPU reset) rather than destroyed by the host.
var browserLoss struct {
	sync.Mutex
	last time.Time
}

// LastBrowserLoss implements LossWatcher: the time a device was last lost
// without the host destroying it (zero if none).
func (Browser) LastBrowserLoss() time.Time {
	browserLoss.Lock()
	defer browserLoss.Unlock()
	return browserLoss.last
}

type browserDevice struct {
	dev      *wg.Device
	features uint32
	limits   wgpubatch.DeviceLimits
	loop     batch.LoopBudget
	noLoop   bool
	ours     atomic.Bool // the host destroyed it
}

func (d *browserDevice) Caps() (uint32, wgpubatch.DeviceLimits) { return d.features, d.limits }
func (d *browserDevice) Destroy()                               { d.ours.Store(true); d.dev.Destroy() }

// Backend returns a backend with the loop budget on (layer 4), unless the
// provider turned it off. If the budget's host objects cannot be made, the
// backend refuses everything with that error: a container never runs
// unbounded by accident.
func (d *browserDevice) Backend() batch.Backend {
	be := batch.NewDeviceBackend(d.dev)
	if d.noLoop {
		return be
	}
	if err := be.SetLoopBudget(d.loop); err != nil {
		return failedBackend{err}
	}
	return be
}

// failedBackend is a Backend whose every creation fails with err.
type failedBackend struct{ err error }

func (f failedBackend) Create(uint32, []byte) (uint32, error) { return 0, f.err }
func (failedBackend) Destroy(uint32)                          {}
func (f failedBackend) Replay([]uint32, []byte) error         { return f.err }
func (failedBackend) Close()                                  {}

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
//
// A browser may refuse the probe while the GPU is still busy: Safari 27.2
// answers requestAdapter with null for as long as a destroyed device's
// shader runs (measured: the container started right after a sanction got
// "no GPU"). A refusal is therefore retried every 100 ms until the timeout:
// the probe being refused is itself the sign that the GPU has not drained.
func (b Browser) Drain(timeout time.Duration) error {
	drainProbe.Lock()
	defer drainProbe.Unlock()
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	for {
		err := b.drainOnce(ctx)
		if err == nil || ctx.Err() != nil {
			return err
		}
		select {
		case <-ctx.Done():
			return err
		case <-time.After(100 * time.Millisecond):
		}
	}
}

// drainOnce opens the probe if needed and waits for one trivial submit.
func (b Browser) drainOnce(ctx context.Context) error {
	{
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
}
