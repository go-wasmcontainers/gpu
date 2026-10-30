//go:build js

package webgpu

import (
	"context"
	"errors"
	"sync"
	"syscall/js"
	"time"

	wg "github.com/go-gfx/webgpu"
	"github.com/go-wasmcontainers/gpu/wit"
)

// Browser is the backend in the browser: WebGPU through the worker's
// navigator.gpu (the playground's engine worker runs the WASI host), by
// way of github.com/go-gfx/webgpu. Each process's GPU memory, pipelines,
// WGSL size and surface size are a wg.Quota the library enforces before
// the browser is called; the handle count stays this package's (Handles).
type Browser struct {
	// AllowFallback offers a software adapter; off by default, since its
	// work is not GPU work.
	AllowFallback bool
	// GPUTime bounds one submission's GPU work; past it the device is
	// destroyed (lost) rather than letting a container hold the GPU
	// (default 2 s, the order of a driver watchdog).
	GPUTime time.Duration
	// OnSurface receives the surfaces' events: "open", "resize", "close",
	// and "frame" (an ImageBitmap the receiver must transfer or close,
	// with its sequence number). Nil drops frames (closing them). The
	// page acknowledges drawn frames with AckFrame (docs/DESIGN.md,
	// "Surfaces").
	OnSurface func(SurfaceEvent)
	// FrameInterval paces present (default 1/60 s): a surface never shows
	// more than one frame per interval.
	FrameInterval time.Duration
	// MaxWidth and MaxHeight bound a surface (default 1920x1080).
	MaxWidth, MaxHeight int
}

// SurfaceEvent is what OnSurface receives.
type SurfaceEvent struct {
	Op            string // "open", "resize", "close", "frame"
	Surface       uint32
	Container     string // open: the container's label (its ID), "" if unknown
	Width, Height int
	Seq           uint64   // frames: from 1
	Bitmap        js.Value // frames: an ImageBitmap
}

// MaxFramesInFlight is how many unacknowledged frames a surface may have
// before present waits (at most AckWait) for AckFrame.
const MaxFramesInFlight = 2

// AckWait bounds one wait for acknowledgements: a page that stops
// acknowledging slows a guest down to one frame per AckWait, never stops
// it.
const AckWait = time.Second

var surfaces = struct {
	sync.Mutex
	n   uint32
	all map[uint32]*surfaceObj
}{all: map[uint32]*surfaceObj{}}

// AckFrame records that frame seq of a surface was drawn or dropped.
func AckFrame(surface uint32, seq uint64) {
	surfaces.Lock()
	sf := surfaces.all[surface]
	surfaces.Unlock()
	if sf == nil {
		return
	}
	sf.mu.Lock()
	if seq > sf.acked {
		sf.acked = seq
		close(sf.ackCh)
		sf.ackCh = make(chan struct{})
	}
	sf.mu.Unlock()
}

// SurfaceInput delivers a pointer or key event from the page to a
// surface's input streams (docs/DESIGN.md, "Surfaces"); it reports
// whether the surface exists and the event was valid.
func SurfaceInput(surface uint32, e InputEvent) bool {
	surfaces.Lock()
	sf := surfaces.all[surface]
	surfaces.Unlock()
	return sf != nil && sf.input.publish(e)
}

// surfaceObj is a container's surface: an OffscreenCanvas in this worker.
// Its WebGPU side (surf) exists once a context is configured.
type surfaceObj struct {
	id     uint32
	canvas js.Value
	surf   *wg.Surface
	last   time.Time
	emit   func(SurfaceEvent)
	input  inputHub

	mu    sync.Mutex
	seq   uint64
	acked uint64
	ackCh chan struct{}
}

func (s *surfaceObj) Destroy() {
	surfaces.Lock()
	delete(surfaces.all, s.id)
	surfaces.Unlock()
	s.input.close()
	if s.emit != nil {
		s.emit(SurfaceEvent{Op: "close", Surface: s.id})
	}
}

type contextObj struct {
	sf *surfaceObj
}

func (c *contextObj) Destroy() {}

// Open implements Backend.
func (b Browser) Open(p Process, lim Limits, h *Handles) Session {
	if b.GPUTime <= 0 {
		b.GPUTime = 2 * time.Second
	}
	if b.FrameInterval <= 0 {
		b.FrameInterval = time.Second / 60
	}
	if b.MaxWidth <= 0 || b.MaxHeight <= 0 {
		b.MaxWidth, b.MaxHeight = 1920, 1080
	}
	// One quota per process, shared by every device it requests. No
	// optional feature and no required limit is listed, so none is ever
	// granted (timestamp-query included).
	q := &wg.Quota{
		MemoryBytes:   lim.MemoryBytes,
		Pipelines:     lim.Pipelines,
		ShaderBytes:   lim.ShaderBytes,
		SurfaceWidth:  b.MaxWidth,
		SurfaceHeight: b.MaxHeight,
	}
	return &session{b: b, gpu: p.GPU, container: p.Container, lim: lim, h: h, quota: q}
}

type session struct {
	b         Browser
	gpu       bool
	container string
	lim       Limits
	h         *Handles
	quota     *wg.Quota
}

var asyncCalls = map[string]bool{
	"[method]gpu.request-adapter":                      true,
	"[method]gpu-adapter.request-device":               true,
	"[method]gpu-buffer.map-async":                     true,
	"[method]gpu-device.create-compute-pipeline-async": true,
	"[method]gpu-queue.on-submitted-work-done":         true,
	"[method]gpu-device.pop-error-scope":               true,
	"[method]gpu-shader-module.get-compilation-info":   true,
	"[method]context.present":                          true, // paced
}

func (s *session) Async(name string) bool { return asyncCalls[name] }
func (s *session) Close()                 {}

// obj is a library object behind a handle; destroy runs at drop.
type obj struct {
	v       any
	destroy func()
}

func (o *obj) Destroy() {
	if o.destroy != nil {
		o.destroy()
	}
}

// queueObj is a queue with its device, whose timeline the GPU-time
// watchdog waits on.
type queueObj struct {
	q   *wg.Queue
	dev *wg.Device
}

// raw returns what a handle holds (nil for an empty one, such as get-gpu
// without --gpu).
func (s *session) raw(h any, res string) (any, error) {
	n, ok := h.(uint32)
	if !ok {
		return nil, Trapf("%s handle is %T", res, h)
	}
	o, err := s.h.Get(n, res)
	if err != nil {
		return nil, err
	}
	if jo, ok := o.(*obj); ok && jo != nil {
		return jo.v, nil
	}
	return nil, nil
}

// get returns a handle's library object as a T.
func get[T any](s *session, h any, res string) (T, error) {
	var zero T
	v, err := s.raw(h, res)
	if err != nil {
		return zero, err
	}
	t, ok := v.(T)
	if !ok {
		return zero, Trapf("%s handle holds no object", res)
	}
	return t, nil
}

func (s *session) put(res string, v any, destroy func()) (any, error) {
	return s.h.New(res, &obj{v: v, destroy: destroy})
}

func u64(v any) uint64 {
	switch x := v.(type) {
	case uint64:
		return x
	case uint32:
		return uint64(x)
	case int64:
		return uint64(x)
	}
	return 0
}

// opt returns an option's value and whether it is present.
func opt(v any) (any, bool) {
	if s, ok := v.(wit.Some); ok {
		return s.Value, true
	}
	return nil, false
}

func record(v any) []any {
	r, _ := v.([]any)
	return r
}

// errRecord is a WIT {kind: variant, message: string} error.
func errRecord(kind int, msg string) wit.ResultValue {
	return wit.ResultValue{Err: true, Value: []any{wit.VariantValue{Case: kind}, msg}}
}

var okUnit = wit.ResultValue{}

// trap turns a library error on a call the WIT gives no error result into
// a trap: a refused argument is loud, never a silent device error.
func trap(op string, err error) error { return Trapf("%s: %v", op, err) }

// dynamicOffsets reads an option<list<u32>>.
func dynamicOffsets(v any) []uint32 {
	l, ok := opt(v)
	if !ok {
		return nil
	}
	xs, _ := l.([]any)
	out := make([]uint32, len(xs))
	for i, x := range xs {
		out[i] = uint32(u64(x))
	}
	return out
}

func (s *session) Call(name string, a []any) (any, error) {
	ctx := context.Background()
	switch name {
	case "get-gpu":
		g, err := wg.Navigator()
		if !s.gpu || err != nil {
			return s.h.New("gpu", nil)
		}
		return s.put("gpu", g, nil)
	case "[method]gpu.request-adapter":
		v, err := s.raw(a[0], "gpu")
		if err != nil {
			return nil, err
		}
		g, ok := v.(*wg.GPU)
		if !ok {
			return nil, nil // no --gpu, or no WebGPU: none
		}
		// A fallback adapter is refused unless allowed: a software
		// adapter is not offered as a GPU.
		ad, err := g.RequestAdapter(ctx, &wg.RequestAdapterOptions{AllowFallback: s.b.AllowFallback})
		if err != nil {
			return nil, nil
		}
		h, err := s.put("gpu-adapter", ad, nil)
		if err != nil {
			return nil, err
		}
		return wit.Some{Value: h}, nil
	case "[method]gpu.get-preferred-canvas-format":
		f := "bgra8unorm"
		if g, err := wg.Navigator(); err == nil {
			f = string(g.PreferredCanvasFormat())
		}
		return formatIndex(f), nil
	case "[method]gpu.wgsl-language-features":
		return s.h.New("wgsl-language-features", nil)
	case "[method]wgsl-language-features.has":
		return false, nil
	case "[method]gpu-adapter.request-device":
		ad, err := get[*wg.Adapter](s, a[0], "gpu-adapter")
		if err != nil {
			return nil, err
		}
		if d, ok := opt(a[1]); ok {
			if f, ok := opt(record(d)[0]); ok && len(f.([]any)) > 0 {
				return errRecord(0, "optional features are not offered to containers"), nil
			}
			if _, ok := opt(record(d)[1]); ok {
				return errRecord(0, "required limits are not offered to containers"), nil
			}
		}
		dev, err := ad.RequestDevice(ctx, &wg.DeviceDescriptor{Quota: s.quota})
		if err != nil {
			return errRecord(1, err.Error()), nil
		}
		h, err := s.put("gpu-device", dev, dev.Destroy)
		if err != nil {
			dev.Destroy()
			return nil, err
		}
		return wit.ResultValue{Value: h}, nil
	case "[method]gpu-device.queue":
		d, err := get[*wg.Device](s, a[0], "gpu-device")
		if err != nil {
			return nil, err
		}
		return s.put("gpu-queue", queueObj{q: d.Queue(), dev: d}, nil)
	case "[method]gpu-device.destroy":
		d, err := get[*wg.Device](s, a[0], "gpu-device")
		if err != nil {
			return nil, err
		}
		d.Destroy()
		return nil, nil
	case "[method]gpu-device.create-buffer":
		return s.createBuffer(a)
	case "[method]gpu-buffer.size":
		b, err := get[*wg.Buffer](s, a[0], "gpu-buffer")
		if err != nil {
			return nil, err
		}
		return b.Size(), nil
	case "[method]gpu-buffer.usage":
		b, err := get[*wg.Buffer](s, a[0], "gpu-buffer")
		if err != nil {
			return nil, err
		}
		return uint64(b.Usage()), nil
	case "[method]gpu-buffer.destroy":
		b, err := get[*wg.Buffer](s, a[0], "gpu-buffer")
		if err != nil {
			return nil, err
		}
		b.Destroy()
		return nil, nil
	case "[method]gpu-queue.write-buffer-with-copy":
		q, err := get[queueObj](s, a[0], "gpu-queue")
		if err != nil {
			return nil, err
		}
		b, err := get[*wg.Buffer](s, a[1], "gpu-buffer")
		if err != nil {
			return nil, err
		}
		data, _ := a[3].([]byte)
		if off, ok := opt(a[4]); ok {
			if u64(off) > uint64(len(data)) {
				return errRecord(0, "data offset past the data"), nil
			}
			data = data[u64(off):]
		}
		if sz, ok := opt(a[5]); ok {
			if u64(sz) > uint64(len(data)) {
				return errRecord(0, "size past the data"), nil
			}
			data = data[:u64(sz)]
		}
		// Range, alignment and COPY_DST are checked by the library: an
		// out-of-range write is an error result, not a device error.
		if err := q.q.WriteBuffer(b, u64(a[2]), data); err != nil {
			return errRecord(0, err.Error()), nil
		}
		return okUnit, nil
	case "[method]gpu-device.create-shader-module":
		d, err := get[*wg.Device](s, a[0], "gpu-device")
		if err != nil {
			return nil, err
		}
		code, _ := record(a[1])[0].(string)
		m, err := d.CreateShaderModule(wg.ShaderModuleDescriptor{Code: code})
		if err != nil {
			var qe *wg.QuotaError
			if errors.As(err, &qe) {
				return nil, Trapf("WGSL source of %d bytes (quota %d)", len(code), s.lim.ShaderBytes)
			}
			return nil, trap("create-shader-module", err)
		}
		return s.put("gpu-shader-module", m, nil)
	case "[method]gpu-device.create-compute-pipeline":
		return s.createComputePipeline(a)
	case "[method]gpu-compute-pipeline.get-bind-group-layout":
		p, err := get[*wg.ComputePipeline](s, a[0], "gpu-compute-pipeline")
		if err != nil {
			return nil, err
		}
		l, err := p.GetBindGroupLayout(uint32(u64(a[1])))
		if err != nil {
			return nil, trap("get-bind-group-layout", err)
		}
		return s.put("gpu-bind-group-layout", l, nil)
	case "[method]gpu-device.create-bind-group":
		return s.createBindGroup(a)
	case "[method]gpu-device.create-command-encoder":
		d, err := get[*wg.Device](s, a[0], "gpu-device")
		if err != nil {
			return nil, err
		}
		e, err := d.CreateCommandEncoder("")
		if err != nil {
			return nil, trap("create-command-encoder", err)
		}
		return s.put("gpu-command-encoder", e, nil)
	case "[method]gpu-command-encoder.begin-compute-pass":
		e, err := get[*wg.CommandEncoder](s, a[0], "gpu-command-encoder")
		if err != nil {
			return nil, err
		}
		p, err := e.BeginComputePass("")
		if err != nil {
			return nil, trap("begin-compute-pass", err)
		}
		return s.put("gpu-compute-pass-encoder", p, nil)
	case "[method]gpu-compute-pass-encoder.set-pipeline":
		pe, err := get[*wg.ComputePassEncoder](s, a[0], "gpu-compute-pass-encoder")
		if err != nil {
			return nil, err
		}
		p, err := get[*wg.ComputePipeline](s, a[1], "gpu-compute-pipeline")
		if err != nil {
			return nil, err
		}
		pe.SetPipeline(p)
		return nil, nil
	case "[method]gpu-compute-pass-encoder.set-bind-group":
		pe, err := get[*wg.ComputePassEncoder](s, a[0], "gpu-compute-pass-encoder")
		if err != nil {
			return nil, err
		}
		var bg *wg.BindGroup
		if h, ok := opt(a[2]); ok {
			if bg, err = get[*wg.BindGroup](s, h, "gpu-bind-group"); err != nil {
				return nil, err
			}
		}
		pe.SetBindGroup(uint32(u64(a[1])), bg, dynamicOffsets(a[3])...)
		return okUnit, nil
	case "[method]gpu-compute-pass-encoder.dispatch-workgroups":
		pe, err := get[*wg.ComputePassEncoder](s, a[0], "gpu-compute-pass-encoder")
		if err != nil {
			return nil, err
		}
		y, z := uint64(1), uint64(1)
		if v, ok := opt(a[2]); ok {
			y = u64(v)
		}
		if v, ok := opt(a[3]); ok {
			z = u64(v)
		}
		pe.DispatchWorkgroups(uint32(u64(a[1])), uint32(y), uint32(z))
		return nil, nil
	case "[method]gpu-compute-pass-encoder.end":
		pe, err := get[*wg.ComputePassEncoder](s, a[0], "gpu-compute-pass-encoder")
		if err != nil {
			return nil, err
		}
		if err := pe.End(); err != nil {
			return nil, trap("compute pass end", err)
		}
		return nil, nil
	case "[method]gpu-command-encoder.copy-buffer-to-buffer":
		e, err := get[*wg.CommandEncoder](s, a[0], "gpu-command-encoder")
		if err != nil {
			return nil, err
		}
		src, err := get[*wg.Buffer](s, a[1], "gpu-buffer")
		if err != nil {
			return nil, err
		}
		dst, err := get[*wg.Buffer](s, a[3], "gpu-buffer")
		if err != nil {
			return nil, err
		}
		so, do := uint64(0), uint64(0)
		if v, ok := opt(a[2]); ok {
			so = u64(v)
		}
		if v, ok := opt(a[4]); ok {
			do = u64(v)
		}
		if so > src.Size() {
			return nil, Trapf("copy-buffer-to-buffer: source offset %d past %d bytes", so, src.Size())
		}
		size := src.Size() - so
		if v, ok := opt(a[5]); ok {
			size = u64(v)
		}
		if err := e.CopyBufferToBuffer(src, so, dst, do, size); err != nil {
			return nil, trap("copy-buffer-to-buffer", err)
		}
		return nil, nil
	case "[method]gpu-command-encoder.finish":
		e, err := get[*wg.CommandEncoder](s, a[0], "gpu-command-encoder")
		if err != nil {
			return nil, err
		}
		cb, err := e.Finish()
		if err != nil {
			return nil, trap("finish", err)
		}
		return s.put("gpu-command-buffer", cb, nil)
	case "[method]gpu-queue.submit":
		return s.submit(a)
	case "[method]gpu-queue.on-submitted-work-done":
		q, err := get[queueObj](s, a[0], "gpu-queue")
		if err != nil {
			return nil, err
		}
		return nil, q.q.OnSubmittedWorkDone(ctx)
	case "[method]gpu-buffer.map-async":
		b, err := get[*wg.Buffer](s, a[0], "gpu-buffer")
		if err != nil {
			return nil, err
		}
		var off, size uint64
		if v, ok := opt(a[2]); ok {
			off = u64(v)
		}
		if v, ok := opt(a[3]); ok {
			size = u64(v) // 0 maps to the end, as the library defines it
		}
		if err := b.MapAsync(ctx, wg.MapMode(u64(a[1])&3), off, size); err != nil {
			return errRecord(0, err.Error()), nil
		}
		return okUnit, nil
	case "[method]gpu-buffer.get-mapped-range-get-with-copy":
		b, err := get[*wg.Buffer](s, a[0], "gpu-buffer")
		if err != nil {
			return nil, err
		}
		var off uint64
		if v, ok := opt(a[1]); ok {
			off = u64(v)
		}
		if off > b.Size() {
			return errRecord(0, "offset past the buffer"), nil
		}
		size := b.Size() - off
		if v, ok := opt(a[2]); ok {
			size = u64(v)
		}
		if size > b.Size() {
			return errRecord(0, "size past the buffer"), nil
		}
		out := make([]byte, size)
		if err := b.ReadMapped(off, out); err != nil {
			return errRecord(0, err.Error()), nil
		}
		return wit.ResultValue{Value: out}, nil
	case "[method]gpu-buffer.unmap":
		b, err := get[*wg.Buffer](s, a[0], "gpu-buffer")
		if err != nil {
			return nil, err
		}
		_ = b.Unmap()
		return okUnit, nil
	}
	if v, err, ok := s.render(name, a); ok {
		return v, err
	}
	return nil, Trapf("%s is not implemented in this engine yet", name)
}

func (s *session) createBuffer(a []any) (any, error) {
	d, err := get[*wg.Device](s, a[0], "gpu-device")
	if err != nil {
		return nil, err
	}
	r := record(a[1])
	desc := wg.BufferDescriptor{Size: u64(r[0]), Usage: wg.BufferUsage(u64(r[1]))}
	if m, ok := opt(r[2]); ok {
		desc.MappedAtCreation, _ = m.(bool)
	}
	buf, err := d.CreateBuffer(desc)
	if err != nil {
		var qe *wg.QuotaError
		if errors.As(err, &qe) {
			return nil, Trapf("GPU memory quota of %d bytes exceeded (buffer of %d)", s.lim.MemoryBytes, desc.Size)
		}
		return nil, trap("create-buffer", err)
	}
	h, err := s.put("gpu-buffer", buf, buf.Destroy)
	if err != nil {
		buf.Destroy()
	}
	return h, err
}

func (s *session) createComputePipeline(a []any) (any, error) {
	d, err := get[*wg.Device](s, a[0], "gpu-device")
	if err != nil {
		return nil, err
	}
	r := record(a[1])
	stage := record(r[0])
	mod, err := get[*wg.ShaderModule](s, stage[0], "gpu-shader-module")
	if err != nil {
		return nil, err
	}
	desc := wg.ComputePipelineDescriptor{Compute: wg.ProgrammableStage{Module: mod}}
	if e, ok := opt(stage[1]); ok {
		desc.Compute.EntryPoint, _ = e.(string)
	}
	if lm, _ := r[1].(wit.VariantValue); lm.Case == 0 {
		if desc.Layout, err = get[*wg.PipelineLayout](s, lm.Value, "gpu-pipeline-layout"); err != nil {
			return nil, err
		}
	}
	p, err := d.CreateComputePipeline(desc)
	if err != nil {
		var qe *wg.QuotaError
		if errors.As(err, &qe) {
			return nil, Trapf("pipeline quota of %d exceeded", s.lim.Pipelines)
		}
		return nil, trap("create-compute-pipeline", err)
	}
	return s.put("gpu-compute-pipeline", p, p.Release)
}

func (s *session) createBindGroup(a []any) (any, error) {
	d, err := get[*wg.Device](s, a[0], "gpu-device")
	if err != nil {
		return nil, err
	}
	r := record(a[1])
	layout, err := get[*wg.BindGroupLayout](s, r[0], "gpu-bind-group-layout")
	if err != nil {
		return nil, err
	}
	var entries []wg.BindGroupEntry
	for _, e := range r[1].([]any) {
		er := record(e)
		ent := wg.BindGroupEntry{Binding: uint32(u64(er[0]))}
		v, _ := er[1].(wit.VariantValue)
		switch v.Case {
		case 0:
			if ent.Buffer, err = get[*wg.Buffer](s, v.Value, "gpu-buffer"); err != nil {
				return nil, err
			}
		case 1:
			bb := record(v.Value)
			if ent.Buffer, err = get[*wg.Buffer](s, bb[0], "gpu-buffer"); err != nil {
				return nil, err
			}
			if o, ok := opt(bb[1]); ok {
				ent.Offset = u64(o)
			}
			if z, ok := opt(bb[2]); ok {
				ent.Size = u64(z)
			}
		default:
			return nil, Trapf("bind group resources other than buffers are not implemented yet")
		}
		entries = append(entries, ent)
	}
	g, err := d.CreateBindGroup(wg.BindGroupDescriptor{Layout: layout, Entries: entries})
	if err != nil {
		return nil, trap("create-bind-group", err)
	}
	return s.put("gpu-bind-group", g, nil)
}

// submit queues command buffers and arms the GPU-time watchdog: a queue
// still busy after GPUTime has its process's devices destroyed (the
// WebGPU spec's own watchdog may otherwise take seconds and stall the
// page). It waits on the device's submission timeline, so many
// submissions share one onSubmittedWorkDone round trip.
func (s *session) submit(a []any) (any, error) {
	q, err := get[queueObj](s, a[0], "gpu-queue")
	if err != nil {
		return nil, err
	}
	var cbs []*wg.CommandBuffer
	for _, h := range a[1].([]any) {
		cb, err := get[*wg.CommandBuffer](s, h, "gpu-command-buffer")
		if err != nil {
			return nil, err
		}
		cbs = append(cbs, cb)
	}
	if err := q.q.Submit(cbs...); err != nil {
		return nil, trap("submit", err)
	}
	seq := q.dev.Submitted()
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), s.b.GPUTime)
		defer cancel()
		if err := q.dev.WaitSubmitted(ctx, seq); errors.Is(err, context.DeadlineExceeded) {
			s.loseDevices()
		}
	}()
	return nil, nil
}

// loseDevices destroys every device of the process (GPU time exceeded).
func (s *session) loseDevices() {
	s.h.mu.Lock()
	var devs []*wg.Device
	for _, e := range s.h.tab {
		if e.res == "gpu-device" {
			if o, ok := e.obj.(*obj); ok {
				if d, ok := o.v.(*wg.Device); ok {
					devs = append(devs, d)
				}
			}
		}
	}
	s.h.mu.Unlock()
	for _, d := range devs {
		d.Destroy()
	}
}
