//go:build js

package webgpu

import (
	"fmt"
	"sync"
	"syscall/js"
	"time"

	"github.com/go-wasmcontainers/gpu/internal/jsgpu"
	"github.com/go-wasmcontainers/gpu/wit"
)

// Browser is the backend in the browser: WebGPU through the worker's
// navigator.gpu (the playground's engine worker runs the WASI host).
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
type surfaceObj struct {
	id     uint32
	canvas js.Value
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
	ctx js.Value
	sf  *surfaceObj
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
	return &session{b: b, gpu: p.GPU, container: p.Container, lim: lim, h: h}
}

type session struct {
	b         Browser
	gpu       bool
	container string
	lim       Limits
	h         *Handles

	mu        sync.Mutex
	memory    int64 // buffer bytes alive
	pipelines int
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

// jsObj is a WebGPU object behind a handle; destroy runs at drop.
type jsObj struct {
	v       js.Value
	destroy func()
}

func (o *jsObj) Destroy() {
	if o.destroy != nil {
		o.destroy()
	}
}

func (s *session) get(h any, res string) (js.Value, error) {
	n, ok := h.(uint32)
	if !ok {
		return js.Value{}, Trapf("%s handle is %T", res, h)
	}
	o, err := s.h.Get(n, res)
	if err != nil {
		return js.Value{}, err
	}
	if jo, ok := o.(*jsObj); ok && jo != nil {
		return jo.v, nil
	}
	return js.Undefined(), nil
}

func (s *session) put(res string, v js.Value, destroy func()) (any, error) {
	return s.h.New(res, &jsObj{v: v, destroy: destroy})
}

func obj() js.Value { return js.Global().Get("Object").New() }

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

func (s *session) Call(name string, a []any) (any, error) {
	switch name {
	case "get-gpu":
		gpu := js.Global().Get("navigator").Get("gpu")
		if !s.gpu || gpu.IsUndefined() || gpu.IsNull() {
			return s.h.New("gpu", nil)
		}
		return s.put("gpu", gpu, nil)
	case "[method]gpu.request-adapter":
		g, err := s.get(a[0], "gpu")
		if err != nil {
			return nil, err
		}
		if g.IsUndefined() {
			return nil, nil // no --gpu, or no WebGPU: none
		}
		ad, err := jsgpu.Await(g.Call("requestAdapter"))
		if err != nil || ad.IsNull() || ad.IsUndefined() {
			return nil, nil
		}
		fb := ad.Get("info").Get("isFallbackAdapter")
		if fb.Type() == js.TypeBoolean && fb.Bool() && !s.b.AllowFallback {
			return nil, nil // a software adapter is not offered as a GPU
		}
		h, err := s.put("gpu-adapter", ad, nil)
		if err != nil {
			return nil, err
		}
		return wit.Some{Value: h}, nil
	case "[method]gpu.get-preferred-canvas-format":
		f := "bgra8unorm"
		if g := js.Global().Get("navigator").Get("gpu"); !g.IsUndefined() {
			f = g.Call("getPreferredCanvasFormat").String()
		}
		return formatIndex(f), nil
	case "[method]gpu.wgsl-language-features":
		return s.h.New("wgsl-language-features", nil)
	case "[method]wgsl-language-features.has":
		return false, nil
	case "[method]gpu-adapter.request-device":
		ad, err := s.get(a[0], "gpu-adapter")
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
		dev, err := jsgpu.Await(ad.Call("requestDevice"))
		if err != nil {
			return errRecord(1, err.Error()), nil
		}
		h, err := s.put("gpu-device", dev, func() { dev.Call("destroy") })
		if err != nil {
			dev.Call("destroy")
			return nil, err
		}
		return wit.ResultValue{Value: h}, nil
	case "[method]gpu-device.queue":
		d, err := s.get(a[0], "gpu-device")
		if err != nil {
			return nil, err
		}
		return s.put("gpu-queue", d.Get("queue"), nil)
	case "[method]gpu-device.destroy":
		d, err := s.get(a[0], "gpu-device")
		if err != nil {
			return nil, err
		}
		d.Call("destroy")
		return nil, nil
	case "[method]gpu-device.create-buffer":
		return s.createBuffer(a)
	case "[method]gpu-buffer.size":
		b, err := s.get(a[0], "gpu-buffer")
		if err != nil {
			return nil, err
		}
		return uint64(b.Get("size").Float()), nil
	case "[method]gpu-buffer.usage":
		b, err := s.get(a[0], "gpu-buffer")
		if err != nil {
			return nil, err
		}
		return uint64(b.Get("usage").Int()), nil
	case "[method]gpu-buffer.destroy":
		b, err := s.get(a[0], "gpu-buffer")
		if err != nil {
			return nil, err
		}
		b.Call("destroy")
		return nil, nil
	case "[method]gpu-queue.write-buffer-with-copy":
		q, err := s.get(a[0], "gpu-queue")
		if err != nil {
			return nil, err
		}
		b, err := s.get(a[1], "gpu-buffer")
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
		u := js.Global().Get("Uint8Array").New(len(data))
		js.CopyBytesToJS(u, data)
		q.Call("writeBuffer", b, float64(u64(a[2])), u)
		return okUnit, nil
	case "[method]gpu-device.create-shader-module":
		d, err := s.get(a[0], "gpu-device")
		if err != nil {
			return nil, err
		}
		code, _ := record(a[1])[0].(string)
		if len(code) > s.lim.ShaderBytes {
			return nil, Trapf("WGSL source of %d bytes (quota %d)", len(code), s.lim.ShaderBytes)
		}
		desc := obj()
		desc.Set("code", code)
		return s.put("gpu-shader-module", d.Call("createShaderModule", desc), nil)
	case "[method]gpu-device.create-compute-pipeline":
		return s.createComputePipeline(a)
	case "[method]gpu-compute-pipeline.get-bind-group-layout":
		p, err := s.get(a[0], "gpu-compute-pipeline")
		if err != nil {
			return nil, err
		}
		return s.put("gpu-bind-group-layout", p.Call("getBindGroupLayout", u64(a[1])), nil)
	case "[method]gpu-device.create-bind-group":
		return s.createBindGroup(a)
	case "[method]gpu-device.create-command-encoder":
		d, err := s.get(a[0], "gpu-device")
		if err != nil {
			return nil, err
		}
		return s.put("gpu-command-encoder", d.Call("createCommandEncoder"), nil)
	case "[method]gpu-command-encoder.begin-compute-pass":
		e, err := s.get(a[0], "gpu-command-encoder")
		if err != nil {
			return nil, err
		}
		return s.put("gpu-compute-pass-encoder", e.Call("beginComputePass"), nil)
	case "[method]gpu-compute-pass-encoder.set-pipeline":
		pe, err := s.get(a[0], "gpu-compute-pass-encoder")
		if err != nil {
			return nil, err
		}
		p, err := s.get(a[1], "gpu-compute-pipeline")
		if err != nil {
			return nil, err
		}
		pe.Call("setPipeline", p)
		return nil, nil
	case "[method]gpu-compute-pass-encoder.set-bind-group":
		pe, err := s.get(a[0], "gpu-compute-pass-encoder")
		if err != nil {
			return nil, err
		}
		bg := js.Null()
		if h, ok := opt(a[2]); ok {
			if bg, err = s.get(h, "gpu-bind-group"); err != nil {
				return nil, err
			}
		}
		if _, ok := opt(a[3]); ok {
			return errRecord(0, "dynamic offsets are not supported"), nil
		}
		pe.Call("setBindGroup", u64(a[1]), bg)
		return okUnit, nil
	case "[method]gpu-compute-pass-encoder.dispatch-workgroups":
		pe, err := s.get(a[0], "gpu-compute-pass-encoder")
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
		pe.Call("dispatchWorkgroups", u64(a[1]), y, z)
		return nil, nil
	case "[method]gpu-compute-pass-encoder.end":
		pe, err := s.get(a[0], "gpu-compute-pass-encoder")
		if err != nil {
			return nil, err
		}
		pe.Call("end")
		return nil, nil
	case "[method]gpu-command-encoder.copy-buffer-to-buffer":
		e, err := s.get(a[0], "gpu-command-encoder")
		if err != nil {
			return nil, err
		}
		src, err := s.get(a[1], "gpu-buffer")
		if err != nil {
			return nil, err
		}
		dst, err := s.get(a[3], "gpu-buffer")
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
		size := uint64(src.Get("size").Float()) - so
		if v, ok := opt(a[5]); ok {
			size = u64(v)
		}
		e.Call("copyBufferToBuffer", src, float64(so), dst, float64(do), float64(size))
		return nil, nil
	case "[method]gpu-command-encoder.finish":
		e, err := s.get(a[0], "gpu-command-encoder")
		if err != nil {
			return nil, err
		}
		return s.put("gpu-command-buffer", e.Call("finish"), nil)
	case "[method]gpu-queue.submit":
		return s.submit(a)
	case "[method]gpu-queue.on-submitted-work-done":
		q, err := s.get(a[0], "gpu-queue")
		if err != nil {
			return nil, err
		}
		_, err = jsgpu.Await(q.Call("onSubmittedWorkDone"))
		return nil, err
	case "[method]gpu-buffer.map-async":
		b, err := s.get(a[0], "gpu-buffer")
		if err != nil {
			return nil, err
		}
		mode := u64(a[1]) & 3
		args := []any{mode}
		off, hasOff := opt(a[2])
		if hasOff {
			args = append(args, float64(u64(off)))
		}
		if sz, ok := opt(a[3]); ok {
			if !hasOff {
				args = append(args, 0)
			}
			args = append(args, float64(u64(sz)))
		}
		if _, err := jsgpu.Await(b.Call("mapAsync", args...)); err != nil {
			return errRecord(0, err.Error()), nil
		}
		return okUnit, nil
	case "[method]gpu-buffer.get-mapped-range-get-with-copy":
		b, err := s.get(a[0], "gpu-buffer")
		if err != nil {
			return nil, err
		}
		var r js.Value
		var cerr error
		func() {
			defer func() {
				if p := recover(); p != nil {
					cerr = fmt.Errorf("%v", p)
				}
			}()
			args := []any{}
			if v, ok := opt(a[1]); ok {
				args = append(args, float64(u64(v)))
				if v, ok := opt(a[2]); ok {
					args = append(args, float64(u64(v)))
				}
			}
			r = b.Call("getMappedRange", args...)
		}()
		if cerr != nil {
			return errRecord(0, cerr.Error()), nil
		}
		u := js.Global().Get("Uint8Array").New(r)
		out := make([]byte, u.Get("length").Int())
		js.CopyBytesToGo(out, u)
		return wit.ResultValue{Value: out}, nil
	case "[method]gpu-buffer.unmap":
		b, err := s.get(a[0], "gpu-buffer")
		if err != nil {
			return nil, err
		}
		b.Call("unmap")
		return okUnit, nil
	}
	if v, err, ok := s.render(name, a); ok {
		return v, err
	}
	return nil, Trapf("%s is not implemented in this engine yet", name)
}

func (s *session) createBuffer(a []any) (any, error) {
	d, err := s.get(a[0], "gpu-device")
	if err != nil {
		return nil, err
	}
	r := record(a[1])
	size := int64(u64(r[0]))
	s.mu.Lock()
	if size < 0 || size > s.lim.MemoryBytes-s.memory {
		s.mu.Unlock()
		return nil, Trapf("GPU memory quota of %d bytes exceeded (buffer of %d)", s.lim.MemoryBytes, size)
	}
	s.memory += size
	s.mu.Unlock()
	desc := obj()
	desc.Set("size", float64(size))
	desc.Set("usage", int(u64(r[1])))
	if m, ok := opt(r[2]); ok {
		desc.Set("mappedAtCreation", m.(bool))
	}
	buf := d.Call("createBuffer", desc)
	h, err := s.put("gpu-buffer", buf, func() {
		buf.Call("destroy")
		s.mu.Lock()
		s.memory -= size
		s.mu.Unlock()
	})
	if err != nil {
		buf.Call("destroy")
		s.mu.Lock()
		s.memory -= size
		s.mu.Unlock()
	}
	return h, err
}

func (s *session) createComputePipeline(a []any) (any, error) {
	d, err := s.get(a[0], "gpu-device")
	if err != nil {
		return nil, err
	}
	s.mu.Lock()
	if s.pipelines >= s.lim.Pipelines {
		s.mu.Unlock()
		return nil, Trapf("pipeline quota of %d exceeded", s.lim.Pipelines)
	}
	s.pipelines++
	s.mu.Unlock()
	r := record(a[1])
	stage := record(r[0])
	mod, err := s.get(stage[0], "gpu-shader-module")
	if err != nil {
		return nil, err
	}
	comp := obj()
	comp.Set("module", mod)
	if e, ok := opt(stage[1]); ok {
		comp.Set("entryPoint", e.(string))
	}
	desc := obj()
	desc.Set("compute", comp)
	lm, _ := r[1].(wit.VariantValue)
	if lm.Case == 0 {
		pl, err := s.get(lm.Value, "gpu-pipeline-layout")
		if err != nil {
			return nil, err
		}
		desc.Set("layout", pl)
	} else {
		desc.Set("layout", "auto")
	}
	p := d.Call("createComputePipeline", desc)
	return s.put("gpu-compute-pipeline", p, func() {
		s.mu.Lock()
		s.pipelines--
		s.mu.Unlock()
	})
}

func (s *session) createBindGroup(a []any) (any, error) {
	d, err := s.get(a[0], "gpu-device")
	if err != nil {
		return nil, err
	}
	r := record(a[1])
	layout, err := s.get(r[0], "gpu-bind-group-layout")
	if err != nil {
		return nil, err
	}
	var entries []any
	for _, e := range r[1].([]any) {
		er := record(e)
		ent := obj()
		ent.Set("binding", u64(er[0]))
		res := obj()
		v, _ := er[1].(wit.VariantValue)
		switch v.Case {
		case 0:
			b, err := s.get(v.Value, "gpu-buffer")
			if err != nil {
				return nil, err
			}
			res.Set("buffer", b)
		case 1:
			bb := record(v.Value)
			b, err := s.get(bb[0], "gpu-buffer")
			if err != nil {
				return nil, err
			}
			res.Set("buffer", b)
			if o, ok := opt(bb[1]); ok {
				res.Set("offset", float64(u64(o)))
			}
			if z, ok := opt(bb[2]); ok {
				res.Set("size", float64(u64(z)))
			}
		default:
			return nil, Trapf("bind group resources other than buffers are not implemented yet")
		}
		ent.Set("resource", res)
		entries = append(entries, ent)
	}
	desc := obj()
	desc.Set("layout", layout)
	desc.Set("entries", entries)
	return s.put("gpu-bind-group", d.Call("createBindGroup", desc), nil)
}

// submit queues command buffers and arms the GPU-time watchdog: a queue
// still busy after GPUTime has its device destroyed (the WebGPU spec's own
// watchdog may otherwise take seconds and stall the page).
func (s *session) submit(a []any) (any, error) {
	q, err := s.get(a[0], "gpu-queue")
	if err != nil {
		return nil, err
	}
	var cbs []any
	for _, h := range a[1].([]any) {
		cb, err := s.get(h, "gpu-command-buffer")
		if err != nil {
			return nil, err
		}
		cbs = append(cbs, cb)
	}
	q.Call("submit", cbs)
	done := make(chan struct{})
	go func() {
		jsgpu.Await(q.Call("onSubmittedWorkDone"))
		close(done)
	}()
	go func() {
		select {
		case <-done:
		case <-time.After(s.b.GPUTime):
			s.loseDevices()
		}
	}()
	return nil, nil
}

// loseDevices destroys every device of the process (GPU time exceeded).
func (s *session) loseDevices() {
	s.h.mu.Lock()
	var devs []*jsObj
	for _, e := range s.h.tab {
		if e.res == "gpu-device" {
			if o, ok := e.obj.(*jsObj); ok {
				devs = append(devs, o)
			}
		}
	}
	s.h.mu.Unlock()
	for _, d := range devs {
		d.v.Call("destroy")
	}
}
