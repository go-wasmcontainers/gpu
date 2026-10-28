//go:build js

package webgpu

import (
	"syscall/js"
	"time"

	"github.com/go-wasmcontainers/gpu/wit"
)

// The rendering subset: surfaces (an OffscreenCanvas each, in the engine's
// worker), their WebGPU contexts, render pipelines without vertex buffers,
// render passes, draws; frames leave through Browser.OnFrame as
// ImageBitmaps, paced to Browser.FrameInterval.

var formats = Interface.Types["gpu-texture-format"].Labels

// formatIndex is a WebGPU texture format's index in the WIT enum.
func formatIndex(f string) uint32 {
	for i, l := range formats {
		if l == f {
			return uint32(i)
		}
	}
	return 0
}

func formatName(v any) string {
	i, _ := v.(uint32)
	if int(i) < len(formats) {
		return formats[i]
	}
	return "bgra8unorm"
}

func enumLabel(t string, v any) string {
	ls := Interface.Types[t].Labels
	i, _ := v.(uint32)
	if int(i) < len(ls) {
		return ls[i]
	}
	return ""
}

func (s *session) getObj(h any, res string) (any, error) {
	n, ok := h.(uint32)
	if !ok {
		return nil, Trapf("%s handle is %T", res, h)
	}
	return s.h.Get(n, res)
}

// size bounds a requested surface size.
func (s *session) size(h, w any, defH, defW int) (int, int, error) {
	height, width := defH, defW
	if v, ok := opt(h); ok {
		height = int(u64(v))
	}
	if v, ok := opt(w); ok {
		width = int(u64(v))
	}
	if width < 1 || height < 1 || width > s.b.MaxWidth || height > s.b.MaxHeight {
		return 0, 0, Trapf("surface of %dx%d (at most %dx%d)", width, height, s.b.MaxWidth, s.b.MaxHeight)
	}
	return height, width, nil
}

// render serves the surface and render functions; ok is false for a name
// it does not know.
func (s *session) render(name string, a []any) (any, error, bool) {
	switch name {
	case "[constructor]surface":
		r := record(a[0])
		h, w, err := s.size(r[0], r[1], 480, 640)
		if err != nil {
			return nil, err, true
		}
		sf := &surfaceObj{canvas: js.Global().Get("OffscreenCanvas").New(w, h), emit: s.b.OnSurface, ackCh: make(chan struct{})}
		surfaces.Lock()
		surfaces.n++
		sf.id = surfaces.n
		surfaces.all[sf.id] = sf
		surfaces.Unlock()
		v, err := s.h.New("surface", sf)
		if err != nil {
			surfaces.Lock()
			delete(surfaces.all, sf.id)
			surfaces.Unlock()
			return nil, err, true
		}
		if sf.emit != nil {
			sf.emit(SurfaceEvent{Op: "open", Surface: sf.id, Width: w, Height: h})
		}
		return v, nil, true
	case "[method]surface.height", "[method]surface.width":
		o, err := s.getObj(a[0], "surface")
		if err != nil {
			return nil, err, true
		}
		c := o.(*surfaceObj).canvas
		if name == "[method]surface.height" {
			return uint64(c.Get("height").Int()), nil, true
		}
		return uint64(c.Get("width").Int()), nil, true
	case "[method]surface.request-set-size":
		o, err := s.getObj(a[0], "surface")
		if err != nil {
			return nil, err, true
		}
		c := o.(*surfaceObj).canvas
		h, w, err := s.size(a[1], a[2], c.Get("height").Int(), c.Get("width").Int())
		if err != nil {
			return nil, err, true
		}
		c.Set("width", w)
		c.Set("height", h)
		if sf := o.(*surfaceObj); sf.emit != nil {
			sf.emit(SurfaceEvent{Op: "resize", Surface: sf.id, Width: w, Height: h})
		}
		return nil, nil, true
	case "[constructor]context":
		o, err := s.getObj(a[0], "surface")
		if err != nil {
			return nil, err, true
		}
		sf := o.(*surfaceObj)
		ctx := sf.canvas.Call("getContext", "webgpu")
		if ctx.IsNull() || ctx.IsUndefined() {
			return nil, Trapf("no WebGPU canvas context here"), true
		}
		v, err := s.h.New("context", &contextObj{ctx: ctx, sf: sf})
		return v, err, true
	case "[method]context.configure":
		o, err := s.getObj(a[0], "context")
		if err != nil {
			return nil, err, true
		}
		r := record(a[1])
		dev, err := s.get(r[0], "gpu-device")
		if err != nil {
			return nil, err, true
		}
		cfg := obj()
		cfg.Set("device", dev)
		cfg.Set("format", formatName(r[1]))
		if u, ok := opt(r[2]); ok {
			cfg.Set("usage", int(u64(u)))
		}
		cfg.Set("alphaMode", "premultiplied")
		if m, ok := opt(r[6]); ok {
			cfg.Set("alphaMode", enumLabel("gpu-canvas-alpha-mode", m))
		}
		o.(*contextObj).ctx.Call("configure", cfg)
		return nil, nil, true
	case "[method]context.unconfigure":
		o, err := s.getObj(a[0], "context")
		if err != nil {
			return nil, err, true
		}
		o.(*contextObj).ctx.Call("unconfigure")
		return nil, nil, true
	case "[method]context.get-current-texture":
		o, err := s.getObj(a[0], "context")
		if err != nil {
			return nil, err, true
		}
		t := o.(*contextObj).ctx.Call("getCurrentTexture")
		v, err := s.put("gpu-texture", t, nil)
		return v, err, true
	case "[method]context.present":
		o, err := s.getObj(a[0], "context")
		if err != nil {
			return nil, err, true
		}
		return nil, s.present(o.(*contextObj)), true
	case "[method]gpu-texture.create-view":
		t, err := s.get(a[0], "gpu-texture")
		if err != nil {
			return nil, err, true
		}
		v, err := s.put("gpu-texture-view", t.Call("createView"), nil)
		return v, err, true
	case "[method]gpu-texture.width", "[method]gpu-texture.height":
		t, err := s.get(a[0], "gpu-texture")
		if err != nil {
			return nil, err, true
		}
		if name == "[method]gpu-texture.width" {
			return uint64(t.Get("width").Int()), nil, true
		}
		return uint64(t.Get("height").Int()), nil, true
	case "[method]gpu-device.create-render-pipeline":
		v, err := s.createRenderPipeline(a)
		return v, err, true
	case "[method]gpu-render-pipeline.get-bind-group-layout":
		p, err := s.get(a[0], "gpu-render-pipeline")
		if err != nil {
			return nil, err, true
		}
		v, err := s.put("gpu-bind-group-layout", p.Call("getBindGroupLayout", u64(a[1])), nil)
		return v, err, true
	case "[method]gpu-command-encoder.begin-render-pass":
		v, err := s.beginRenderPass(a)
		return v, err, true
	case "[method]gpu-render-pass-encoder.set-pipeline":
		pe, err := s.get(a[0], "gpu-render-pass-encoder")
		if err != nil {
			return nil, err, true
		}
		p, err := s.get(a[1], "gpu-render-pipeline")
		if err != nil {
			return nil, err, true
		}
		pe.Call("setPipeline", p)
		return nil, nil, true
	case "[method]gpu-render-pass-encoder.set-bind-group":
		pe, err := s.get(a[0], "gpu-render-pass-encoder")
		if err != nil {
			return nil, err, true
		}
		bg := js.Null()
		if h, ok := opt(a[2]); ok {
			if bg, err = s.get(h, "gpu-bind-group"); err != nil {
				return nil, err, true
			}
		}
		if _, ok := opt(a[3]); ok {
			return errRecord(0, "dynamic offsets are not supported"), nil, true
		}
		pe.Call("setBindGroup", u64(a[1]), bg)
		return okUnit, nil, true
	case "[method]gpu-render-pass-encoder.draw":
		pe, err := s.get(a[0], "gpu-render-pass-encoder")
		if err != nil {
			return nil, err, true
		}
		args := []any{u64(a[1]), uint64(1), uint64(0), uint64(0)}
		for i := 2; i <= 4; i++ {
			if v, ok := opt(a[i]); ok {
				args[i-1] = u64(v)
			}
		}
		pe.Call("draw", args...)
		return nil, nil, true
	case "[method]gpu-render-pass-encoder.end":
		pe, err := s.get(a[0], "gpu-render-pass-encoder")
		if err != nil {
			return nil, err, true
		}
		pe.Call("end")
		return nil, nil, true
	}
	return nil, nil, false
}

// present paces the surface, waits (bounded) while too many frames are
// unacknowledged, and hands the frame to OnSurface. It runs off the host
// call (async), so its waits never block the engine.
func (s *session) present(c *contextObj) error {
	sf := c.sf
	if wait := s.b.FrameInterval - time.Since(sf.last); wait > 0 {
		time.Sleep(wait)
	}
	if sf.emit != nil {
		sf.mu.Lock()
		for sf.seq-sf.acked >= MaxFramesInFlight {
			ch := sf.ackCh
			sf.mu.Unlock()
			select {
			case <-ch:
			case <-time.After(AckWait):
			}
			sf.mu.Lock()
			if sf.seq-sf.acked >= MaxFramesInFlight {
				sf.acked = sf.seq - MaxFramesInFlight + 1 // give up waiting on this one
			}
		}
		sf.seq++
		seq := sf.seq
		sf.mu.Unlock()
		sf.last = time.Now()
		bm := sf.canvas.Call("transferToImageBitmap")
		sf.emit(SurfaceEvent{Op: "frame", Surface: sf.id, Seq: seq, Width: sf.canvas.Get("width").Int(), Height: sf.canvas.Get("height").Int(), Bitmap: bm})
		return nil
	}
	sf.last = time.Now()
	sf.canvas.Call("transferToImageBitmap").Call("close")
	return nil
}

func (s *session) stage(v any) (js.Value, error) {
	r := record(v)
	mod, err := s.get(r[len(r)-3], "gpu-shader-module")
	if err != nil {
		return js.Value{}, err
	}
	st := obj()
	st.Set("module", mod)
	if e, ok := opt(r[len(r)-2]); ok {
		st.Set("entryPoint", e.(string))
	}
	return st, nil
}

func (s *session) createRenderPipeline(a []any) (any, error) {
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
	vr := record(r[0]) // vertex: buffers, module, entry-point, constants
	if b, ok := opt(vr[0]); ok && len(b.([]any)) > 0 {
		return nil, Trapf("vertex buffers are not implemented yet")
	}
	vertex, err := s.stage(r[0])
	if err != nil {
		return nil, err
	}
	desc := obj()
	desc.Set("vertex", vertex)
	if p, ok := opt(r[1]); ok {
		prim := obj()
		if t, ok := opt(record(p)[0]); ok {
			prim.Set("topology", enumLabel("gpu-primitive-topology", t))
		}
		desc.Set("primitive", prim)
	}
	if _, ok := opt(r[2]); ok {
		return nil, Trapf("depth-stencil states are not implemented yet")
	}
	if f, ok := opt(r[4]); ok {
		frag, err := s.stage(f)
		if err != nil {
			return nil, err
		}
		var targets []any
		for _, t := range record(f)[0].([]any) {
			tv, ok := opt(t)
			if !ok {
				targets = append(targets, nil)
				continue
			}
			tt := obj()
			tt.Set("format", formatName(record(tv)[0]))
			targets = append(targets, tt)
		}
		frag.Set("targets", targets)
		desc.Set("fragment", frag)
	}
	lm, _ := r[5].(wit.VariantValue)
	if lm.Case == 0 {
		pl, err := s.get(lm.Value, "gpu-pipeline-layout")
		if err != nil {
			return nil, err
		}
		desc.Set("layout", pl)
	} else {
		desc.Set("layout", "auto")
	}
	return s.put("gpu-render-pipeline", d.Call("createRenderPipeline", desc), func() {
		s.mu.Lock()
		s.pipelines--
		s.mu.Unlock()
	})
}

func (s *session) beginRenderPass(a []any) (any, error) {
	e, err := s.get(a[0], "gpu-command-encoder")
	if err != nil {
		return nil, err
	}
	r := record(a[1])
	var atts []any
	for _, ca := range r[0].([]any) {
		cv, ok := opt(ca)
		if !ok {
			atts = append(atts, nil)
			continue
		}
		c := record(cv) // view, depth-slice, resolve-target, clear-value, load-op, store-op
		view, err := s.get(c[0], "gpu-texture-view")
		if err != nil {
			return nil, err
		}
		at := obj()
		at.Set("view", view)
		if cl, ok := opt(c[3]); ok {
			col := record(cl)
			cc := obj()
			cc.Set("r", col[0].(float64))
			cc.Set("g", col[1].(float64))
			cc.Set("b", col[2].(float64))
			cc.Set("a", col[3].(float64))
			at.Set("clearValue", cc)
		}
		at.Set("loadOp", enumLabel("gpu-load-op", c[4]))
		at.Set("storeOp", enumLabel("gpu-store-op", c[5]))
		atts = append(atts, at)
	}
	if _, ok := opt(r[1]); ok {
		return nil, Trapf("depth-stencil attachments are not implemented yet")
	}
	desc := obj()
	desc.Set("colorAttachments", atts)
	return s.put("gpu-render-pass-encoder", e.Call("beginRenderPass", desc), nil)
}
