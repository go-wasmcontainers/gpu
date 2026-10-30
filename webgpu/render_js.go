//go:build js

package webgpu

import (
	"syscall/js"
	"time"

	wg "github.com/go-gfx/webgpu"
	"github.com/go-wasmcontainers/gpu/wit"
)

// The rendering subset: surfaces (an OffscreenCanvas each, in the engine's
// worker), their WebGPU contexts (wg.Surface), render pipelines without
// vertex buffers, render passes, draws; frames leave through
// Browser.OnSurface as ImageBitmaps, paced to Browser.FrameInterval.

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
	if kind := inputKind(name); kind != "" {
		o, err := s.getObj(a[0], "surface")
		if err != nil {
			return nil, err, true
		}
		st, err := o.(*surfaceObj).input.openStream(kind)
		if err != nil {
			return nil, err, true
		}
		n, err := s.h.New("stream", st)
		if err != nil {
			st.Destroy()
			return nil, err, true
		}
		return n, nil, true
	}
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
			sf.emit(SurfaceEvent{Op: "open", Surface: sf.id, Container: s.container, Width: w, Height: h})
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
		sf := o.(*surfaceObj)
		c := sf.canvas
		h, w, err := s.size(a[1], a[2], c.Get("height").Int(), c.Get("width").Int())
		if err != nil {
			return nil, err, true
		}
		if sf.surf != nil {
			if err := sf.surf.Resize(w, h); err != nil {
				return nil, trap("request-set-size", err), true
			}
		} else {
			c.Set("width", w)
			c.Set("height", h)
		}
		if sf.emit != nil {
			sf.emit(SurfaceEvent{Op: "resize", Surface: sf.id, Width: w, Height: h})
		}
		return nil, nil, true
	case "[constructor]context":
		o, err := s.getObj(a[0], "surface")
		if err != nil {
			return nil, err, true
		}
		if err := wg.CanvasSupport(); err != nil {
			return nil, Trapf("no WebGPU canvas context here: %v", err), true
		}
		v, err := s.h.New("context", &contextObj{sf: o.(*surfaceObj)})
		return v, err, true
	case "[method]context.configure":
		o, err := s.getObj(a[0], "context")
		if err != nil {
			return nil, err, true
		}
		r := record(a[1])
		dev, err := get[*wg.Device](s, r[0], "gpu-device")
		if err != nil {
			return nil, err, true
		}
		desc := wg.SurfaceDescriptor{Format: wg.TextureFormat(formatName(r[1])), AlphaMode: wg.CanvasAlphaModePremultiplied}
		if u, ok := opt(r[2]); ok {
			desc.Usage = wg.TextureUsage(u64(u))
		}
		if m, ok := opt(r[6]); ok {
			desc.AlphaMode = wg.CanvasAlphaMode(enumLabel("gpu-canvas-alpha-mode", m))
		}
		sf := o.(*contextObj).sf
		surf, err := dev.NewSurfaceFromCanvas(sf.canvas, desc)
		if err != nil {
			return nil, trap("configure", err), true
		}
		sf.surf = surf
		return nil, nil, true
	case "[method]context.unconfigure":
		o, err := s.getObj(a[0], "context")
		if err != nil {
			return nil, err, true
		}
		if sf := o.(*contextObj).sf; sf.surf != nil {
			sf.surf.Destroy()
			sf.surf = nil
		}
		return nil, nil, true
	case "[method]context.get-current-texture":
		o, err := s.getObj(a[0], "context")
		if err != nil {
			return nil, err, true
		}
		sf := o.(*contextObj).sf
		if sf.surf == nil {
			return nil, Trapf("get-current-texture: the context is not configured"), true
		}
		t, err := sf.surf.CurrentTexture()
		if err != nil {
			return nil, trap("get-current-texture", err), true
		}
		v, err := s.put("gpu-texture", t, nil)
		return v, err, true
	case "[method]context.present":
		o, err := s.getObj(a[0], "context")
		if err != nil {
			return nil, err, true
		}
		return nil, s.present(o.(*contextObj)), true
	case "[method]gpu-texture.create-view":
		t, err := get[*wg.Texture](s, a[0], "gpu-texture")
		if err != nil {
			return nil, err, true
		}
		tv, err := t.CreateView(nil)
		if err != nil {
			return nil, trap("create-view", err), true
		}
		v, err := s.put("gpu-texture-view", tv, nil)
		return v, err, true
	case "[method]gpu-texture.width", "[method]gpu-texture.height":
		t, err := get[*wg.Texture](s, a[0], "gpu-texture")
		if err != nil {
			return nil, err, true
		}
		if name == "[method]gpu-texture.width" {
			return uint64(t.Width()), nil, true
		}
		return uint64(t.Height()), nil, true
	case "[method]gpu-device.create-render-pipeline":
		v, err := s.createRenderPipeline(a)
		return v, err, true
	case "[method]gpu-render-pipeline.get-bind-group-layout":
		p, err := get[*wg.RenderPipeline](s, a[0], "gpu-render-pipeline")
		if err != nil {
			return nil, err, true
		}
		l, err := p.GetBindGroupLayout(uint32(u64(a[1])))
		if err != nil {
			return nil, trap("get-bind-group-layout", err), true
		}
		v, err := s.put("gpu-bind-group-layout", l, nil)
		return v, err, true
	case "[method]gpu-command-encoder.begin-render-pass":
		v, err := s.beginRenderPass(a)
		return v, err, true
	case "[method]gpu-render-pass-encoder.set-pipeline":
		pe, err := get[*wg.RenderPassEncoder](s, a[0], "gpu-render-pass-encoder")
		if err != nil {
			return nil, err, true
		}
		p, err := get[*wg.RenderPipeline](s, a[1], "gpu-render-pipeline")
		if err != nil {
			return nil, err, true
		}
		pe.SetPipeline(p)
		return nil, nil, true
	case "[method]gpu-render-pass-encoder.set-bind-group":
		pe, err := get[*wg.RenderPassEncoder](s, a[0], "gpu-render-pass-encoder")
		if err != nil {
			return nil, err, true
		}
		var bg *wg.BindGroup
		if h, ok := opt(a[2]); ok {
			if bg, err = get[*wg.BindGroup](s, h, "gpu-bind-group"); err != nil {
				return nil, err, true
			}
		}
		pe.SetBindGroup(uint32(u64(a[1])), bg, dynamicOffsets(a[3])...)
		return okUnit, nil, true
	case "[method]gpu-render-pass-encoder.draw":
		pe, err := get[*wg.RenderPassEncoder](s, a[0], "gpu-render-pass-encoder")
		if err != nil {
			return nil, err, true
		}
		args := [4]uint32{uint32(u64(a[1])), 1, 0, 0}
		for i := 2; i <= 4; i++ {
			if v, ok := opt(a[i]); ok {
				args[i-1] = uint32(u64(v))
			}
		}
		pe.Draw(args[0], args[1], args[2], args[3])
		return nil, nil, true
	case "[method]gpu-render-pass-encoder.end":
		pe, err := get[*wg.RenderPassEncoder](s, a[0], "gpu-render-pass-encoder")
		if err != nil {
			return nil, err, true
		}
		if err := pe.End(); err != nil {
			return nil, trap("render pass end", err), true
		}
		return nil, nil, true
	}
	return nil, nil, false
}

// present paces the surface, waits (bounded) while too many frames are
// unacknowledged, and hands the frame to OnSurface. It runs off the host
// call (async), so its waits never block the engine. The canvas is this
// package's own (it is created before any device configures it), so the
// frame is moved out with transferToImageBitmap here.
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

// stage reads a vertex or fragment state's module and entry point (the
// last three fields: module, entry-point, constants).
func (s *session) stage(v any) (*wg.ShaderModule, string, error) {
	r := record(v)
	mod, err := get[*wg.ShaderModule](s, r[len(r)-3], "gpu-shader-module")
	if err != nil {
		return nil, "", err
	}
	entry := ""
	if e, ok := opt(r[len(r)-2]); ok {
		entry, _ = e.(string)
	}
	return mod, entry, nil
}

func (s *session) createRenderPipeline(a []any) (any, error) {
	d, err := get[*wg.Device](s, a[0], "gpu-device")
	if err != nil {
		return nil, err
	}
	r := record(a[1])
	vr := record(r[0]) // vertex: buffers, module, entry-point, constants
	if b, ok := opt(vr[0]); ok && len(b.([]any)) > 0 {
		return nil, Trapf("vertex buffers are not implemented yet")
	}
	mod, entry, err := s.stage(r[0])
	if err != nil {
		return nil, err
	}
	desc := wg.RenderPipelineDescriptor{Vertex: wg.VertexState{Module: mod, EntryPoint: entry}}
	if p, ok := opt(r[1]); ok {
		if t, ok := opt(record(p)[0]); ok {
			desc.Primitive.Topology = wg.PrimitiveTopology(enumLabel("gpu-primitive-topology", t))
		}
	}
	if _, ok := opt(r[2]); ok {
		return nil, Trapf("depth-stencil states are not implemented yet")
	}
	if f, ok := opt(r[4]); ok {
		fmod, fentry, err := s.stage(f)
		if err != nil {
			return nil, err
		}
		frag := &wg.FragmentState{Module: fmod, EntryPoint: fentry}
		for _, t := range record(f)[0].([]any) {
			tv, ok := opt(t)
			if !ok {
				return nil, Trapf("empty color target slots are not implemented yet")
			}
			frag.Targets = append(frag.Targets, wg.ColorTargetState{Format: wg.TextureFormat(formatName(record(tv)[0]))})
		}
		desc.Fragment = frag
	}
	if lm, _ := r[5].(wit.VariantValue); lm.Case == 0 {
		if desc.Layout, err = get[*wg.PipelineLayout](s, lm.Value, "gpu-pipeline-layout"); err != nil {
			return nil, err
		}
	}
	p, err := d.CreateRenderPipeline(desc)
	if err != nil {
		return nil, trap("create-render-pipeline", err)
	}
	return s.put("gpu-render-pipeline", p, p.Release)
}

func (s *session) beginRenderPass(a []any) (any, error) {
	e, err := get[*wg.CommandEncoder](s, a[0], "gpu-command-encoder")
	if err != nil {
		return nil, err
	}
	r := record(a[1])
	var desc wg.RenderPassDescriptor
	for _, ca := range r[0].([]any) {
		cv, ok := opt(ca)
		if !ok {
			desc.ColorAttachments = append(desc.ColorAttachments, nil)
			continue
		}
		c := record(cv) // view, depth-slice, resolve-target, clear-value, load-op, store-op
		view, err := get[*wg.TextureView](s, c[0], "gpu-texture-view")
		if err != nil {
			return nil, err
		}
		at := &wg.RenderPassColorAttachment{View: view,
			LoadOp:  wg.LoadOp(enumLabel("gpu-load-op", c[4])),
			StoreOp: wg.StoreOp(enumLabel("gpu-store-op", c[5]))}
		if cl, ok := opt(c[3]); ok {
			col := record(cl)
			at.ClearValue = wg.Color{R: col[0].(float64), G: col[1].(float64), B: col[2].(float64), A: col[3].(float64)}
		}
		desc.ColorAttachments = append(desc.ColorAttachments, at)
	}
	if _, ok := opt(r[1]); ok {
		return nil, Trapf("depth-stencil attachments are not implemented yet")
	}
	p, err := e.BeginRenderPass(desc)
	if err != nil {
		return nil, trap("begin-render-pass", err)
	}
	return s.put("gpu-render-pass-encoder", p, nil)
}
