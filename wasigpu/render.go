package wasigpu

import (
	"sync/atomic"
	"unsafe"

	"github.com/go-wasmcontainers/gpu/wit"
)

// Rendering to a surface (wasi-gfx:surface, surface-webgpu): a surface the
// host shows (in the playground, a pane of the dashboard), a WebGPU context
// on it, render pipelines without vertex buffers, and a draw-and-present.

// Surface is a window-like target and its configured context.
type Surface struct {
	d       *Device
	sf, ctx uint32
	W, H    uint32
	format  uint32
}

// Surface creates a surface of w x h pixels configured for d.
func (d *Device) Surface(w, h uint32) (*Surface, error) {
	sf, err := handle(callIn(surfaceModule, "[constructor]surface", []any{wit.Some{Value: uint64(h)}, wit.Some{Value: uint64(w)}}))
	if err != nil {
		return nil, err
	}
	ctx, err := handle(callIn(contextModule, "[constructor]context", sf))
	if err != nil {
		dropIn(surfaceModule, "surface", sf)
		return nil, err
	}
	s := &Surface{d: d, sf: sf, ctx: ctx, W: w, H: h, format: d.format}
	cfg := []any{d.dev, s.format, nil, nil, nil, nil, wit.Some{Value: enum("gpu-canvas-alpha-mode", "opaque")}}
	if _, err := callIn(contextModule, "[method]context.configure", ctx, cfg); err != nil {
		s.Drop()
		return nil, err
	}
	return s, nil
}

// Drop releases the surface.
func (s *Surface) Drop() {
	dropIn(contextModule, "context", s.ctx)
	dropIn(surfaceModule, "surface", s.sf)
}

// RenderPipeline is a render pipeline and its first bind group's layout.
type RenderPipeline struct {
	module, pipe, layout uint32
}

// RenderPipeline compiles WGSL with a vertex and a fragment entry point
// drawing triangle lists into s's format.
func (s *Surface) RenderPipeline(wgsl, vertex, fragment string) (*RenderPipeline, error) {
	mod, err := handle(call("[method]gpu-device.create-shader-module", s.d.dev, []any{wgsl, nil, nil}))
	if err != nil {
		return nil, err
	}
	vs := []any{nil, mod, wit.Some{Value: vertex}, nil}
	prim := wit.Some{Value: []any{wit.Some{Value: enum("gpu-primitive-topology", "triangle-list")}, nil, nil, nil, nil}}
	target := wit.Some{Value: []any{s.format, nil, nil}}
	fs := wit.Some{Value: []any{[]any{target}, mod, wit.Some{Value: fragment}, nil}}
	desc := []any{vs, prim, nil, nil, fs, wit.VariantValue{Case: 1}, nil}
	pipe, err := handle(call("[method]gpu-device.create-render-pipeline", s.d.dev, desc))
	if err != nil {
		drop("gpu-shader-module", mod)
		return nil, err
	}
	bgl, err := handle(call("[method]gpu-render-pipeline.get-bind-group-layout", pipe, uint64(0)))
	if err != nil {
		drop("gpu-render-pipeline", pipe)
		drop("gpu-shader-module", mod)
		return nil, err
	}
	return &RenderPipeline{module: mod, pipe: pipe, layout: bgl}, nil
}

// Drop releases the pipeline.
func (p *RenderPipeline) Drop() {
	drop("gpu-bind-group-layout", p.layout)
	drop("gpu-render-pipeline", p.pipe)
	drop("gpu-shader-module", p.module)
}

// ReadyToPresent reports whether Frame would hand its frame to the page without
// waiting for the page to acknowledge an earlier one (an extension of this
// host, not in wasi-gfx). A guest with work of its own while it waits (a sound
// mix to keep filled) polls it in slices and does that work between them,
// instead of blocking inside Frame.
func (s *Surface) ReadyToPresent() (bool, error) {
	v, err := callIn(contextModule, "[method]context.ready-to-present", s.ctx)
	if err != nil {
		return false, err
	}
	b, _ := v.(bool)
	return b, nil
}

// Frame clears the surface to clear (RGBA in [0, 1]), draws vertices with
// p (buffers bound at group 0, bindings 0.. in order; none for none) and
// presents the frame. Present is paced by the host (60 per second at
// most in the playground), so a loop of Frame calls is an animation.
func (s *Surface) Frame(p *RenderPipeline, buffers []*Buffer, clear [4]float64, vertices uint32) error {
	var drops dropList
	defer drops.run()
	cb, err := s.record(&drops, p, buffers, clear, vertices)
	if err != nil {
		return err
	}
	if _, err := call("[method]gpu-queue.submit", s.d.queue, []any{cb}); err != nil {
		return err
	}
	_, err = callIn(contextModule, "[method]context.present", s.ctx)
	return err
}

// dropList is the objects of a frame to release, in the reverse order of their
// creation, once the frame is over.
type dropList []func()

func (l *dropList) add(res string, h uint32) { *l = append(*l, func() { drop(res, h) }) }

func (l *dropList) run() {
	for i := len(*l) - 1; i >= 0; i-- {
		(*l)[i]()
	}
	*l = nil
}

// record encodes the frame's render pass and returns its command buffer; every
// object it creates is on drops.
func (s *Surface) record(drops *dropList, p *RenderPipeline, buffers []*Buffer, clear [4]float64, vertices uint32) (uint32, error) {
	tex, err := handle(callIn(contextModule, "[method]context.get-current-texture", s.ctx))
	if err != nil {
		return 0, err
	}
	drops.add("gpu-texture", tex)
	view, err := handle(call("[method]gpu-texture.create-view", tex, nil))
	if err != nil {
		return 0, err
	}
	drops.add("gpu-texture-view", view)
	enc, err := handle(call("[method]gpu-device.create-command-encoder", s.d.dev, nil))
	if err != nil {
		return 0, err
	}
	drops.add("gpu-command-encoder", enc)
	color := wit.Some{Value: []any{clear[0], clear[1], clear[2], clear[3]}}
	att := wit.Some{Value: []any{view, nil, nil, color, enum("gpu-load-op", "clear"), enum("gpu-store-op", "store")}}
	pass, err := handle(call("[method]gpu-command-encoder.begin-render-pass", enc, []any{[]any{att}, nil, nil, nil, nil, nil}))
	if err != nil {
		return 0, err
	}
	drops.add("gpu-render-pass-encoder", pass)
	if _, err := call("[method]gpu-render-pass-encoder.set-pipeline", pass, p.pipe); err != nil {
		return 0, err
	}
	if len(buffers) > 0 {
		var entries []any
		for i, b := range buffers {
			entries = append(entries, []any{uint64(i), wit.VariantValue{Case: 0, Value: b.h}})
		}
		bg, err := handle(call("[method]gpu-device.create-bind-group", s.d.dev, []any{p.layout, entries, nil}))
		if err != nil {
			return 0, err
		}
		drops.add("gpu-bind-group", bg)
		if _, err := unwrap("set-bind-group")(call("[method]gpu-render-pass-encoder.set-bind-group", pass, uint64(0), wit.Some{Value: bg}, nil, nil, nil)); err != nil {
			return 0, err
		}
	}
	if _, err := call("[method]gpu-render-pass-encoder.draw", pass, uint64(vertices), nil, nil, nil); err != nil {
		return 0, err
	}
	if _, err := call("[method]gpu-render-pass-encoder.end", pass); err != nil {
		return 0, err
	}
	cb, err := handle(call("[method]gpu-command-encoder.finish", enc, nil))
	if err != nil {
		return 0, err
	}
	drops.add("gpu-command-buffer", cb)
	return cb, nil
}

// PendingFrame is a frame BeginFrame started: the host submits it and presents
// it on its own while the guest works, and Ready says when that is over.
type PendingFrame struct {
	s     *Surface
	done  *uint32 // the host stores 1 here when the frame is over
	drops dropList
}

// BeginFrame is Frame without the wait (an extension of this host, not in
// wasi-gfx; engine#160). It encodes the frame, then asks the host to submit and
// present it and returns at once: the browser may hold the submit and the
// present for hundreds of milliseconds, a span in which a guest that must keep
// a sound output fed has to run. Poll Ready (it reads the guest's own memory,
// it is not a host call, so the host thread the browser holds does not delay
// it) and call End; no other call that uses the device may come in between.
func (s *Surface) BeginFrame(p *RenderPipeline, buffers []*Buffer, clear [4]float64, vertices uint32) (*PendingFrame, error) {
	f := &PendingFrame{s: s, done: new(uint32)}
	cb, err := s.record(&f.drops, p, buffers, clear, vertices)
	if err != nil {
		f.drops.run()
		return nil, err
	}
	if _, err := callIn(contextModule, "[method]context.submit-present-begin", s.ctx, s.d.queue, []any{cb}, uint32(uintptr(unsafe.Pointer(f.done)))); err != nil {
		f.drops.run()
		return nil, err
	}
	return f, nil
}

// Ready reports whether the host is done with the frame (End will not wait).
func (f *PendingFrame) Ready() bool { return atomic.LoadUint32(f.done) == 1 }

// End waits for the end of the frame and releases what it held. It returns the
// failure of the submit or of the present, if any.
func (f *PendingFrame) End() error {
	_, err := callIn(contextModule, "[method]context.submit-present-end", f.s.ctx)
	f.drops.run()
	return err
}
