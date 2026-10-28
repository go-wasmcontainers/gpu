package wasigpu

import "github.com/go-wasmcontainers/gpu/wit"

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

// Frame clears the surface to clear (RGBA in [0, 1]), draws vertices with
// p (buffers bound at group 0, bindings 0.. in order; none for none) and
// presents the frame. Present is paced by the host (60 per second at
// most in the playground), so a loop of Frame calls is an animation.
func (s *Surface) Frame(p *RenderPipeline, buffers []*Buffer, clear [4]float64, vertices uint32) error {
	tex, err := handle(callIn(contextModule, "[method]context.get-current-texture", s.ctx))
	if err != nil {
		return err
	}
	defer drop("gpu-texture", tex)
	view, err := handle(call("[method]gpu-texture.create-view", tex, nil))
	if err != nil {
		return err
	}
	defer drop("gpu-texture-view", view)
	enc, err := handle(call("[method]gpu-device.create-command-encoder", s.d.dev, nil))
	if err != nil {
		return err
	}
	defer drop("gpu-command-encoder", enc)
	color := wit.Some{Value: []any{clear[0], clear[1], clear[2], clear[3]}}
	att := wit.Some{Value: []any{view, nil, nil, color, enum("gpu-load-op", "clear"), enum("gpu-store-op", "store")}}
	pass, err := handle(call("[method]gpu-command-encoder.begin-render-pass", enc, []any{[]any{att}, nil, nil, nil, nil, nil}))
	if err != nil {
		return err
	}
	defer drop("gpu-render-pass-encoder", pass)
	if _, err := call("[method]gpu-render-pass-encoder.set-pipeline", pass, p.pipe); err != nil {
		return err
	}
	if len(buffers) > 0 {
		var entries []any
		for i, b := range buffers {
			entries = append(entries, []any{uint64(i), wit.VariantValue{Case: 0, Value: b.h}})
		}
		bg, err := handle(call("[method]gpu-device.create-bind-group", s.d.dev, []any{p.layout, entries, nil}))
		if err != nil {
			return err
		}
		defer drop("gpu-bind-group", bg)
		if _, err := unwrap("set-bind-group")(call("[method]gpu-render-pass-encoder.set-bind-group", pass, uint64(0), wit.Some{Value: bg}, nil, nil, nil)); err != nil {
			return err
		}
	}
	if _, err := call("[method]gpu-render-pass-encoder.draw", pass, uint64(vertices), nil, nil, nil); err != nil {
		return err
	}
	if _, err := call("[method]gpu-render-pass-encoder.end", pass); err != nil {
		return err
	}
	if err := s.d.submit(enc); err != nil {
		return err
	}
	_, err = callIn(contextModule, "[method]context.present", s.ctx)
	return err
}
