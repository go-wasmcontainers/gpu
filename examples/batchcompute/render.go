//go:build !js

package main

import (
	"fmt"
	"math"
	"os"

	"github.com/go-gfx/webgpu"
	"github.com/go-wasmcontainers/gpu/wasigpu/batchsink"
)

// The render mode is the probe for drawing through the batch stream (Quake
// level 2: go-quake1/engine's gpuworld draws r8uint textures with vertex and
// index buffers, explicit layouts and DrawIndexed). One submit:
//
//  1. WriteTexture a 4x4 r8uint source texture;
//  2. a render pass draws a quad (4 vertices, 6 indices, DrawIndexed) into a
//     64x64 r8uint target; each pixel is src[x%4, y%4] + x/4, so a transposed
//     axis or an off-by-one row cannot pass;
//  3. CopyTextureToBuffer copies the target into a MAP_READ buffer (rows
//     padded to 256 bytes) and the stream reads it back.
//
// The fragment shader has a constant-bound loop of at most 64 iterations with
// a break (go-gfx/webgpu batch.MaxRenderLoop), the one shape the host accepts
// in a render module, and the draw is counted against the draw quota.
const (
	renderSize = 64
	srcSize    = 4
)

const renderShader = `
@group(0) @binding(0) var src: texture_2d<u32>;
struct VOut { @builtin(position) pos: vec4<f32> };
@vertex fn vs(@location(0) p: vec2<f32>) -> VOut {
  var o: VOut;
  o.pos = vec4<f32>(p, 0.0, 1.0);
  return o;
}
@fragment fn fs(i: VOut) -> @location(0) vec4<u32> {
  let x = u32(i.pos.x);
  let y = u32(i.pos.y);
  let v = textureLoad(src, vec2<u32>(x % 4u, y % 4u), 0).x;
  // x / 4 counted by a bounded loop (x < 256).
  var q = 0u;
  for (var k = 0u; k < 64u; k++) {
    if (k >= x / 4u) { break; }
    q++;
  }
  return vec4<u32>(v + q, 0u, 0u, 0u);
}`

// srcTexel is the source texture's value at (x, y).
func srcTexel(x, y int) uint32 { return uint32((y*srcSize+x)*7%200) + 1 }

// wantPixel is what the render pass must leave at (x, y).
func wantPixel(x, y int) uint32 { return srcTexel(x%srcSize, y%srcSize) + uint32(x/4) }

func render(s *batchsink.Sink) {
	d := s.Device("batchrender")
	tex := func(w, h uint32, usage webgpu.TextureUsage) *webgpu.Texture {
		t, err := d.CreateTexture(webgpu.TextureDescriptor{Size: webgpu.Extent3D{Width: w, Height: h},
			Format: webgpu.TextureFormatR8Uint, Usage: usage})
		must(err)
		return t
	}
	view := func(t *webgpu.Texture) *webgpu.TextureView {
		v, err := t.CreateView(nil)
		must(err)
		return v
	}
	buf := func(size uint64, usage webgpu.BufferUsage) *webgpu.Buffer {
		b, err := d.CreateBuffer(webgpu.BufferDescriptor{Size: size, Usage: usage})
		must(err)
		return b
	}
	src := tex(srcSize, srcSize, webgpu.TextureUsageTextureBinding|webgpu.TextureUsageCopyDst)
	tgt := tex(renderSize, renderSize, webgpu.TextureUsageRenderAttachment|webgpu.TextureUsageCopySrc)
	srcView, tgtView := view(src), view(tgt)

	uintTex := &webgpu.TextureBindingLayout{SampleType: webgpu.TextureSampleTypeUint, ViewDimension: webgpu.TextureViewDimension2D}
	rbgl, err := d.CreateBindGroupLayout(webgpu.BindGroupLayoutDescriptor{Entries: []webgpu.BindGroupLayoutEntry{
		{Binding: 0, Visibility: webgpu.ShaderStageFragment, Texture: uintTex}}})
	must(err)
	rpl, err := d.CreatePipelineLayout(webgpu.PipelineLayoutDescriptor{BindGroupLayouts: []*webgpu.BindGroupLayout{rbgl}})
	must(err)
	rm, err := d.CreateShaderModule(webgpu.ShaderModuleDescriptor{Code: renderShader})
	must(err)
	pipe, err := d.CreateRenderPipeline(webgpu.RenderPipelineDescriptor{
		Layout: rpl,
		Vertex: webgpu.VertexState{Module: rm, EntryPoint: "vs", Buffers: []webgpu.VertexBufferLayout{{
			ArrayStride: 8,
			Attributes:  []webgpu.VertexAttribute{{Format: webgpu.VertexFormatFloat32x2, ShaderLocation: 0}},
		}}},
		Primitive: webgpu.PrimitiveState{Topology: webgpu.PrimitiveTopologyTriangleList},
		Fragment: &webgpu.FragmentState{Module: rm, EntryPoint: "fs",
			Targets: []webgpu.ColorTargetState{{Format: webgpu.TextureFormatR8Uint}}},
	})
	must(err)
	rbg, err := d.CreateBindGroup(webgpu.BindGroupDescriptor{Layout: rbgl, Entries: []webgpu.BindGroupEntry{
		{Binding: 0, TextureView: srcView}}})
	must(err)

	const pixels = renderSize * renderSize
	const pitch = 256 // bytesPerRow of a texture-to-buffer copy: a multiple of 256
	read := buf(pitch*renderSize, webgpu.BufferUsageMapRead|webgpu.BufferUsageCopyDst)

	verts := buf(4*8, webgpu.BufferUsageVertex|webgpu.BufferUsageCopyDst)
	idx := buf(6*4, webgpu.BufferUsageIndex|webgpu.BufferUsageCopyDst)

	srcData := make([]byte, srcSize*srcSize)
	for y := 0; y < srcSize; y++ {
		for x := 0; x < srcSize; x++ {
			srcData[y*srcSize+x] = byte(srcTexel(x, y))
		}
	}
	rec := d.NewRecorder()
	rec.WriteTexture(webgpu.TexelCopyTextureInfo{Texture: src}, srcData,
		webgpu.TexelCopyBufferLayout{BytesPerRow: srcSize, RowsPerImage: srcSize},
		webgpu.Extent3D{Width: srcSize, Height: srcSize})
	rec.WriteBuffer(verts, 0, f32le(-1, -1, 1, -1, -1, 1, 1, 1))
	rec.WriteBuffer(idx, 0, u32le(0, 1, 2, 2, 1, 3))
	p := rec.BeginRenderPass(webgpu.RenderPassDescriptor{ColorAttachments: []*webgpu.RenderPassColorAttachment{{
		View: tgtView, LoadOp: webgpu.LoadOpClear, StoreOp: webgpu.StoreOpStore}}})
	p.SetPipeline(pipe)
	p.SetBindGroup(0, rbg)
	p.SetVertexBuffer(0, verts, 0, 4*8)
	p.SetIndexBuffer(idx, webgpu.IndexFormatUint32, 0, 6*4)
	p.DrawIndexed(6, 1, 0, 0, 0)
	must(p.End())
	rec.CopyTextureToBuffer(webgpu.TexelCopyTextureInfo{Texture: tgt}, read,
		webgpu.TexelCopyBufferLayout{BytesPerRow: pitch, RowsPerImage: renderSize},
		webgpu.Extent3D{Width: renderSize, Height: renderSize})
	rb := rec.ReadBuffer(read, 0, pitch*renderSize)
	before := s.Submits()
	_, err = rec.Submit()
	must(err)
	got, err := rb.Bytes()
	must(err)

	bad := 0
	for i := 0; i < pixels; i++ {
		x, y := i%renderSize, i/renderSize
		v := uint32(got[y*pitch+x])
		if v != wantPixel(x, y) {
			if bad < 5 {
				fmt.Printf("pixel (%d,%d): %d, want %d\n", x, y, v, wantPixel(x, y))
			}
			bad++
		}
	}
	if bad > 0 {
		fmt.Printf("%d of %d pixels differ\n", bad, pixels)
		os.Exit(1)
	}
	fmt.Printf("render: %dx%d r8uint drawn with DrawIndexed from a WriteTexture source and read back with CopyTextureToBuffer, %d pixels match the CPU, %d host crossing(s)\n",
		renderSize, renderSize, pixels, s.Submits()-before)
}

func f32le(v ...float32) []byte {
	b := make([]byte, 0, 4*len(v))
	for _, f := range v {
		b = appendU32(b, math.Float32bits(f))
	}
	return b
}

func u32le(v ...uint32) []byte {
	b := make([]byte, 0, 4*len(v))
	for _, u := range v {
		b = appendU32(b, u)
	}
	return b
}

func appendU32(b []byte, u uint32) []byte {
	return append(b, byte(u), byte(u>>8), byte(u>>16), byte(u>>24))
}
