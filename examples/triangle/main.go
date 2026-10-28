// Command triangle is a WASI container that draws a spinning triangle on a
// surface through wasi:webgpu and wasi-gfx:surface: in the playground the
// surface is a pane of the dashboard. It draws a number of frames (120 by
// default, or the first argument) and reports the rate the host paced it
// to.
//
//	GOOS=wasip1 GOARCH=wasm go build -o triangle.wasm ./examples/triangle
//	wctr run --gpu IMAGE [frames]
package main

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"strconv"
	"time"

	"github.com/go-wasmcontainers/gpu/wasigpu"
)

const shader = `
struct U { angle: f32 };
@group(0) @binding(0) var<uniform> u: U;

struct Out { @builtin(position) pos: vec4<f32>, @location(0) color: vec3<f32> };

@vertex fn vs(@builtin(vertex_index) i: u32) -> Out {
  var p = array<vec2<f32>, 3>(vec2(0.0, 0.7), vec2(-0.6, -0.5), vec2(0.6, -0.5));
  var c = array<vec3<f32>, 3>(vec3(1.0, 0.2, 0.2), vec3(0.2, 1.0, 0.2), vec3(0.2, 0.3, 1.0));
  let s = sin(u.angle); let k = cos(u.angle);
  let v = p[i];
  var o: Out;
  o.pos = vec4(v.x * k - v.y * s, v.x * s + v.y * k, 0.0, 1.0);
  o.color = c[i];
  return o;
}

@fragment fn fs(in: Out) -> @location(0) vec4<f32> { return vec4(in.color, 1.0); }
`

func main() {
	frames := 120
	if len(os.Args) > 1 {
		if n, err := strconv.Atoi(os.Args[1]); err == nil && n > 0 {
			frames = n
		}
	}
	d, err := wasigpu.Open()
	if errors.Is(err, wasigpu.ErrNoAdapter) {
		fmt.Println("no GPU adapter")
		os.Exit(2)
	}
	if err != nil {
		fail(err)
	}
	defer d.Drop()
	s, err := d.Surface(320, 240)
	if err != nil {
		fail(err)
	}
	defer s.Drop()
	p, err := s.RenderPipeline(shader, "vs", "fs")
	if err != nil {
		fail(err)
	}
	defer p.Drop()
	u, err := d.Buffer(16, wasigpu.Uniform|wasigpu.CopyDst)
	if err != nil {
		fail(err)
	}
	defer u.Drop()
	t0 := time.Now()
	var b [16]byte
	for f := 0; f < frames; f++ {
		binary.LittleEndian.PutUint32(b[:], math.Float32bits(float32(f)*0.05))
		if err := d.Write(u, 0, b[:]); err != nil {
			fail(err)
		}
		if err := s.Frame(p, []*wasigpu.Buffer{u}, [4]float64{0.05, 0.05, 0.1, 1}, 3); err != nil {
			fail(err)
		}
	}
	el := time.Since(t0)
	fmt.Printf("%d frames in %v: %.1f frames per second\n", frames, el.Round(time.Millisecond), float64(frames)/el.Seconds())
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "triangle:", err)
	os.Exit(1)
}
