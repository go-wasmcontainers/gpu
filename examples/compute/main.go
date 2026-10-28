// Command compute is a WASI container that computes on the GPU through
// wasi:webgpu and checks every result against the CPU: a vector add of
// 1M floats and a 256x256 matrix product.
//
//	GOOS=wasip1 GOARCH=wasm go build -o compute.wasm ./examples/compute
//	wctr run --gpu IMAGE
//
// It exits 0 when the GPU matched the CPU, 1 on a mismatch or an error,
// and 2 (printing "no GPU adapter") when the engine gives it no GPU: no
// --gpu, or no WebGPU where it runs.
package main

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"time"

	"github.com/go-wasmcontainers/gpu/wasigpu"
)

const vecAdd = `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x + g.y * 4194240u;
  if (i < arrayLength(&c)) { c[i] = a[i] + b[i]; }
}`

const matMul = `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
const N = 256u;
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let row = g.y; let col = g.x;
  if (row >= N || col >= N) { return; }
  var acc = 0.0;
  for (var k = 0u; k < N; k++) { acc += a[row * N + k] * b[k * N + col]; }
  c[row * N + col] = acc;
}`

func main() {
	d, err := wasigpu.Open()
	if errors.Is(err, wasigpu.ErrNoAdapter) {
		fmt.Println("no GPU adapter")
		os.Exit(2)
	}
	if err != nil {
		fail(err)
	}
	defer d.Drop()
	if len(os.Args) > 1 && os.Args[1] == "quota" {
		// Past the engine's GPU memory quota (256 MiB per process) the
		// host traps the guest: this never prints "done".
		for i := 0; i < 8; i++ {
			if _, err := d.Buffer(64<<20, wasigpu.Storage); err != nil {
				fail(err)
			}
			fmt.Println("buffer", i)
		}
		fmt.Println("done")
		return
	}
	if err := run(d, "vector add", vecAdd, 1<<20, 0); err != nil {
		fail(err)
	}
	if err := run(d, "matmul 256x256", matMul, 256*256, 256); err != nil {
		fail(err)
	}
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "compute:", err)
	os.Exit(1)
}

func bytesOf(f []float32) []byte {
	b := make([]byte, 4*len(f))
	for i, v := range f {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(v))
	}
	return b
}

// run computes c from a and b on the GPU and on the CPU and compares
// them; n is the element count, side the matrix side (0: vector add).
func run(d *wasigpu.Device, name, wgsl string, n, side int) error {
	a, b := make([]float32, n), make([]float32, n)
	for i := range a {
		a[i] = float32(i%97) / 7
		b[i] = float32(i%89) / 5
	}
	want := make([]float32, n)
	t0 := time.Now()
	if side == 0 {
		for i := range want {
			want[i] = a[i] + b[i]
		}
	} else {
		for r := 0; r < side; r++ {
			for k := 0; k < side; k++ {
				av := a[r*side+k]
				for c := 0; c < side; c++ {
					want[r*side+c] += av * b[k*side+c]
				}
			}
		}
	}
	cpu := time.Since(t0)
	size := uint64(4 * n)
	var bufs []*wasigpu.Buffer
	defer func() {
		for _, b := range bufs {
			b.Drop()
		}
	}()
	mk := func(usage uint32) (*wasigpu.Buffer, error) {
		b, err := d.Buffer(size, usage)
		if err == nil {
			bufs = append(bufs, b)
		}
		return b, err
	}
	ga, err := mk(wasigpu.Storage | wasigpu.CopyDst)
	if err != nil {
		return err
	}
	gb, err := mk(wasigpu.Storage | wasigpu.CopyDst)
	if err != nil {
		return err
	}
	gc, err := mk(wasigpu.Storage | wasigpu.CopySrc)
	if err != nil {
		return err
	}
	rd, err := mk(wasigpu.MapRead | wasigpu.CopyDst)
	if err != nil {
		return err
	}
	if err := d.Write(ga, 0, bytesOf(a)); err != nil {
		return err
	}
	if err := d.Write(gb, 0, bytesOf(b)); err != nil {
		return err
	}
	p, err := d.ComputePipeline(wgsl, "main")
	if err != nil {
		return err
	}
	defer p.Drop()
	t1 := time.Now()
	x, y := uint32((n+63)/64), uint32(1)
	if side > 0 {
		x, y = uint32(side/8), uint32(side/8)
	} else if x > 65535 {
		y = (x + 65534) / 65535
		x = 65535
	}
	if err := d.Dispatch(p, []*wasigpu.Buffer{ga, gb, gc}, x, y, 1); err != nil {
		return err
	}
	if err := d.Copy(gc, rd, size); err != nil {
		return err
	}
	out, err := rd.Read()
	if err != nil {
		return err
	}
	gpu := time.Since(t1)
	var maxErr float64
	for i := range want {
		g := math.Float32frombits(binary.LittleEndian.Uint32(out[i*4:]))
		e := math.Abs(float64(g - want[i]))
		maxErr = max(maxErr, e)
		if e > 1e-3*math.Max(1, math.Abs(float64(want[i]))) {
			return fmt.Errorf("%s: element %d is %g on the GPU, %g on the CPU", name, i, g, want[i])
		}
	}
	fmt.Printf("%s: %d elements match the CPU (max |error| %.2g); GPU %v (with transfers), CPU %v\n", name, n, maxErr, gpu.Round(time.Microsecond), cpu.Round(time.Microsecond))
	return nil
}
