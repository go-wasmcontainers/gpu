//go:build js

package webgpu

import (
	"encoding/binary"
	"fmt"
	"math"
	"syscall/js"

	"github.com/go-wasmcontainers/gpu/internal/jsgpu"
)

// gpuDevice runs the kernels on a WebGPU device.
type gpuDevice struct {
	d *jsgpu.Device
	// limit bounds the bytes this device holds at once (a container's
	// GPU memory quota); used counts them.
	limit, used int64
}

type gbuf struct {
	v    js.Value
	size int64
}

func newGPUDevice(d *jsgpu.Device, limit int64) *gpuDevice {
	return &gpuDevice{d: d, limit: limit}
}

func f32bytes(f []float32) []byte {
	b := make([]byte, 4*len(f))
	for i, v := range f {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(v))
	}
	return b
}

func (g *gpuDevice) charge(n int64) error {
	if g.used+n > g.limit {
		return fmt.Errorf("GPU memory quota of %d bytes exceeded", g.limit)
	}
	g.used += n
	return nil
}

func (g *gpuDevice) upload(f []float32) (buf, error) {
	n := int64(max(4, 4*len(f)))
	if err := g.charge(n); err != nil {
		return nil, err
	}
	return &gbuf{v: g.d.Upload(f32bytes(f)), size: n}, nil
}

func (g *gpuDevice) alloc(n int64) (buf, error) {
	size := max(4, 4*n)
	if err := g.charge(size); err != nil {
		return nil, err
	}
	return &gbuf{v: g.d.Buffer(int(size), jsgpu.UsageStorage|jsgpu.UsageCopySrc|jsgpu.UsageCopyDst), size: size}, nil
}

func (g *gpuDevice) release(b buf) {
	gb := b.(*gbuf)
	gb.v.Call("destroy")
	g.used -= gb.size
}

func (g *gpuDevice) read(b buf, n int64) ([]float32, error) {
	raw, err := g.d.Read(b.(*gbuf).v, int(4*n))
	if err != nil {
		return nil, err
	}
	f := make([]float32, n)
	for i := range f {
		f[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return f, nil
}

func (g *gpuDevice) run(kernel string, n int64, prm []uint32, bufs ...buf) error {
	src, ok := kernelSource(kernel)
	if !ok {
		return fmt.Errorf("no kernel %q", kernel)
	}
	p, err := g.d.Pipeline(src)
	if err != nil {
		return err
	}
	pb := make([]byte, 4*max(len(prm), 1))
	for i, w := range prm {
		binary.LittleEndian.PutUint32(pb[i*4:], w)
	}
	pbuf := g.d.Upload(pb)
	defer pbuf.Call("destroy")
	vals := []js.Value{pbuf}
	for _, b := range bufs {
		vals = append(vals, b.(*gbuf).v)
	}
	if n > 0 {
		g.d.Dispatch(p, int(n), vals...)
	}
	return nil
}
