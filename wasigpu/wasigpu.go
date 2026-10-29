// Package wasigpu is the guest side of wasi:webgpu (wasi:webgpu@0.3.0-rc.2)
// for Go programs built with GOOS=wasip1: the compute subset (adapter,
// device, buffers, WGSL modules, compute pipelines, bind groups, command
// encoders, queue, mapping). Every call lowers its arguments and lifts its
// result with package wit, the same Canonical ABI code the host uses;
// imports_wasip1.go is generated from the WIT by internal/cmd/witgen.
// Outside wasip1 every call fails with ErrUnavailable.
//
// Resources are released with Drop (the WIT's resource-drop). Objects not
// dropped are released when the process ends.
package wasigpu

import (
	"errors"
	"fmt"

	"github.com/go-wasmcontainers/gpu/cabi"
	witdata "github.com/go-wasmcontainers/gpu/webgpu/wit"
	"github.com/go-wasmcontainers/gpu/wit"
)

// The interfaces wasigpu imports, by ImportModule.
const (
	webgpuModule  = "wasi:webgpu/webgpu@0.3.0-rc.2"
	surfaceModule = "wasi-gfx:surface/surface@0.2.0"
	contextModule = "wasi-gfx:surface/surface-webgpu@0.2.0"
)

var webgpuIface, surfaceIface *wit.Interface

// enum is a WIT enum label's index (e.g. enum("gpu-load-op", "clear")).
func enum(typ, label string) uint32 {
	for i, l := range webgpuIface.Types[typ].Labels {
		if l == label {
			return uint32(i)
		}
	}
	panic("wasigpu: no " + typ + " " + label)
}

// funcs maps "module name" (a calls key) to its WIT function.
var funcs = func() map[string]*wit.Func {
	w, err := wit.Parse(witdata.WebGPU)
	if err != nil {
		panic(err)
	}
	webgpuIface = w["webgpu"]
	s, err := wit.ParseFiles([]*wit.Interface{w["webgpu"]}, witdata.Surface)
	if err != nil {
		panic(err)
	}
	surfaceIface = s["surface"]
	out := map[string]*wit.Func{}
	for _, i := range []*wit.Interface{w["webgpu"], s["surface"], s["surface-webgpu"]} {
		for _, f := range i.Funcs {
			out[i.ImportModule()+" "+f.ImportName("")] = f
		}
		for _, r := range i.ResourceNames() {
			for _, f := range i.Resources[r].Funcs {
				out[i.ImportModule()+" "+f.ImportName(r)] = f
			}
		}
	}
	return out
}()

// ErrUnavailable is returned outside GOOS=wasip1.
var ErrUnavailable = errors.New("wasi:webgpu: not available on this platform")

// ErrNoAdapter is returned when the host offers no GPU (none there, or the
// container was not started with --gpu).
var ErrNoAdapter = errors.New("wasi:webgpu: no adapter")

// call runs a wasi:webgpu import.
func call(name string, args ...any) (any, error) { return callIn(webgpuModule, name, args...) }

// callIn runs an import of a module with high-level arguments and returns
// its lifted result.
func callIn(module, name string, args ...any) (any, error) {
	c, ok := calls[module+" "+name]
	if !ok {
		return nil, ErrUnavailable
	}
	f := funcs[module+" "+name]
	if f == nil {
		return nil, fmt.Errorf("wasi:webgpu: no function %s", name)
	}
	defer cabi.Release()
	var mem cabi.Memory
	var alloc cabi.Allocator
	s := wit.Sig(f)
	var flat []uint64
	if s.ParamsInMemory {
		tt := wit.ParamTuple(f)
		p := cabi.Alloc(wit.Align(tt), wit.Size(tt))
		if err := wit.Store(mem, alloc, p, tt, args); err != nil {
			return nil, err
		}
		flat = []uint64{uint64(p)}
	} else {
		for i, p := range f.Params {
			v, err := wit.LowerFlat(mem, alloc, p.Type, args[i])
			if err != nil {
				return nil, fmt.Errorf("%s: %s: %w", name, p.Name, err)
			}
			flat = append(flat, v...)
		}
	}
	var ret uint32
	if s.RetPtr {
		ret = cabi.Alloc(wit.Align(f.Result), wit.Size(f.Result))
		flat = append(flat, uint64(ret))
	}
	r := c(flat)
	switch {
	case f.Result == nil:
		return nil, nil
	case s.RetPtr:
		return wit.Load(mem, ret, f.Result)
	}
	return wit.LiftFlat(f.Result, []uint64{r}, mem)
}

func drop(res string, h uint32) { dropIn(webgpuModule, res, h) }

func dropIn(module, res string, h uint32) {
	if c, ok := calls[module+" [resource-drop]"+res]; ok {
		c([]uint64{uint64(h)})
	}
}

// errOf turns a WIT error record {kind, message} into an error.
func errOf(op string, v any) error {
	if fs, ok := v.([]any); ok && len(fs) == 2 {
		return fmt.Errorf("wasi:webgpu: %s: %v", op, fs[1])
	}
	return fmt.Errorf("wasi:webgpu: %s failed", op)
}

func handle(v any, err error) (uint32, error) {
	if err != nil {
		return 0, err
	}
	h, ok := v.(uint32)
	if !ok {
		return 0, fmt.Errorf("wasi:webgpu: expected a handle, got %T", v)
	}
	return h, nil
}

// unwrap(op)(call(...)) returns a result's ok value or its error.
func unwrap(op string) func(any, error) (any, error) {
	return func(v any, err error) (any, error) {
		if err != nil {
			return nil, err
		}
		r, ok := v.(wit.ResultValue)
		if !ok {
			return nil, fmt.Errorf("wasi:webgpu: %s: expected a result, got %T", op, v)
		}
		if r.Err {
			return nil, errOf(op, r.Value)
		}
		return r.Value, nil
	}
}

// Buffer usages (gpu-buffer-usage flags, in WIT order).
const (
	MapRead  = 1 << 0
	MapWrite = 1 << 1
	CopySrc  = 1 << 2
	CopyDst  = 1 << 3
	Index    = 1 << 4
	Vertex   = 1 << 5
	Uniform  = 1 << 6
	Storage  = 1 << 7
)

// Device is a GPU device and its queue.
type Device struct {
	adapter, dev, queue uint32
	format              uint32 // the preferred canvas format (gpu-texture-format)
}

// Open asks for an adapter and a device: ErrNoAdapter when the host has
// none for this process.
func Open() (*Device, error) {
	gpu, err := handle(call("get-gpu"))
	if err != nil {
		return nil, err
	}
	defer drop("gpu", gpu)
	v, err := call("[method]gpu.request-adapter", gpu, nil)
	if err != nil {
		return nil, err
	}
	some, ok := v.(wit.Some)
	if !ok {
		return nil, ErrNoAdapter
	}
	d := &Device{adapter: some.Value.(uint32)}
	if f, err := call("[method]gpu.get-preferred-canvas-format", gpu); err == nil {
		d.format, _ = f.(uint32)
	}
	v, err = unwrap("request-device")(call("[method]gpu-adapter.request-device", d.adapter, nil))
	if err != nil {
		drop("gpu-adapter", d.adapter)
		return nil, err
	}
	d.dev = v.(uint32)
	if d.queue, err = handle(call("[method]gpu-device.queue", d.dev)); err != nil {
		d.Drop()
		return nil, err
	}
	return d, nil
}

// Drop releases the device.
func (d *Device) Drop() {
	if d.queue != 0 {
		drop("gpu-queue", d.queue)
	}
	drop("gpu-device", d.dev)
	drop("gpu-adapter", d.adapter)
}

// Buffer is a GPU buffer.
type Buffer struct {
	h    uint32
	Size uint64
}

// Buffer creates a buffer of size bytes with usage flags.
func (d *Device) Buffer(size uint64, usage uint32) (*Buffer, error) {
	desc := []any{size, uint64(usage), nil, nil}
	h, err := handle(call("[method]gpu-device.create-buffer", d.dev, desc))
	if err != nil {
		return nil, err
	}
	return &Buffer{h: h, Size: size}, nil
}

// Drop releases the buffer.
func (b *Buffer) Drop() { drop("gpu-buffer", b.h) }

// Write copies data into b at offset (queue.writeBuffer).
func (d *Device) Write(b *Buffer, offset uint64, data []byte) error {
	_, err := unwrap("write-buffer")(call("[method]gpu-queue.write-buffer-with-copy", d.queue, b.h, offset, data, nil, nil))
	return err
}

// Read maps a MAP_READ buffer and copies its bytes out.
func (b *Buffer) Read() ([]byte, error) {
	if _, err := unwrap("map-async")(call("[method]gpu-buffer.map-async", b.h, uint64(1), nil, nil)); err != nil {
		return nil, err
	}
	v, err := unwrap("get-mapped-range")(call("[method]gpu-buffer.get-mapped-range-get-with-copy", b.h, nil, nil))
	if err != nil {
		return nil, err
	}
	if _, err := unwrap("unmap")(call("[method]gpu-buffer.unmap", b.h)); err != nil {
		return nil, err
	}
	data, ok := v.([]byte)
	if !ok {
		return nil, fmt.Errorf("wasi:webgpu: mapped range is %T", v)
	}
	return data, nil
}

// Pipeline is a compute pipeline and its first bind group's layout.
type Pipeline struct {
	module, pipe, layout uint32
}

// ComputePipeline compiles WGSL and makes a compute pipeline of its entry
// point (layout "auto").
func (d *Device) ComputePipeline(wgsl, entry string) (*Pipeline, error) {
	mod, err := handle(call("[method]gpu-device.create-shader-module", d.dev, []any{wgsl, nil, nil}))
	if err != nil {
		return nil, err
	}
	stage := []any{mod, wit.Some{Value: entry}, nil}
	layout := wit.VariantValue{Case: 1} // auto
	pipe, err := handle(call("[method]gpu-device.create-compute-pipeline", d.dev, []any{stage, layout, nil}))
	if err != nil {
		drop("gpu-shader-module", mod)
		return nil, err
	}
	bgl, err := handle(call("[method]gpu-compute-pipeline.get-bind-group-layout", pipe, uint64(0)))
	if err != nil {
		drop("gpu-compute-pipeline", pipe)
		drop("gpu-shader-module", mod)
		return nil, err
	}
	return &Pipeline{module: mod, pipe: pipe, layout: bgl}, nil
}

// Drop releases the pipeline.
func (p *Pipeline) Drop() {
	drop("gpu-bind-group-layout", p.layout)
	drop("gpu-compute-pipeline", p.pipe)
	drop("gpu-shader-module", p.module)
}

// Dispatch runs p over x*y*z workgroups with buffers bound at group 0,
// bindings 0.. in order, and waits until the queue has submitted it.
func (d *Device) Dispatch(p *Pipeline, buffers []*Buffer, x, y, z uint32) error {
	var entries []any
	for i, b := range buffers {
		entries = append(entries, []any{uint64(i), wit.VariantValue{Case: 0, Value: b.h}})
	}
	bg, err := handle(call("[method]gpu-device.create-bind-group", d.dev, []any{p.layout, entries, nil}))
	if err != nil {
		return err
	}
	defer drop("gpu-bind-group", bg)
	enc, err := handle(call("[method]gpu-device.create-command-encoder", d.dev, nil))
	if err != nil {
		return err
	}
	defer drop("gpu-command-encoder", enc)
	pass, err := handle(call("[method]gpu-command-encoder.begin-compute-pass", enc, nil))
	if err != nil {
		return err
	}
	defer drop("gpu-compute-pass-encoder", pass)
	if _, err := call("[method]gpu-compute-pass-encoder.set-pipeline", pass, p.pipe); err != nil {
		return err
	}
	if _, err := unwrap("set-bind-group")(call("[method]gpu-compute-pass-encoder.set-bind-group", pass, uint64(0), wit.Some{Value: bg}, nil, nil, nil)); err != nil {
		return err
	}
	if _, err := call("[method]gpu-compute-pass-encoder.dispatch-workgroups", pass, uint64(x), wit.Some{Value: uint64(y)}, wit.Some{Value: uint64(z)}); err != nil {
		return err
	}
	if _, err := call("[method]gpu-compute-pass-encoder.end", pass); err != nil {
		return err
	}
	return d.submit(enc)
}

// Copy copies size bytes from src to dst on the GPU.
func (d *Device) Copy(src, dst *Buffer, size uint64) error {
	enc, err := handle(call("[method]gpu-device.create-command-encoder", d.dev, nil))
	if err != nil {
		return err
	}
	defer drop("gpu-command-encoder", enc)
	if _, err := call("[method]gpu-command-encoder.copy-buffer-to-buffer", enc, src.h, nil, dst.h, nil, wit.Some{Value: size}); err != nil {
		return err
	}
	return d.submit(enc)
}

func (d *Device) submit(enc uint32) error {
	cb, err := handle(call("[method]gpu-command-encoder.finish", enc, nil))
	if err != nil {
		return err
	}
	defer drop("gpu-command-buffer", cb)
	_, err = call("[method]gpu-queue.submit", d.queue, []any{cb})
	return err
}
