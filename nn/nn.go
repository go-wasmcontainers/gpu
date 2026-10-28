// Package nn is the host side of wasi-nn for go-wasmcontainers: the
// preview 1 (witx) interface, import module "wasi_ephemeral_nn"
// (github.com/WebAssembly/wasi-nn, wasi-nn.witx), served by the engine's
// WASI host as a host module, natively and in the browser.
//
// A graph is loaded by a Backend chosen by encoding, as Wasmtime does; the
// execution target is checked, not passed through: "gpu" runs on a GPU
// backend or fails with unsupported_operation, never quietly on the CPU,
// and needs the container's "gpu" capability (run --gpu).
//
// Everything a guest passes is bounded: model bytes, graphs and contexts
// per process, tensor dimensions and data.
package nn

import (
	"encoding/binary"
	"math"

	"github.com/go-wasmcontainers/engine/wasi"
	"github.com/go-wasmcontainers/gpu/onnx"
)

// Errno is wasi-nn's nn_errno (witx order).
type Errno = uint32

// nn_errno values.
const (
	Success              Errno = 0
	InvalidArgument      Errno = 1
	InvalidEncoding      Errno = 2
	MissingMemory        Errno = 3
	Busy                 Errno = 4
	RuntimeError         Errno = 5
	UnsupportedOperation Errno = 6
	TooLarge             Errno = 7
	NotFound             Errno = 8
)

// Encoding is graph_encoding.
type Encoding = uint32

// graph_encoding values.
const (
	OpenVINO       Encoding = 0
	ONNX           Encoding = 1
	TensorFlow     Encoding = 2
	PyTorch        Encoding = 3
	TensorFlowLite Encoding = 4
	Autodetect     Encoding = 5
)

// Target is execution_target.
type Target = uint32

// execution_target values.
const (
	CPU Target = 0
	GPU Target = 1
	TPU Target = 2
)

// tensor_type values.
const (
	TypeF16 = 0
	TypeF32 = 1
	TypeF64 = 2
	TypeU8  = 3
	TypeI32 = 4
	TypeI64 = 5
)

// Backend loads graphs of one encoding.
type Backend interface {
	// Encoding is the graph encoding it reads.
	Encoding() Encoding
	// Targets reports whether it runs on target.
	Runs(target Target) bool
	// Load builds a graph from the guest's builder buffers.
	Load(builders [][]byte, target Target) (Graph, Errno)
}

// Graph is a loaded model.
type Graph interface {
	NewContext() (Context, Errno)
}

// Context is one inference's inputs and outputs.
type Context interface {
	SetInput(index uint32, t *onnx.Tensor) Errno
	Compute() Errno
	Output(index uint32) (*onnx.Tensor, Errno)
}

// Limits bound what one process may make the host hold.
type Limits struct {
	ModelBytes int // all builders of one load, together
	Graphs     int // graphs alive per process
	Contexts   int // contexts alive per process
	Dims       int // tensor rank
	TensorData int // bytes of one input tensor
}

// DefaultLimits suit small models.
var DefaultLimits = Limits{ModelBytes: 64 << 20, Graphs: 8, Contexts: 32, Dims: 8, TensorData: 64 << 20}

// Module is the wasi_ephemeral_nn host module.
type Module struct {
	Backends []Backend
	Limits   Limits
	// Named are the graphs load_by_name knows: name to its builders and
	// encoding. load_by_name of any other name is not_found.
	Named map[string]Named
}

// Named is a graph registered for load_by_name.
type Named struct {
	Encoding Encoding
	Builders [][]byte
}

// NewModule returns the module with the pure-Go CPU backend.
func NewModule() *Module {
	return &Module{Backends: []Backend{CPUBackend{Limits: onnx.DefaultLimits}}, Limits: DefaultLimits}
}

// Module implements wasi.HostModule.
func (*Module) Module() string { return "wasi_ephemeral_nn" }

// Funcs implements wasi.HostModule (wasi-nn.witx, as the wasi-nn crate's
// generated bindings lower it: every argument an i32, nn_errno returned).
func (*Module) Funcs() []wasi.HostFunc {
	return []wasi.HostFunc{
		{Name: "load", Params: "iiiii", Result: "i"},
		{Name: "load_by_name", Params: "iii", Result: "i"},
		{Name: "init_execution_context", Params: "ii", Result: "i"},
		{Name: "set_input", Params: "iii", Result: "i"},
		{Name: "get_output", Params: "iiiii", Result: "i"},
		{Name: "compute", Params: "i", Result: "i"},
	}
}

// Open implements wasi.HostModule.
func (m *Module) Open(p *wasi.Proc) wasi.Instance {
	return &instance{m: m, gpu: p.HasCap("gpu"), graphs: map[uint32]Graph{}, ctxs: map[uint32]Context{}}
}

type instance struct {
	m      *Module
	gpu    bool
	graphs map[uint32]Graph
	ctxs   map[uint32]Context
	next   uint32
}

func (in *instance) Close() {
	in.graphs, in.ctxs = nil, nil
}

func r(e Errno) wasi.Result { return wasi.Result{Ret: uint64(e)} }

func (in *instance) Call(fn int, a []uint64, m wasi.Memory) wasi.Result {
	if in.graphs == nil {
		return r(RuntimeError)
	}
	u := func(i int) uint32 { return uint32(a[i]) }
	switch fn {
	case 0:
		return r(in.load(u(0), u(1), u(2), u(3), u(4), m))
	case 1:
		return r(in.loadByName(u(0), u(1), u(2), m))
	case 2:
		return r(in.initContext(u(0), u(1), m))
	case 3:
		return r(in.setInput(u(0), u(1), u(2), m))
	case 4:
		return r(in.getOutput(u(0), u(1), u(2), u(3), u(4), m))
	case 5:
		c := in.ctxs[u(0)]
		if c == nil {
			return r(InvalidArgument)
		}
		return r(c.Compute())
	}
	return wasi.Result{Trap: "wasi_ephemeral_nn: no such function"}
}

func (in *instance) handle() uint32 {
	in.next++
	return in.next
}

func put32(m wasi.Memory, off, v uint32) bool {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], v)
	return m.Write(off, b[:])
}

// load reads a graph_builder_array: n (ptr, len) pairs at ptr.
func (in *instance) load(ptr, n, enc, target, out uint32, m wasi.Memory) Errno {
	if n == 0 || n > 16 {
		return InvalidArgument
	}
	pairs, ok := m.Read(ptr, n*8)
	if !ok {
		return InvalidArgument
	}
	pairs = append([]byte(nil), pairs...)
	builders := make([][]byte, n)
	total := 0
	for i := range builders {
		p, l := binary.LittleEndian.Uint32(pairs[i*8:]), binary.LittleEndian.Uint32(pairs[i*8+4:])
		if int64(total)+int64(l) > int64(in.m.Limits.ModelBytes) {
			return TooLarge
		}
		b, ok := m.Read(p, l)
		if !ok {
			return InvalidArgument
		}
		builders[i] = append([]byte(nil), b...)
		total += int(l)
	}
	return in.loadBuilders(builders, enc, target, out, m)
}

func (in *instance) loadBuilders(builders [][]byte, enc, target, out uint32, m wasi.Memory) Errno {
	if target > TPU {
		return InvalidArgument
	}
	if target == GPU && !in.gpu {
		return UnsupportedOperation // the container was not started with --gpu
	}
	if len(in.graphs) >= in.m.Limits.Graphs {
		return Busy
	}
	var be Backend
	for _, b := range in.m.Backends {
		if b.Encoding() == enc {
			be = b
			break
		}
	}
	if be == nil {
		return InvalidEncoding
	}
	if !be.Runs(target) {
		return UnsupportedOperation
	}
	g, e := be.Load(builders, target)
	if e != Success {
		return e
	}
	h := in.handle()
	if !put32(m, out, h) {
		return InvalidArgument
	}
	in.graphs[h] = g
	return Success
}

func (in *instance) loadByName(ptr, n, out uint32, m wasi.Memory) Errno {
	if n > 1024 {
		return InvalidArgument
	}
	b, ok := m.Read(ptr, n)
	if !ok {
		return InvalidArgument
	}
	named, ok := in.m.Named[string(b)]
	if !ok {
		return NotFound
	}
	return in.loadBuilders(named.Builders, named.Encoding, CPU, out, m)
}

func (in *instance) initContext(graph, out uint32, m wasi.Memory) Errno {
	g := in.graphs[graph]
	if g == nil {
		return InvalidArgument
	}
	if len(in.ctxs) >= in.m.Limits.Contexts {
		return Busy
	}
	c, e := g.NewContext()
	if e != Success {
		return e
	}
	h := in.handle()
	if !put32(m, out, h) {
		return InvalidArgument
	}
	in.ctxs[h] = c
	return Success
}

// setInput reads a tensor record at ptr: dimensions (ptr, len of u32),
// type (u8 at 8), data (ptr, len of u8 at 12); 20 bytes, aligned to 4.
func (in *instance) setInput(ctx, index, ptr uint32, m wasi.Memory) Errno {
	c := in.ctxs[ctx]
	if c == nil {
		return InvalidArgument
	}
	rec, ok := m.Read(ptr, 20)
	if !ok {
		return InvalidArgument
	}
	rec = append([]byte(nil), rec...)
	dimsPtr, dimsLen := binary.LittleEndian.Uint32(rec[0:]), binary.LittleEndian.Uint32(rec[4:])
	typ := rec[8]
	dataPtr, dataLen := binary.LittleEndian.Uint32(rec[12:]), binary.LittleEndian.Uint32(rec[16:])
	if dimsLen > uint32(in.m.Limits.Dims) || dataLen > uint32(in.m.Limits.TensorData) {
		return TooLarge
	}
	db, ok := m.Read(dimsPtr, dimsLen*4)
	if !ok {
		return InvalidArgument
	}
	dims := make([]int64, dimsLen)
	for i := range dims {
		dims[i] = int64(binary.LittleEndian.Uint32(db[i*4:]))
	}
	data, ok := m.Read(dataPtr, dataLen)
	if !ok {
		return InvalidArgument
	}
	var t *onnx.Tensor
	switch typ {
	case TypeF32:
		if dataLen%4 != 0 {
			return InvalidArgument
		}
		f := make([]float32, dataLen/4)
		for i := range f {
			f[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
		}
		var err error
		if t, err = onnx.NewFloat(dims, f); err != nil {
			return InvalidArgument
		}
	case TypeI64:
		if dataLen%8 != 0 {
			return InvalidArgument
		}
		v := make([]int64, dataLen/8)
		for i := range v {
			v[i] = int64(binary.LittleEndian.Uint64(data[i*8:]))
		}
		t = &onnx.Tensor{Type: onnx.Int64, Dims: dims, I: v}
		if n, ok := count(dims, int64(len(v))); !ok || n != int64(len(v)) {
			return InvalidArgument
		}
	case TypeF16, TypeF64, TypeU8, TypeI32:
		return UnsupportedOperation
	default:
		return InvalidArgument
	}
	return c.SetInput(index, t)
}

// count is the product of dims when it does not exceed max.
func count(dims []int64, max int64) (int64, bool) {
	n := int64(1)
	for _, d := range dims {
		if d < 0 || d > 0 && n > max/d {
			return 0, false
		}
		n *= d
	}
	return n, true
}

// getOutput writes output index's data (little-endian elements) into the
// guest's buffer and its size at written; too_large, writing nothing,
// when the buffer is smaller.
func (in *instance) getOutput(ctx, index, buf, max, written uint32, m wasi.Memory) Errno {
	c := in.ctxs[ctx]
	if c == nil {
		return InvalidArgument
	}
	t, e := c.Output(index)
	if e != Success {
		return e
	}
	var b []byte
	switch t.Type {
	case onnx.Float:
		b = make([]byte, 4*len(t.F))
		for i, v := range t.F {
			binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(v))
		}
	default:
		b = make([]byte, 8*len(t.I))
		for i, v := range t.I {
			binary.LittleEndian.PutUint64(b[i*8:], uint64(v))
		}
	}
	if uint64(len(b)) > uint64(max) {
		return TooLarge
	}
	if !m.Write(buf, b) || !put32(m, written, uint32(len(b))) {
		return InvalidArgument
	}
	return Success
}
