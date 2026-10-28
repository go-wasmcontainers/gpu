// Package wasinn is the guest side of wasi-nn (preview 1, import module
// "wasi_ephemeral_nn") for Go programs built with GOOS=wasip1: load a
// model, set inputs, compute, read outputs. Outside wasip1 every call
// fails with ErrUnavailable, so code using it still builds and tests
// natively.
package wasinn

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"unsafe"
)

// Encoding is a graph encoding.
type Encoding uint32

// Graph encodings (wasi-nn.witx).
const (
	OpenVINO Encoding = iota
	ONNX
	TensorFlow
	PyTorch
	TensorFlowLite
	Autodetect
)

// Target is where a graph runs.
type Target uint32

// Execution targets.
const (
	CPU Target = iota
	GPU
	TPU
)

// Error is an nn_errno.
type Error uint32

var names = [...]string{"success", "invalid argument", "invalid encoding", "missing memory", "busy",
	"runtime error", "unsupported operation", "too large", "not found"}

func (e Error) Error() string {
	if int(e) < len(names) {
		return "wasi-nn: " + names[e]
	}
	return fmt.Sprintf("wasi-nn: error %d", uint32(e))
}

// nn_errno values a caller may test with errors.Is.
const (
	ErrInvalidArgument      Error = 1
	ErrInvalidEncoding      Error = 2
	ErrBusy                 Error = 4
	ErrRuntime              Error = 5
	ErrUnsupportedOperation Error = 6
	ErrTooLarge             Error = 7
	ErrNotFound             Error = 8
)

// ErrUnavailable is returned outside GOOS=wasip1.
var ErrUnavailable = errors.New("wasi-nn: not available on this platform")

func check(e uint32) error {
	if e == 0 {
		return nil
	}
	return Error(e)
}

// Graph is a loaded model.
type Graph uint32

// Load loads a model from its builder buffers (one for ONNX).
func Load(builders [][]byte, enc Encoding, target Target) (Graph, error) {
	pairs := make([]uint32, 2*len(builders))
	for i, b := range builders {
		if len(b) > 0 {
			pairs[2*i] = uint32(uintptr(unsafe.Pointer(&b[0])))
		}
		pairs[2*i+1] = uint32(len(b))
	}
	var g uint32
	var p unsafe.Pointer
	if len(pairs) > 0 {
		p = unsafe.Pointer(&pairs[0])
	}
	err := callLoad(p, uint32(len(builders)), uint32(enc), uint32(target), unsafe.Pointer(&g))
	keep(builders)
	return Graph(g), err
}

// LoadByName loads a model the host registered under name.
func LoadByName(name string) (Graph, error) {
	var g uint32
	b := []byte(name)
	var p unsafe.Pointer
	if len(b) > 0 {
		p = unsafe.Pointer(&b[0])
	}
	err := callLoadByName(p, uint32(len(b)), unsafe.Pointer(&g))
	return Graph(g), err
}

// Context is one inference.
type Context uint32

// NewContext starts an inference on g.
func (g Graph) NewContext() (Context, error) {
	var c uint32
	err := callInit(uint32(g), unsafe.Pointer(&c))
	return Context(c), err
}

// tensor is the witx record: dimensions (ptr, len), type (u8), data
// (ptr, len); 20 bytes.
type tensor struct {
	dimsPtr, dimsLen uint32
	typ              uint32 // u8 in the low byte, padding above
	dataPtr, dataLen uint32
}

// SetInputF32 sets input index to a float32 tensor of shape dims.
func (c Context) SetInputF32(index uint32, dims []uint32, data []float32) error {
	b := make([]byte, 4*len(data))
	for i, v := range data {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(v))
	}
	t := tensor{dimsLen: uint32(len(dims)), typ: 1, dataLen: uint32(len(b))}
	if len(dims) > 0 {
		t.dimsPtr = uint32(uintptr(unsafe.Pointer(&dims[0])))
	}
	if len(b) > 0 {
		t.dataPtr = uint32(uintptr(unsafe.Pointer(&b[0])))
	}
	err := callSetInput(uint32(c), index, unsafe.Pointer(&t))
	keep(dims, b)
	return err
}

// Compute runs the inference.
func (c Context) Compute() error { return callCompute(uint32(c)) }

// OutputF32 reads output index as float32 values into out and returns
// how many it holds; ErrTooLarge when out is too small.
func (c Context) OutputF32(index uint32, out []float32) (int, error) {
	b := make([]byte, 4*len(out))
	var n uint32
	var p unsafe.Pointer
	if len(b) > 0 {
		p = unsafe.Pointer(&b[0])
	}
	if err := callGetOutput(uint32(c), index, p, uint32(len(b)), unsafe.Pointer(&n)); err != nil {
		return 0, err
	}
	k := int(n / 4)
	for i := 0; i < k; i++ {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return k, nil
}

// keep holds values alive across a host call that reads them through
// pointers the compiler cannot see.
//
//go:noinline
func keep(...any) {}
