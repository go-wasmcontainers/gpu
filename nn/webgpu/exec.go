package webgpu

import (
	"errors"
	"fmt"
	"slices"

	"github.com/go-wasmcontainers/gpu/onnx"
)

// buf is a device buffer of float32.
type buf any

// device runs the kernels. The browser one is WebGPU (device_js.go); the
// tests use an emulator that runs each kernel's index arithmetic in Go,
// so the meta layouts are checked natively against package onnx.
type device interface {
	upload(f []float32) (buf, error)
	alloc(n int64) (buf, error)
	// run dispatches kernel over n invocations with prm at binding 0 and
	// bufs at 1.. in order.
	run(kernel string, n int64, prm []uint32, bufs ...buf) error
	read(b buf, n int64) ([]float32, error)
	release(b buf)
}

// Kernel names, mapped to WGSL sources by the browser device.
const (
	kRelu    = "relu"
	kMatmul  = "matmul"
	kConv    = "conv"
	kPool    = "pool"
	kSoftmax = "softmax"
	kCopy    = "copy"
)

func kBinary(op string) string { return "binary:" + op }

// kernelSource is the WGSL of a kernel name.
func kernelSource(name string) (string, bool) {
	switch name {
	case kRelu:
		return reluKernel, true
	case kMatmul:
		return matmulKernel, true
	case kConv:
		return convKernel, true
	case kPool:
		return poolKernel, true
	case kSoftmax:
		return softmaxKernel, true
	case kCopy:
		return copyKernel, true
	}
	if op, ok := binaryOps[name[len("binary:"):]]; ok && len(name) > 7 && name[:7] == "binary:" {
		return binaryKernel(op), true
	}
	return "", false
}

// gtensor is a value on the device.
type gtensor struct {
	b    buf
	dims []int64
}

// executor runs one plan on a device.
type executor struct {
	dev  device
	plan *onnx.Plan
	// owned buffers, released after the run (constants stay).
	scratch []buf
	consts  map[string]gtensor
}

var errUnsupported = errors.New("not supported on the GPU")

// newExecutor uploads a plan's float constants.
func newExecutor(dev device, p *onnx.Plan) (*executor, error) {
	e := &executor{dev: dev, plan: p, consts: map[string]gtensor{}}
	for name, t := range p.Consts {
		if t.Type != onnx.Float {
			continue // int64 constants (shapes) are consumed by the plan
		}
		b, err := dev.upload(t.F)
		if err != nil {
			return nil, err
		}
		e.consts[name] = gtensor{b: b, dims: t.Dims}
	}
	return e, nil
}

func (e *executor) alloc(dims []int64) (buf, error) {
	b, err := e.dev.alloc(prod(dims))
	if err == nil {
		e.scratch = append(e.scratch, b)
	}
	return b, err
}

// zero is a one-element placeholder for an absent optional input.
func (e *executor) zero() (buf, error) { return e.alloc([]int64{1}) }

// run computes the graph outputs (by name) from float inputs.
func (e *executor) run(inputs map[string]*onnx.Tensor, outputs []string) (map[string][]float32, map[string][]int64, error) {
	defer func() {
		for _, b := range e.scratch {
			e.dev.release(b)
		}
		e.scratch = nil
	}()
	vals := map[string]gtensor{}
	for k, v := range e.consts {
		vals[k] = v
	}
	for name, t := range inputs {
		if t.Type != onnx.Float {
			return nil, nil, fmt.Errorf("input %q: %w (%v)", name, errUnsupported, t.Type)
		}
		b, err := e.dev.upload(t.F)
		if err != nil {
			return nil, nil, err
		}
		e.scratch = append(e.scratch, b)
		vals[name] = gtensor{b: b, dims: t.Dims}
	}
	for i, st := range e.plan.Steps {
		if st.Const {
			continue
		}
		if err := e.step(st, vals); err != nil {
			return nil, nil, fmt.Errorf("node %d (%s %q): %w", i, st.Node.Op, st.Node.Name, err)
		}
	}
	out := map[string][]float32{}
	shapes := map[string][]int64{}
	for _, name := range outputs {
		v, ok := vals[name]
		if !ok {
			return nil, nil, fmt.Errorf("output %q: %w (not a float value)", name, errUnsupported)
		}
		f, err := e.dev.read(v.b, prod(v.dims))
		if err != nil {
			return nil, nil, err
		}
		out[name], shapes[name] = f, v.dims
	}
	return out, shapes, nil
}

func (e *executor) step(st onnx.Step, vals map[string]gtensor) error {
	n := st.Node
	in := func(i int) (gtensor, error) {
		if i >= len(n.Inputs) || n.Inputs[i] == "" {
			return gtensor{}, fmt.Errorf("input %d missing", i)
		}
		v, ok := vals[n.Inputs[i]]
		if !ok {
			return gtensor{}, fmt.Errorf("input %q: %w (not a float value)", n.Inputs[i], errUnsupported)
		}
		return v, nil
	}
	outDims := st.OutShapes[0]
	set := func(b buf) { vals[n.Outputs[0]] = gtensor{b: b, dims: outDims} }
	switch n.Op {
	case "Identity", "Dropout", "Reshape", "Flatten":
		x, err := in(0)
		if err != nil {
			return err
		}
		set(x.b) // same values, new shape
		return nil
	}
	y, err := e.alloc(outDims)
	if err != nil {
		return err
	}
	switch n.Op {
	case "Relu":
		x, err := in(0)
		if err != nil {
			return err
		}
		if err := e.dev.run(kRelu, prod(outDims), []uint32{uint32(prod(outDims))}, x.b, y); err != nil {
			return err
		}
	case "Add", "Sub", "Mul", "Div":
		a, err := in(0)
		if err != nil {
			return err
		}
		b, err := in(1)
		if err != nil {
			return err
		}
		prm, err := binaryMeta(outDims, a.dims, b.dims)
		if err != nil {
			return err
		}
		if err := e.dev.run(kBinary(n.Op), prod(outDims), prm, a.b, b.b, y); err != nil {
			return err
		}
	case "MatMul", "Gemm":
		a, err := in(0)
		if err != nil {
			return err
		}
		b, err := in(1)
		if err != nil {
			return err
		}
		var cb buf
		var cdims []int64
		hasC := n.Op == "Gemm" && len(n.Inputs) > 2 && n.Inputs[2] != ""
		if hasC {
			c, err := in(2)
			if err != nil {
				return err
			}
			cb, cdims = c.b, c.dims
		} else if cb, err = e.zero(); err != nil {
			return err
		}
		prm, err := matmulMeta(n, a.dims, b.dims, cdims, outDims, hasC)
		if err != nil {
			return err
		}
		if err := e.dev.run(kMatmul, prod(outDims), prm, a.b, b.b, cb, y); err != nil {
			return err
		}
	case "Conv":
		x, err := in(0)
		if err != nil {
			return err
		}
		w, err := in(1)
		if err != nil {
			return err
		}
		bias := gtensor{}
		hasBias := len(n.Inputs) > 2 && n.Inputs[2] != ""
		if hasBias {
			if bias, err = in(2); err != nil {
				return err
			}
		} else if bias.b, err = e.zero(); err != nil {
			return err
		}
		prm, err := convMeta(n, x.dims, w.dims, outDims, hasBias)
		if err != nil {
			return err
		}
		if err := e.dev.run(kConv, prod(outDims), prm, x.b, w.b, bias.b, y); err != nil {
			return err
		}
	case "MaxPool", "AveragePool":
		x, err := in(0)
		if err != nil {
			return err
		}
		prm, err := poolMeta(n, x.dims, outDims)
		if err != nil {
			return err
		}
		if err := e.dev.run(kPool, prod(outDims), prm, x.b, y); err != nil {
			return err
		}
	case "Softmax":
		x, err := in(0)
		if err != nil {
			return err
		}
		prm, rows, err := softmaxMeta(n, e.plan.Opset, x.dims)
		if err != nil {
			return err
		}
		if err := e.dev.run(kSoftmax, rows, prm, x.b, y); err != nil {
			return err
		}
	case "Transpose":
		x, err := in(0)
		if err != nil {
			return err
		}
		prm, err := transposeMeta(n, x.dims, outDims)
		if err != nil {
			return err
		}
		if err := e.dev.run(kCopy, prod(outDims), prm, x.b, y); err != nil {
			return err
		}
	case "Concat":
		var shapes [][]int64
		var parts []gtensor
		for i := range n.Inputs {
			v, err := in(i)
			if err != nil {
				return err
			}
			parts = append(parts, v)
			shapes = append(shapes, v.dims)
		}
		prms, err := concatParts(n, shapes, outDims)
		if err != nil {
			return err
		}
		for i, prm := range prms {
			if err := e.dev.run(kCopy, prod(shapes[i]), prm, parts[i].b, y); err != nil {
				return err
			}
		}
	default:
		return fmt.Errorf("%s: %w", n.Op, errUnsupported)
	}
	set(y)
	return nil
}

// Supported lists the operators with a GPU kernel (or a copy-free
// rewrite); it equals onnx.Supported, checked by a test.
var Supported = []string{"Add", "AveragePool", "Concat", "Constant", "Conv", "Div", "Dropout", "Flatten",
	"Gemm", "Identity", "MatMul", "MaxPool", "Mul", "Relu", "Reshape", "Softmax", "Sub", "Transpose"}

func supportedOp(op string) bool { return slices.Contains(Supported, op) }
