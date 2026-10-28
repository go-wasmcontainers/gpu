package nn

import (
	"errors"

	"github.com/go-wasmcontainers/gpu/onnx"
)

// CPUBackend runs ONNX graphs with package onnx, in pure Go. It serves the
// "cpu" target only.
type CPUBackend struct {
	Limits onnx.Limits
}

// Encoding implements Backend.
func (CPUBackend) Encoding() Encoding { return ONNX }

// Runs implements Backend.
func (CPUBackend) Runs(t Target) bool { return t == CPU }

// Load implements Backend: one builder, the encoded model.
func (b CPUBackend) Load(builders [][]byte, _ Target) (Graph, Errno) {
	if len(builders) != 1 {
		return nil, InvalidArgument
	}
	s, err := onnx.Load(builders[0], b.Limits)
	if err != nil {
		var u *onnx.UnsupportedError
		if errors.As(err, &u) {
			return nil, UnsupportedOperation
		}
		return nil, InvalidEncoding
	}
	return cpuGraph{s}, Success
}

type cpuGraph struct{ s *onnx.Session }

func (g cpuGraph) NewContext() (Context, Errno) {
	return &cpuContext{s: g.s, in: map[string]*onnx.Tensor{}}, Success
}

type cpuContext struct {
	s   *onnx.Session
	in  map[string]*onnx.Tensor
	out map[string]*onnx.Tensor
}

func (c *cpuContext) SetInput(i uint32, t *onnx.Tensor) Errno {
	ins := c.s.Inputs()
	if int(i) >= len(ins) {
		return InvalidArgument
	}
	c.in[ins[i].Name] = t
	return Success
}

func (c *cpuContext) Compute() Errno {
	out, err := c.s.Run(c.in)
	if err != nil {
		return RuntimeError
	}
	c.out = out
	return Success
}

func (c *cpuContext) Output(i uint32) (*onnx.Tensor, Errno) {
	outs := c.s.Outputs()
	if c.out == nil || int(i) >= len(outs) {
		return nil, InvalidArgument
	}
	t := c.out[outs[i].Name]
	if t == nil {
		return nil, RuntimeError
	}
	return t, Success
}
