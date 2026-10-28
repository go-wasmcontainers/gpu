package onnx

import (
	"fmt"
	"maps"
)

// A Plan is what another executor (the WebGPU backend) needs to run a
// session's graph for given input shapes: each node's input and output
// shapes, and the values that do not depend on the inputs (initializers,
// Constant outputs and everything computed only from them, such as a
// Reshape's shape), already computed.
//
// Plan derives the shapes by running the graph once on the CPU with zero
// inputs, so the other executor and the CPU agree on every shape by
// construction rather than by a second implementation of the shape rules.

// Step is one node of a plan.
type Step struct {
	Node      *Node
	InShapes  [][]int64 // nil for an omitted optional input
	OutShapes [][]int64
	// Const says the outputs do not depend on the graph inputs; they are
	// in Plan.Consts.
	Const bool
}

// Plan is a graph laid out for given input shapes.
type Plan struct {
	Opset  int64
	Steps  []Step
	Shapes map[string][]int64
	Consts map[string]*Tensor
}

// Plan lays the graph out for the given inputs (by name). Only the shapes
// of float inputs matter (the dry run uses zeros); int64 inputs (shapes,
// axes) are used by value and count as constants, so a plan holds for
// inputs of the same float shapes and the same int64 values.
func (s *Session) Plan(inputs map[string]*Tensor) (*Plan, error) {
	g := s.model.Graph
	vals := map[string]*Tensor{}
	maps.Copy(vals, g.Initializers)
	isConst := map[string]bool{"": true}
	for name := range g.Initializers {
		isConst[name] = true
	}
	for _, in := range s.inputs {
		given, ok := inputs[in.Name]
		if !ok || given == nil {
			return nil, fmt.Errorf("onnx: plan: input %q missing", in.Name)
		}
		if err := checkInput(in, given, s.lim); err != nil {
			return nil, err
		}
		t := given
		if given.Type == Float {
			t = &Tensor{Name: in.Name, Type: Float, Dims: given.Dims, F: make([]float32, len(given.F))}
		} else {
			isConst[in.Name] = true
		}
		vals[in.Name] = t
	}
	p := &Plan{Opset: s.opset, Shapes: map[string][]int64{}, Consts: map[string]*Tensor{}}
	for name, t := range vals {
		p.Shapes[name] = t.Dims
		if isConst[name] {
			p.Consts[name] = t
		}
	}
	b := &budget{lim: s.lim}
	for i, n := range g.Nodes {
		st := Step{Node: n, Const: true}
		args := make([]*Tensor, len(n.Inputs))
		for j, name := range n.Inputs {
			if name == "" {
				st.InShapes = append(st.InShapes, nil)
				continue
			}
			args[j] = vals[name]
			st.InShapes = append(st.InShapes, vals[name].Dims)
			st.Const = st.Const && isConst[name]
		}
		outs, err := run(n, s.opset, args, b)
		if err != nil {
			return nil, fmt.Errorf("onnx: plan: node %d (%s %q): %w", i, n.Op, n.Name, err)
		}
		for j, name := range n.Outputs {
			if name == "" || j >= len(outs) {
				continue
			}
			vals[name] = outs[j]
			p.Shapes[name] = outs[j].Dims
			st.OutShapes = append(st.OutShapes, outs[j].Dims)
			if st.Const {
				isConst[name] = true
				p.Consts[name] = outs[j]
			}
		}
		p.Steps = append(p.Steps, st)
	}
	return p, nil
}

// Window is the geometry of a convolution or pooling over the spatial
// axes: sizes, kernel, strides, dilations, padding at both ends, output.
type Window struct {
	In, Kernel, Stride, Dil, PadB, PadE, Out []int64
}

// NodeWindow resolves a Conv, MaxPool or AveragePool node's window for an
// input of shape x (N, C, spatial...) and, for Conv, weights of shape w.
func NodeWindow(n *Node, x, w []int64) (*Window, error) {
	if len(x) < 3 {
		return nil, fmt.Errorf("input rank %d", len(x))
	}
	var kernel []int64
	ceil := false
	switch n.Op {
	case "Conv":
		if len(w) != len(x) {
			return nil, fmt.Errorf("weights %v for input %v", w, x)
		}
		kernel = w[2:]
	case "MaxPool", "AveragePool":
		kernel = attrInts(n, "kernel_shape")
		ceil = attrInt(n, "ceil_mode", 0) != 0
	default:
		return nil, fmt.Errorf("%s has no window", n.Op)
	}
	g, err := geometry(n, x[2:], kernel, ceil)
	if err != nil {
		return nil, err
	}
	return &Window{In: g.in, Kernel: g.kernel, Stride: g.stride, Dil: g.dil, PadB: g.padB, PadE: g.padE, Out: g.out}, nil
}

// Attribute accessors for other executors, with the operator defaults.

// AttrInt is an int attribute or def.
func (n *Node) AttrInt(name string, def int64) int64 { return attrInt(n, name, def) }

// AttrFloat is a float attribute or def.
func (n *Node) AttrFloat(name string, def float32) float32 { return attrFloat(n, name, def) }

// AttrInts is an ints attribute or nil.
func (n *Node) AttrInts(name string) []int64 { return attrInts(n, name) }

// NormAxis normalises an axis attribute for rank r (inclusive allows r).
func NormAxis(a int64, r int, inclusive bool) (int, error) { return axis(a, r, inclusive) }
