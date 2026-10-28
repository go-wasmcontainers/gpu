package onnx

import (
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"
)

// Supported lists the operators (default domain) this package runs, with
// the operator-set versions their semantics follow.
var Supported = map[string]string{
	"Add":         "7+ (multidirectional broadcasting)",
	"AveragePool": "7+ (ceil_mode from 10, dilations from 19)",
	"Concat":      "4+",
	"Constant":    "1+ (value attribute)",
	"Conv":        "1+",
	"Div":         "7+",
	"Dropout":     "1+ (inference: identity; no mask output)",
	"Flatten":     "1+",
	"Gemm":        "7+",
	"Identity":    "1+",
	"MatMul":      "1+",
	"MaxPool":     "1+ (no Indices output)",
	"Mul":         "7+",
	"Relu":        "6+",
	"Reshape":     "5+ (allowzero from 14)",
	"Softmax":     "1+ (2-D coercion before 13, one axis from 13)",
	"Sub":         "7+",
	"Transpose":   "1+",
}

// MinOpset is the oldest default-domain operator set Load accepts: before
// 7, Add and its kin broadcast through an attribute this package lacks.
const MinOpset = 7

// UnsupportedError lists every operator of a graph this package cannot
// run, with how many nodes use it.
type UnsupportedError struct {
	Ops map[string]int // "domain:Op" or "Op" -> nodes
}

func (e *UnsupportedError) Error() string {
	names := slices.Sorted(maps.Keys(e.Ops))
	parts := make([]string, len(names))
	for i, n := range names {
		parts[i] = fmt.Sprintf("%s (%d node", n, e.Ops[n])
		if e.Ops[n] > 1 {
			parts[i] += "s"
		}
		parts[i] += ")"
	}
	return "onnx: unsupported operators: " + strings.Join(parts, ", ") +
		"; supported: " + strings.Join(slices.Sorted(maps.Keys(Supported)), ", ")
}

// Session is a loaded model ready to run.
type Session struct {
	model  *Model
	opset  int64
	lim    Limits
	inputs []ValueInfo // graph inputs that are not initializers
}

// Load decodes a model and checks that this package can run it: opset,
// operators, and that the nodes are in topological order with every input
// defined.
func Load(b []byte, lim Limits) (*Session, error) {
	m, err := Decode(b, lim)
	if err != nil {
		return nil, err
	}
	return NewSession(m, lim)
}

// NewSession checks a decoded model (see Load).
func NewSession(m *Model, lim Limits) (*Session, error) {
	opset, ok := m.Opsets[""]
	if !ok {
		return nil, errors.New("onnx: the model imports no default-domain operator set")
	}
	if opset < MinOpset {
		return nil, fmt.Errorf("onnx: operator set %d is older than %d, the oldest supported", opset, MinOpset)
	}
	g := m.Graph
	bad := map[string]int{}
	for _, n := range g.Nodes {
		if n.Domain != "" {
			bad[n.Domain+":"+n.Op]++
			continue
		}
		if _, ok := Supported[n.Op]; !ok {
			bad[n.Op]++
		}
	}
	if len(bad) > 0 {
		return nil, &UnsupportedError{Ops: bad}
	}
	for _, vs := range [][]ValueInfo{g.Inputs, g.Outputs} {
		for _, v := range vs {
			if _, init := g.Initializers[v.Name]; !init && v.Type != Float && v.Type != Int64 {
				return nil, fmt.Errorf("onnx: graph value %q is not a float or int64 tensor (element type %v)", v.Name, v.Type)
			}
		}
	}
	s := &Session{model: m, opset: opset, lim: lim}
	defined := map[string]bool{"": true} // "" is an omitted optional input
	for name := range g.Initializers {
		defined[name] = true
	}
	for _, in := range g.Inputs {
		if _, init := g.Initializers[in.Name]; !init {
			if in.Name == "" || defined[in.Name] {
				return nil, fmt.Errorf("onnx: graph input %q unnamed or defined twice", in.Name)
			}
			s.inputs = append(s.inputs, in)
			defined[in.Name] = true
		}
	}
	for i, n := range g.Nodes {
		for _, in := range n.Inputs {
			if !defined[in] {
				return nil, fmt.Errorf("onnx: node %d (%s %q) reads %q before anything defines it", i, n.Op, n.Name, in)
			}
		}
		for _, out := range n.Outputs {
			if out == "" {
				continue
			}
			if defined[out] {
				return nil, fmt.Errorf("onnx: node %d (%s %q) redefines %q", i, n.Op, n.Name, out)
			}
			defined[out] = true
		}
		if err := checkNode(n); err != nil {
			return nil, fmt.Errorf("onnx: node %d (%s %q): %w", i, n.Op, n.Name, err)
		}
	}
	for _, out := range g.Outputs {
		if !defined[out.Name] {
			return nil, fmt.Errorf("onnx: graph output %q is never defined", out.Name)
		}
	}
	return s, nil
}

// checkNode refuses the variants of a supported operator this package
// does not implement, at load time.
func checkNode(n *Node) error {
	switch n.Op {
	case "MaxPool", "Dropout":
		if len(n.Outputs) > 1 && n.Outputs[1] != "" {
			return fmt.Errorf("its second output is not supported")
		}
		if n.Op == "Dropout" && len(n.Inputs) > 2 && n.Inputs[2] != "" {
			return fmt.Errorf("training_mode is not supported (inference only)")
		}
		if n.Op == "Dropout" && attrInt(n, "is_test", 1) == 0 {
			return fmt.Errorf("is_test=0 (training) is not supported")
		}
	case "Constant":
		if a := n.Attrs["value"]; a == nil || a.T == nil || len(n.Attrs) != 1 {
			return fmt.Errorf("only the 'value' tensor attribute is supported")
		}
	}
	return nil
}

// Inputs are the graph inputs the caller provides, in order.
func (s *Session) Inputs() []ValueInfo { return s.inputs }

// Outputs are the graph outputs, in order.
func (s *Session) Outputs() []ValueInfo { return s.model.Graph.Outputs }

// Opset is the model's default-domain operator-set version.
func (s *Session) Opset() int64 { return s.opset }

// Run computes the graph outputs from the inputs (by name). Every
// intermediate tensor is bounded by Limits.TensorElems and the whole run
// by Limits.MACs.
func (s *Session) Run(inputs map[string]*Tensor) (map[string]*Tensor, error) {
	g := s.model.Graph
	vals := make(map[string]*Tensor, len(g.Initializers)+len(g.Nodes))
	maps.Copy(vals, g.Initializers)
	for _, in := range s.inputs {
		t, ok := inputs[in.Name]
		if !ok {
			return nil, fmt.Errorf("onnx: input %q missing", in.Name)
		}
		if err := checkInput(in, t, s.lim); err != nil {
			return nil, err
		}
		vals[in.Name] = t
	}
	b := &budget{lim: s.lim}
	for i, n := range g.Nodes {
		args := make([]*Tensor, len(n.Inputs))
		for j, name := range n.Inputs {
			if name != "" {
				args[j] = vals[name]
			}
		}
		outs, err := run(n, s.opset, args, b)
		if err != nil {
			return nil, fmt.Errorf("onnx: node %d (%s %q): %w", i, n.Op, n.Name, err)
		}
		for j, name := range n.Outputs {
			if name != "" && j < len(outs) {
				vals[name] = outs[j]
			}
		}
	}
	out := make(map[string]*Tensor, len(g.Outputs))
	for _, o := range g.Outputs {
		out[o.Name] = vals[o.Name]
	}
	return out, nil
}

// checkInput compares a provided input with the graph's declaration.
func checkInput(in ValueInfo, t *Tensor, lim Limits) error {
	if t == nil {
		return fmt.Errorf("onnx: input %q is nil", in.Name)
	}
	if in.Type != 0 && t.Type != in.Type {
		return fmt.Errorf("onnx: input %q is %v, the model wants %v", in.Name, t.Type, in.Type)
	}
	if in.Dims != nil {
		if len(in.Dims) != len(t.Dims) {
			return fmt.Errorf("onnx: input %q has shape %v, the model wants %v", in.Name, t.Dims, in.Dims)
		}
		for i, d := range in.Dims {
			if d >= 0 && t.Dims[i] != d {
				return fmt.Errorf("onnx: input %q has shape %v, the model wants %v", in.Name, t.Dims, in.Dims)
			}
		}
	}
	if len(t.Dims) > lim.Dims {
		return fmt.Errorf("onnx: input %q has rank %d (limit %d)", in.Name, len(t.Dims), lim.Dims)
	}
	n, err := shapeLen(t.Dims, lim.TensorElems)
	if err != nil {
		return fmt.Errorf("onnx: input %q: %w", in.Name, err)
	}
	switch t.Type {
	case Float:
		if int64(len(t.F)) != n {
			return fmt.Errorf("onnx: input %q: %d values for shape %v", in.Name, len(t.F), t.Dims)
		}
	case Int64, Int32:
		if int64(len(t.I)) != n {
			return fmt.Errorf("onnx: input %q: %d values for shape %v", in.Name, len(t.I), t.Dims)
		}
	default:
		return fmt.Errorf("onnx: input %q: %v is not supported", in.Name, t.Type)
	}
	return nil
}

// shapeLen is the product of dims, refused past max or for a negative
// dimension, computed without overflow.
func shapeLen(dims []int64, max int64) (int64, error) {
	n := int64(1)
	for _, d := range dims {
		if d < 0 {
			return 0, fmt.Errorf("negative dimension in %v", dims)
		}
		if d > 0 && n > max/d {
			return 0, fmt.Errorf("shape %v exceeds %d elements", dims, max)
		}
		n *= d
	}
	return n, nil
}

// budget accounts for one Run's work.
type budget struct {
	lim  Limits
	macs int64
}

// spend charges n multiply-accumulates.
func (b *budget) spend(n int64) error {
	if n < 0 || n > b.lim.MACs-b.macs {
		return fmt.Errorf("the inference exceeds %d multiply-accumulates", b.lim.MACs)
	}
	b.macs += n
	return nil
}

// alloc returns a float tensor of shape dims, refused past the limits.
func (b *budget) alloc(dims []int64) (*Tensor, error) {
	if len(dims) > b.lim.Dims {
		return nil, fmt.Errorf("rank %d exceeds %d", len(dims), b.lim.Dims)
	}
	n, err := shapeLen(dims, b.lim.TensorElems)
	if err != nil {
		return nil, err
	}
	return &Tensor{Type: Float, Dims: slices.Clone(dims), F: make([]float32, n)}, nil
}
