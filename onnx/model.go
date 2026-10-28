// Package onnx loads ONNX models (onnx/onnx.proto) and runs a bounded set
// of operators on the CPU, in pure Go.
//
// A model is untrusted input: Decode bounds every size (Limits), refuses
// what it does not implement with an error naming it, and is fuzzed.
// Load further refuses a graph that uses an operator outside Supported,
// listing every such operator, so a container learns at load time, not
// half-way through an inference, that its model cannot run here.
package onnx

import (
	"errors"
	"fmt"
	"math"
)

// DataType is TensorProto.DataType.
type DataType int32

// The data types this package reads (onnx.proto TensorProto.DataType).
const (
	Float  DataType = 1
	Uint8  DataType = 2
	Int8   DataType = 3
	Int32  DataType = 6
	Int64  DataType = 7
	Double DataType = 11
)

func (t DataType) String() string {
	switch t {
	case Float:
		return "float"
	case Uint8:
		return "uint8"
	case Int8:
		return "int8"
	case Int32:
		return "int32"
	case Int64:
		return "int64"
	case Double:
		return "double"
	}
	return fmt.Sprintf("data type %d", int32(t))
}

// Tensor is a dense tensor in row-major order. Float tensors use F,
// integer ones (Int64, Int32) I; other types are converted on decoding
// (Double to Float; Uint8 and Int8 to Float, as their values).
type Tensor struct {
	Name string
	Dims []int64
	Type DataType
	F    []float32
	I    []int64
}

// Len is the product of the dimensions (1 for a scalar).
func (t *Tensor) Len() int64 {
	n := int64(1)
	for _, d := range t.Dims {
		n *= d
	}
	return n
}

// ValueInfo is a graph input or output: its name, element type and
// dimensions (-1 for a symbolic or unknown one).
type ValueInfo struct {
	Name string
	Type DataType
	Dims []int64
}

// AttrType is AttributeProto.AttributeType.
type AttrType int32

// Attribute types this package reads.
const (
	AttrFloat   AttrType = 1
	AttrInt     AttrType = 2
	AttrString  AttrType = 3
	AttrTensor  AttrType = 4
	AttrGraph   AttrType = 5
	AttrFloats  AttrType = 6
	AttrInts    AttrType = 7
	AttrStrings AttrType = 8
)

// Attr is a node attribute.
type Attr struct {
	Name   string
	Type   AttrType
	F      float32
	I      int64
	S      string
	T      *Tensor
	Floats []float32
	Ints   []int64
}

// Node is one operator application.
type Node struct {
	Name, Op, Domain string
	Inputs, Outputs  []string
	Attrs            map[string]*Attr
}

// Graph is a model's main graph.
type Graph struct {
	Name         string
	Nodes        []*Node
	Initializers map[string]*Tensor
	Inputs       []ValueInfo
	Outputs      []ValueInfo
}

// Model is a decoded ModelProto.
type Model struct {
	IRVersion int64
	// Opsets maps an operator-set domain ("" is the default ai.onnx) to
	// its version.
	Opsets   map[string]int64
	Producer string
	Graph    *Graph
}

// Limits bound what a model may make this package allocate or compute.
type Limits struct {
	// ModelBytes bounds the encoded model.
	ModelBytes int
	// Nodes bounds the graph's nodes.
	Nodes int
	// Dims bounds a tensor's rank.
	Dims int
	// TensorElems bounds one tensor's elements (initializers, inputs and
	// every intermediate result).
	TensorElems int64
	// TotalElems bounds the sum of the initializers' elements.
	TotalElems int64
	// MACs bounds the multiply-accumulates of one inference.
	MACs int64
	// Strings bounds a name or attribute string, in bytes.
	Strings int
}

// DefaultLimits suit small vision models (MNIST-sized to a few MB).
var DefaultLimits = Limits{
	ModelBytes:  64 << 20,
	Nodes:       4096,
	Dims:        8,
	TensorElems: 16 << 20,
	TotalElems:  32 << 20,
	MACs:        4 << 30,
	Strings:     4096,
}

// decoder carries the limits and running totals of one Decode.
type decoder struct {
	lim   Limits
	total int64 // elements of every tensor decoded so far
}

// Decode parses an encoded ModelProto within lim. It does not check the
// operators (Load does).
func Decode(b []byte, lim Limits) (*Model, error) {
	if len(b) > lim.ModelBytes {
		return nil, fmt.Errorf("onnx: model of %d bytes exceeds the limit of %d", len(b), lim.ModelBytes)
	}
	d := &decoder{lim: lim}
	m, err := d.model(b)
	if err != nil {
		return nil, fmt.Errorf("onnx: %w", err)
	}
	if m.Graph == nil {
		return nil, errors.New("onnx: the model has no graph")
	}
	return m, nil
}

func (d *decoder) model(b []byte) (*Model, error) {
	m := &Model{Opsets: map[string]int64{}}
	r := pbReader{b: b}
	for !r.done() {
		f, w, err := r.key()
		if err != nil {
			return nil, err
		}
		switch {
		case f == 1 && w == wireVarint:
			v, err := r.varint()
			if err != nil {
				return nil, err
			}
			m.IRVersion = int64(v)
		case f == 2 && w == wireLen:
			if m.Producer, err = r.str(w, d.lim.Strings); err != nil {
				return nil, err
			}
		case f == 7 && w == wireLen:
			if m.Graph != nil {
				return nil, errors.New("two graphs")
			}
			sub, err := r.bytes()
			if err != nil {
				return nil, err
			}
			if m.Graph, err = d.graph(sub); err != nil {
				return nil, fmt.Errorf("graph: %w", err)
			}
		case f == 8 && w == wireLen:
			sub, err := r.bytes()
			if err != nil {
				return nil, err
			}
			dom, ver, err := d.opset(sub)
			if err != nil {
				return nil, fmt.Errorf("opset_import: %w", err)
			}
			if len(m.Opsets) >= 64 {
				return nil, errors.New("more than 64 opset imports")
			}
			m.Opsets[dom] = ver
		case f == 25 && w == wireLen:
			return nil, errors.New("model-local functions are not supported")
		default:
			if err := r.skip(w); err != nil {
				return nil, err
			}
		}
	}
	return m, nil
}

func (d *decoder) opset(b []byte) (string, int64, error) {
	r := pbReader{b: b}
	var dom string
	var ver int64
	for !r.done() {
		f, w, err := r.key()
		if err != nil {
			return "", 0, err
		}
		switch {
		case f == 1 && w == wireLen:
			if dom, err = r.str(w, d.lim.Strings); err != nil {
				return "", 0, err
			}
		case f == 2 && w == wireVarint:
			v, err := r.varint()
			if err != nil {
				return "", 0, err
			}
			ver = int64(v)
		default:
			if err := r.skip(w); err != nil {
				return "", 0, err
			}
		}
	}
	if dom == "ai.onnx" {
		dom = ""
	}
	return dom, ver, nil
}

func (d *decoder) graph(b []byte) (*Graph, error) {
	g := &Graph{Initializers: map[string]*Tensor{}}
	r := pbReader{b: b}
	for !r.done() {
		f, w, err := r.key()
		if err != nil {
			return nil, err
		}
		if w != wireLen {
			if err := r.skip(w); err != nil {
				return nil, err
			}
			continue
		}
		sub, err := r.bytes()
		if err != nil {
			return nil, err
		}
		switch f {
		case 1:
			if len(g.Nodes) >= d.lim.Nodes {
				return nil, fmt.Errorf("more than %d nodes", d.lim.Nodes)
			}
			n, err := d.node(sub)
			if err != nil {
				return nil, fmt.Errorf("node %d: %w", len(g.Nodes), err)
			}
			g.Nodes = append(g.Nodes, n)
		case 2:
			if g.Name, err = (&pbReader{b: sub}).strAll(d.lim.Strings); err != nil {
				return nil, err
			}
		case 5:
			t, err := d.tensor(sub)
			if err != nil {
				return nil, fmt.Errorf("initializer %d: %w", len(g.Initializers), err)
			}
			if _, dup := g.Initializers[t.Name]; dup || t.Name == "" {
				return nil, fmt.Errorf("initializer %q unnamed or defined twice", t.Name)
			}
			g.Initializers[t.Name] = t
		case 11, 12:
			if len(g.Inputs)+len(g.Outputs) >= d.lim.Nodes {
				return nil, fmt.Errorf("more than %d inputs and outputs", d.lim.Nodes)
			}
			v, err := d.valueInfo(sub)
			if err != nil {
				return nil, fmt.Errorf("value info: %w", err)
			}
			if f == 11 {
				g.Inputs = append(g.Inputs, v)
			} else {
				g.Outputs = append(g.Outputs, v)
			}
		case 15:
			return nil, errors.New("sparse initializers are not supported")
		}
	}
	return g, nil
}

// strAll reads a whole buffer as a string.
func (r *pbReader) strAll(max int) (string, error) {
	if len(r.b) > max {
		return "", fmt.Errorf("string of %d bytes (limit %d)", len(r.b), max)
	}
	return string(r.b), nil
}

func (d *decoder) node(b []byte) (*Node, error) {
	n := &Node{Attrs: map[string]*Attr{}}
	r := pbReader{b: b}
	for !r.done() {
		f, w, err := r.key()
		if err != nil {
			return nil, err
		}
		switch {
		case (f == 1 || f == 2) && w == wireLen:
			s, err := r.str(w, d.lim.Strings)
			if err != nil {
				return nil, err
			}
			if len(n.Inputs)+len(n.Outputs) >= 1024 {
				return nil, errors.New("more than 1024 inputs and outputs")
			}
			if f == 1 {
				n.Inputs = append(n.Inputs, s)
			} else {
				n.Outputs = append(n.Outputs, s)
			}
		case f == 3 && w == wireLen:
			if n.Name, err = r.str(w, d.lim.Strings); err != nil {
				return nil, err
			}
		case f == 4 && w == wireLen:
			if n.Op, err = r.str(w, d.lim.Strings); err != nil {
				return nil, err
			}
		case f == 7 && w == wireLen:
			if n.Domain, err = r.str(w, d.lim.Strings); err != nil {
				return nil, err
			}
		case f == 5 && w == wireLen:
			sub, err := r.bytes()
			if err != nil {
				return nil, err
			}
			a, err := d.attr(sub)
			if err != nil {
				return nil, fmt.Errorf("attribute %q: %w", a.Name, err)
			}
			if _, dup := n.Attrs[a.Name]; dup || len(n.Attrs) >= 64 {
				return nil, fmt.Errorf("attribute %q defined twice, or more than 64", a.Name)
			}
			n.Attrs[a.Name] = a
		default:
			if err := r.skip(w); err != nil {
				return nil, err
			}
		}
	}
	if n.Domain == "ai.onnx" {
		n.Domain = ""
	}
	return n, nil
}

func (d *decoder) attr(b []byte) (*Attr, error) {
	a := &Attr{}
	r := pbReader{b: b}
	const maxList = 1 << 16
	for !r.done() {
		f, w, err := r.key()
		if err != nil {
			return a, err
		}
		switch {
		case f == 1 && w == wireLen:
			if a.Name, err = r.str(w, d.lim.Strings); err != nil {
				return a, err
			}
		case f == 20 && w == wireVarint:
			v, err := r.varint()
			if err != nil {
				return a, err
			}
			a.Type = AttrType(int32(v))
		case f == 2 && w == wireI32:
			v, err := r.fixed32()
			if err != nil {
				return a, err
			}
			a.F = math.Float32frombits(v)
		case f == 3 && w == wireVarint:
			v, err := r.varint()
			if err != nil {
				return a, err
			}
			a.I = int64(v)
		case f == 4 && w == wireLen:
			if a.S, err = r.str(w, d.lim.Strings); err != nil {
				return a, err
			}
		case f == 5 && w == wireLen:
			sub, err := r.bytes()
			if err != nil {
				return a, err
			}
			if a.T, err = d.tensor(sub); err != nil {
				return a, err
			}
		case f == 7:
			if a.Floats, err = r.float32s(w, a.Floats, maxList); err != nil {
				return a, err
			}
		case f == 8:
			if a.Ints, err = r.int64s(w, a.Ints, maxList); err != nil {
				return a, err
			}
		default:
			// Graphs, strings lists, sparse tensors, type protos: kept
			// out; the operators that need them are not supported.
			if err := r.skip(w); err != nil {
				return a, err
			}
		}
	}
	return a, nil
}

func (d *decoder) valueInfo(b []byte) (ValueInfo, error) {
	var v ValueInfo
	r := pbReader{b: b}
	for !r.done() {
		f, w, err := r.key()
		if err != nil {
			return v, err
		}
		switch {
		case f == 1 && w == wireLen:
			if v.Name, err = r.str(w, d.lim.Strings); err != nil {
				return v, err
			}
		case f == 2 && w == wireLen:
			sub, err := r.bytes()
			if err != nil {
				return v, err
			}
			if err := d.typeProto(sub, &v); err != nil {
				return v, err
			}
		default:
			if err := r.skip(w); err != nil {
				return v, err
			}
		}
	}
	return v, nil
}

// typeProto reads TypeProto.tensor_type (field 1): elem_type (1) and
// shape (2) of dims (1), each dim_value (1) or dim_param (2).
func (d *decoder) typeProto(b []byte, v *ValueInfo) error {
	r := pbReader{b: b}
	for !r.done() {
		f, w, err := r.key()
		if err != nil {
			return err
		}
		if f != 1 || w != wireLen {
			if err := r.skip(w); err != nil {
				return err
			}
			continue
		}
		tt, err := r.bytes()
		if err != nil {
			return err
		}
		tr := pbReader{b: tt}
		for !tr.done() {
			f, w, err := tr.key()
			if err != nil {
				return err
			}
			switch {
			case f == 1 && w == wireVarint:
				e, err := tr.varint()
				if err != nil {
					return err
				}
				v.Type = DataType(int32(e))
			case f == 2 && w == wireLen:
				shape, err := tr.bytes()
				if err != nil {
					return err
				}
				sr := pbReader{b: shape}
				v.Dims = []int64{}
				for !sr.done() {
					f, w, err := sr.key()
					if err != nil {
						return err
					}
					if f != 1 || w != wireLen {
						if err := sr.skip(w); err != nil {
							return err
						}
						continue
					}
					dim, err := sr.bytes()
					if err != nil {
						return err
					}
					if len(v.Dims) >= d.lim.Dims {
						return fmt.Errorf("rank above %d", d.lim.Dims)
					}
					v.Dims = append(v.Dims, dimValue(dim))
				}
			default:
				if err := tr.skip(w); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

// dimValue is a Dimension's dim_value, or -1 for a dim_param or nothing.
func dimValue(b []byte) int64 {
	r := pbReader{b: b}
	for !r.done() {
		f, w, err := r.key()
		if err != nil {
			return -1
		}
		if f == 1 && w == wireVarint {
			v, err := r.varint()
			if err != nil || int64(v) < 0 {
				return -1
			}
			return int64(v)
		}
		if r.skip(w) != nil {
			return -1
		}
	}
	return -1
}

func (d *decoder) tensor(b []byte) (*Tensor, error) {
	t := &Tensor{}
	r := pbReader{b: b}
	var raw []byte
	var f64 []float64
	var i32 []int64
	var hasRaw bool
	max := int(min(d.lim.TensorElems, math.MaxInt32))
	for !r.done() {
		f, w, err := r.key()
		if err != nil {
			return nil, err
		}
		switch {
		case f == 1:
			if t.Dims, err = r.int64s(w, t.Dims, d.lim.Dims); err != nil {
				return nil, fmt.Errorf("dims: %w", err)
			}
		case f == 2 && w == wireVarint:
			v, err := r.varint()
			if err != nil {
				return nil, err
			}
			t.Type = DataType(int32(v))
		case f == 4:
			if t.F, err = r.float32s(w, t.F, max); err != nil {
				return nil, fmt.Errorf("float_data: %w", err)
			}
		case f == 5:
			if i32, err = r.int64s(w, i32, max); err != nil {
				return nil, fmt.Errorf("int32_data: %w", err)
			}
		case f == 7:
			if t.I, err = r.int64s(w, t.I, max); err != nil {
				return nil, fmt.Errorf("int64_data: %w", err)
			}
		case f == 8 && w == wireLen:
			if t.Name, err = r.str(w, d.lim.Strings); err != nil {
				return nil, err
			}
		case f == 9 && w == wireLen:
			if raw, err = r.bytes(); err != nil {
				return nil, err
			}
			hasRaw = true
		case f == 10:
			if f64, err = r.float64s(w, f64, max); err != nil {
				return nil, fmt.Errorf("double_data: %w", err)
			}
		case f == 3, f == 13, f == 14:
			return nil, fmt.Errorf("tensor %q: segments and external data are not supported", t.Name)
		default:
			if err := r.skip(w); err != nil {
				return nil, err
			}
		}
	}
	n, err := d.elems(t.Dims)
	if err != nil {
		return nil, fmt.Errorf("tensor %q: %w", t.Name, err)
	}
	if err := fill(t, n, hasRaw, raw, f64, i32); err != nil {
		return nil, fmt.Errorf("tensor %q: %w", t.Name, err)
	}
	return t, nil
}

// elems checks dimensions and counts elements against the limits.
func (d *decoder) elems(dims []int64) (int64, error) {
	n := int64(1)
	for _, x := range dims {
		if x < 0 {
			return 0, fmt.Errorf("negative dimension %d", x)
		}
		if x > 0 && n > d.lim.TensorElems/x {
			return 0, fmt.Errorf("more than %d elements", d.lim.TensorElems)
		}
		n *= x
	}
	if n > d.lim.TensorElems {
		return 0, fmt.Errorf("more than %d elements", d.lim.TensorElems)
	}
	d.total += n
	if d.total > d.lim.TotalElems {
		return 0, fmt.Errorf("the model's tensors exceed %d elements", d.lim.TotalElems)
	}
	return n, nil
}
