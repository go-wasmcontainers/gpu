package onnx

import (
	"errors"
	"os"
	"strings"
	"testing"
)

// TestMNIST runs the model zoo's MNIST-12 on its own test data set.
func TestMNIST(t *testing.T) {
	if err := runNodeTest2(t, "testdata/models/mnist-12/mnist-12.onnx", "testdata/models/mnist-12/test_data_set_0"); err != nil {
		t.Fatal(err)
	}
}

func runNodeTest2(t *testing.T, model, set string) error {
	t.Helper()
	b, err := os.ReadFile(model)
	if err != nil {
		return err
	}
	s, err := Load(b, DefaultLimits)
	if err != nil {
		return err
	}
	in, err := readTensor(set + "/input_0.pb")
	if err != nil {
		return err
	}
	want, err := readTensor(set + "/output_0.pb")
	if err != nil {
		return err
	}
	out, err := s.Run(map[string]*Tensor{s.Inputs()[0].Name: in})
	if err != nil {
		return err
	}
	return allClose(out[s.Outputs()[0].Name], want)
}

// TestRefusesUnsupportedOperators: the error names every unsupported
// operator with its node count, and the supported list.
func TestRefusesUnsupportedOperators(t *testing.T) {
	m := &Model{Opsets: map[string]int64{"": 13}, Graph: &Graph{
		Initializers: map[string]*Tensor{},
		Inputs:       []ValueInfo{{Name: "x", Type: Float, Dims: []int64{1}}},
		Outputs:      []ValueInfo{{Name: "z", Type: Float}},
		Nodes: []*Node{
			{Op: "Relu", Inputs: []string{"x"}, Outputs: []string{"a"}},
			{Op: "LSTM", Inputs: []string{"a"}, Outputs: []string{"b"}},
			{Op: "LSTM", Inputs: []string{"b"}, Outputs: []string{"c"}},
			{Op: "Foo", Domain: "com.example", Inputs: []string{"c"}, Outputs: []string{"z"}},
		},
	}}
	_, err := NewSession(m, DefaultLimits)
	var u *UnsupportedError
	if !errors.As(err, &u) {
		t.Fatalf("got %v, want an UnsupportedError", err)
	}
	msg := err.Error()
	for _, want := range []string{"LSTM (2 nodes)", "com.example:Foo (1 node)", "supported: Add, AveragePool"} {
		if !strings.Contains(msg, want) {
			t.Errorf("%q lacks %q", msg, want)
		}
	}
	if strings.Contains(msg, "Relu (") {
		t.Errorf("%q lists a supported operator", msg)
	}
}

func TestLoadRefusals(t *testing.T) {
	base := func() *Model {
		return &Model{Opsets: map[string]int64{"": 13}, Graph: &Graph{
			Initializers: map[string]*Tensor{},
			Inputs:       []ValueInfo{{Name: "x", Type: Float}},
			Outputs:      []ValueInfo{{Name: "y", Type: Float}},
			Nodes:        []*Node{{Op: "Relu", Inputs: []string{"x"}, Outputs: []string{"y"}}},
		}}
	}
	for name, c := range map[string]struct {
		edit func(*Model)
		want string
	}{
		"old opset":  {func(m *Model) { m.Opsets[""] = 6 }, "older than"},
		"no opset":   {func(m *Model) { delete(m.Opsets, "") }, "imports no default-domain"},
		"undefined":  {func(m *Model) { m.Graph.Nodes[0].Inputs[0] = "nope" }, "before anything defines it"},
		"redefined":  {func(m *Model) { m.Graph.Nodes[0].Outputs[0] = "x" }, "redefines"},
		"no output":  {func(m *Model) { m.Graph.Outputs[0].Name = "nope" }, "never defined"},
		"int8 input": {func(m *Model) { m.Graph.Inputs[0].Type = Int8 }, "not a float or int64"},
		"maxpool idx": {func(m *Model) {
			m.Graph.Nodes[0] = &Node{Op: "MaxPool", Inputs: []string{"x"}, Outputs: []string{"y", "i"}}
		}, "second output"},
		"constant attrs": {func(m *Model) { m.Graph.Nodes[0] = &Node{Op: "Constant", Outputs: []string{"y"}} }, "'value'"},
	} {
		m := base()
		c.edit(m)
		if _, err := NewSession(m, DefaultLimits); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: %v, want %q", name, err, c.want)
		}
	}
}

// TestRunLimits: an inference past the element or work budget stops with
// an error instead of allocating or spinning.
func TestRunLimits(t *testing.T) {
	m := &Model{Opsets: map[string]int64{"": 13}, Graph: &Graph{
		Initializers: map[string]*Tensor{"w": {Name: "w", Type: Float, Dims: []int64{64, 64}, F: make([]float32, 64*64)}},
		Inputs:       []ValueInfo{{Name: "x", Type: Float}},
		Outputs:      []ValueInfo{{Name: "y", Type: Float}},
		Nodes:        []*Node{{Op: "MatMul", Inputs: []string{"x", "w"}, Outputs: []string{"y"}}},
	}}
	x, _ := NewFloat([]int64{64, 64}, make([]float32, 64*64))
	lim := DefaultLimits
	lim.MACs = 64 * 64 * 63
	s, err := NewSession(m, lim)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Run(map[string]*Tensor{"x": x}); err == nil || !strings.Contains(err.Error(), "multiply-accumulates") {
		t.Errorf("MAC limit: %v", err)
	}
	lim = DefaultLimits
	lim.TensorElems = 64*64 - 1
	s, _ = NewSession(m, lim)
	if _, err := s.Run(map[string]*Tensor{"x": x}); err == nil || !strings.Contains(err.Error(), "exceeds") {
		t.Errorf("element limit: %v", err)
	}
	if _, err := s.Run(map[string]*Tensor{}); err == nil {
		t.Error("a missing input ran")
	}
}

func TestNewFloat(t *testing.T) {
	if _, err := NewFloat([]int64{2, 3}, make([]float32, 6)); err != nil {
		t.Error(err)
	}
	for _, dims := range [][]int64{{2, 4}, {-1, -6}, {1 << 40, 1 << 40}, {0, 1 << 62, 4}} {
		if _, err := NewFloat(dims, make([]float32, 6)); err == nil {
			t.Errorf("NewFloat(%v, 6 values) accepted", dims)
		}
	}
}
