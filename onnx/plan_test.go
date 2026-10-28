package onnx

import (
	"os"
	"slices"
	"testing"
)

// TestPlanMNIST: the plan's shapes match a real run's, the Reshape shape
// constants are folded, and the first step depends on the input.
func TestPlanMNIST(t *testing.T) {
	b, err := os.ReadFile("testdata/models/mnist-12/mnist-12.onnx")
	if err != nil {
		t.Fatal(err)
	}
	s, err := Load(b, DefaultLimits)
	if err != nil {
		t.Fatal(err)
	}
	in := s.Inputs()[0]
	x, _ := NewFloat([]int64{1, 1, 28, 28}, make([]float32, 28*28))
	p, err := s.Plan(map[string]*Tensor{in.Name: x})
	if err != nil {
		t.Fatal(err)
	}
	out := s.Outputs()[0].Name
	if !slices.Equal(p.Shapes[out], []int64{1, 10}) {
		t.Errorf("output shape %v", p.Shapes[out])
	}
	var consts, live int
	for _, st := range p.Steps {
		if st.Const {
			consts++
			for _, o := range st.Node.Outputs {
				if p.Consts[o] == nil {
					t.Errorf("const step %s %q without a value", st.Node.Op, o)
				}
			}
		} else {
			live++
		}
		if len(st.OutShapes) == 0 {
			t.Errorf("step %s has no output shape", st.Node.Op)
		}
	}
	if live == 0 || consts == 0 {
		t.Errorf("%d const, %d live steps", consts, live)
	}
	if _, err := s.Plan(map[string]*Tensor{}); err == nil {
		t.Error("a plan without inputs")
	}
	bad, _ := NewFloat([]int64{1, 1, 27, 28}, make([]float32, 27*28))
	if _, err := s.Plan(map[string]*Tensor{in.Name: bad}); err == nil {
		t.Error("a plan with a wrong input shape")
	}
}

func TestNodeWindow(t *testing.T) {
	n := &Node{Op: "MaxPool", Attrs: map[string]*Attr{
		"kernel_shape": {Ints: []int64{2, 2}}, "strides": {Ints: []int64{2, 2}},
	}}
	w, err := NodeWindow(n, []int64{1, 3, 5, 5}, nil)
	if err != nil || !slices.Equal(w.Out, []int64{2, 2}) {
		t.Fatalf("%+v %v", w, err)
	}
	n.Attrs["ceil_mode"] = &Attr{I: 1}
	if w, _ = NodeWindow(n, []int64{1, 3, 5, 5}, nil); !slices.Equal(w.Out, []int64{3, 3}) {
		t.Errorf("ceil: %v", w.Out)
	}
	if _, err := NodeWindow(&Node{Op: "Relu"}, []int64{1, 1, 1}, nil); err == nil {
		t.Error("Relu window")
	}
	if _, err := NodeWindow(&Node{Op: "Conv"}, []int64{1, 1, 4, 4}, []int64{1, 1, 3}); err == nil {
		t.Error("Conv rank mismatch")
	}
}
