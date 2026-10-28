package onnx

import (
	"os"
	"path/filepath"
	"testing"
)

// fuzzLimits keep one fuzz input cheap.
var fuzzLimits = Limits{ModelBytes: 1 << 20, Nodes: 64, Dims: 6, TensorElems: 1 << 16, TotalElems: 1 << 18, MACs: 1 << 22, Strings: 256}

func addModels(f *testing.F) {
	paths, _ := filepath.Glob("testdata/node/*/model.onnx")
	paths = append(paths, "testdata/models/mnist-12/mnist-12.onnx")
	for _, p := range paths {
		if b, err := os.ReadFile(p); err == nil {
			f.Add(b)
		}
	}
}

// FuzzRun: any model bytes either fail to load, or load and run on zero
// inputs (symbolic dimensions as 1) without panicking or exceeding the
// limits.
func FuzzRun(f *testing.F) {
	addModels(f)
	f.Fuzz(func(t *testing.T, b []byte) {
		s, err := Load(b, fuzzLimits)
		if err != nil {
			return
		}
		in := map[string]*Tensor{}
		for _, vi := range s.Inputs() {
			dims := make([]int64, len(vi.Dims))
			for i, d := range vi.Dims {
				dims[i] = max(d, 1)
			}
			n, err := shapeLen(dims, 4096)
			if err != nil {
				return
			}
			tt := &Tensor{Name: vi.Name, Type: vi.Type, Dims: dims}
			if vi.Type == Float {
				tt.F = make([]float32, n)
			} else {
				tt.I = make([]int64, n)
			}
			in[vi.Name] = tt
		}
		out, err := s.Run(in)
		if err != nil {
			return
		}
		for name, o := range out {
			if o == nil {
				continue
			}
			n, err := shapeLen(o.Dims, fuzzLimits.TensorElems)
			if err != nil {
				t.Fatalf("output %q of shape %v: %v", name, o.Dims, err)
			}
			if (o.Type == Float && int64(len(o.F)) != n) || (o.Type != Float && int64(len(o.I)) != n) {
				t.Fatalf("output %q: shape %v with %d/%d values", name, o.Dims, len(o.F), len(o.I))
			}
		}
	})
}

// FuzzDecodeTensor: a TensorProto decodes within the limits or fails.
func FuzzDecodeTensor(f *testing.F) {
	paths, _ := filepath.Glob("testdata/node/*/test_data_set_0/*.pb")
	for _, p := range paths[:min(len(paths), 64)] {
		if b, err := os.ReadFile(p); err == nil {
			f.Add(b)
		}
	}
	f.Fuzz(func(t *testing.T, b []byte) {
		tt, err := DecodeTensor(b, fuzzLimits)
		if err != nil {
			return
		}
		n, err := shapeLen(tt.Dims, fuzzLimits.TensorElems)
		if err != nil {
			t.Fatalf("decoded shape %v: %v", tt.Dims, err)
		}
		if (tt.Type == Float && int64(len(tt.F)) != n) || (tt.Type != Float && int64(len(tt.I)) != n) {
			t.Fatalf("shape %v with %d/%d values", tt.Dims, len(tt.F), len(tt.I))
		}
	})
}
