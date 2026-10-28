package onnx

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// The ONNX backend's tolerances (onnx/backend/test/loader/__init__.py):
// numpy.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-7), i.e.
// |out - ref| <= atol + rtol*|ref|.
const (
	rtol = 1e-3
	atol = 1e-7
)

// TestNodeConformance runs every vendored ONNX node test (testdata/node,
// copied from onnx v1.22.0 by internal/cmd/onnxvendor: the tests whose
// models this package loads) and compares each output with the reference
// within the backend's tolerances.
func TestNodeConformance(t *testing.T) {
	dirs, err := filepath.Glob("testdata/node/test_*")
	if err != nil {
		t.Fatal(err)
	}
	if len(dirs) < 100 {
		t.Fatalf("only %d node tests vendored", len(dirs))
	}
	for _, dir := range dirs {
		t.Run(filepath.Base(dir), func(t *testing.T) {
			if err := runNodeTest(dir); err != nil {
				t.Error(err)
			}
		})
	}
}

func readTensor(path string) (*Tensor, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return DecodeTensor(b, DefaultLimits)
}

func runNodeTest(dir string) error {
	b, err := os.ReadFile(filepath.Join(dir, "model.onnx"))
	if err != nil {
		return err
	}
	s, err := Load(b, DefaultLimits)
	if err != nil {
		return err
	}
	sets, _ := filepath.Glob(filepath.Join(dir, "test_data_set_*"))
	if len(sets) == 0 {
		return fmt.Errorf("no test data")
	}
	for _, set := range sets {
		in := map[string]*Tensor{}
		for i, vi := range s.Inputs() {
			tt, err := readTensor(filepath.Join(set, fmt.Sprintf("input_%d.pb", i)))
			if err != nil {
				return fmt.Errorf("%s input %d: %w", filepath.Base(set), i, err)
			}
			in[vi.Name] = tt
		}
		out, err := s.Run(in)
		if err != nil {
			return fmt.Errorf("%s: %w", filepath.Base(set), err)
		}
		for i, vo := range s.Outputs() {
			want, err := readTensor(filepath.Join(set, fmt.Sprintf("output_%d.pb", i)))
			if err != nil {
				return fmt.Errorf("%s output %d: %w", filepath.Base(set), i, err)
			}
			if err := allClose(out[vo.Name], want); err != nil {
				return fmt.Errorf("%s output %q: %w", filepath.Base(set), vo.Name, err)
			}
		}
	}
	return nil
}

// allClose compares got with want: same shape and type, values within
// the tolerances (NaN equal to NaN).
func allClose(got, want *Tensor) error {
	if got == nil {
		return fmt.Errorf("no output")
	}
	if !slices.Equal(got.Dims, want.Dims) {
		return fmt.Errorf("shape %v, want %v", got.Dims, want.Dims)
	}
	if want.Type != got.Type {
		return fmt.Errorf("type %v, want %v", got.Type, want.Type)
	}
	if want.Type != Float {
		if !slices.Equal(got.I, want.I) {
			return fmt.Errorf("values %v, want %v", got.I, want.I)
		}
		return nil
	}
	var bad []string
	for i := range want.F {
		g, w := float64(got.F[i]), float64(want.F[i])
		if math.IsNaN(w) && math.IsNaN(g) {
			continue
		}
		if math.Abs(g-w) > atol+rtol*math.Abs(w) || math.IsNaN(g) != math.IsNaN(w) {
			if len(bad) < 5 {
				bad = append(bad, fmt.Sprintf("[%d] %g want %g", i, g, w))
			} else {
				bad = append(bad, "…")
				break
			}
		}
	}
	if bad != nil {
		return fmt.Errorf("values differ: %s", strings.Join(bad, ", "))
	}
	return nil
}
