package onnx

import (
	"encoding/binary"
	"fmt"
	"math"
)

// fill sets a decoded tensor's values from raw_data or its typed field,
// checking that their count is the dimensions' product. Double tensors
// become Float (F), Int32 ones are kept in I; other types are refused.
func fill(t *Tensor, n int64, hasRaw bool, raw []byte, f64 []float64, i32 []int64) error {
	count := func(got int) error {
		if int64(got) != n {
			return fmt.Errorf("%d values for %d elements", got, n)
		}
		return nil
	}
	switch t.Type {
	case Float:
		if hasRaw {
			if int64(len(raw)) != 4*n {
				return fmt.Errorf("raw_data of %d bytes for %d floats", len(raw), n)
			}
			t.F = make([]float32, n)
			for i := range t.F {
				t.F[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
			}
			return nil
		}
		return count(len(t.F))
	case Double:
		t.Type = Float
		if hasRaw {
			if int64(len(raw)) != 8*n {
				return fmt.Errorf("raw_data of %d bytes for %d doubles", len(raw), n)
			}
			t.F = make([]float32, n)
			for i := range t.F {
				t.F[i] = float32(math.Float64frombits(binary.LittleEndian.Uint64(raw[8*i:])))
			}
			return nil
		}
		if err := count(len(f64)); err != nil {
			return err
		}
		t.F = make([]float32, n)
		for i, v := range f64 {
			t.F[i] = float32(v)
		}
		return nil
	case Int64:
		if hasRaw {
			if int64(len(raw)) != 8*n {
				return fmt.Errorf("raw_data of %d bytes for %d int64", len(raw), n)
			}
			t.I = make([]int64, n)
			for i := range t.I {
				t.I[i] = int64(binary.LittleEndian.Uint64(raw[8*i:]))
			}
			return nil
		}
		return count(len(t.I))
	case Int32:
		if hasRaw {
			if int64(len(raw)) != 4*n {
				return fmt.Errorf("raw_data of %d bytes for %d int32", len(raw), n)
			}
			t.I = make([]int64, n)
			for i := range t.I {
				t.I[i] = int64(int32(binary.LittleEndian.Uint32(raw[4*i:])))
			}
			return nil
		}
		if err := count(len(i32)); err != nil {
			return err
		}
		t.I = make([]int64, n)
		for i, v := range i32 {
			t.I[i] = int64(int32(v))
		}
		return nil
	}
	return fmt.Errorf("%v tensors are not supported (float, double, int32, int64 are)", t.Type)
}

// DecodeTensor parses an encoded TensorProto (the input_N.pb / output_N.pb
// files of the ONNX test data) within lim.
func DecodeTensor(b []byte, lim Limits) (*Tensor, error) {
	if len(b) > lim.ModelBytes {
		return nil, fmt.Errorf("onnx: tensor of %d bytes exceeds the limit of %d", len(b), lim.ModelBytes)
	}
	d := &decoder{lim: lim}
	t, err := d.tensor(b)
	if err != nil {
		return nil, fmt.Errorf("onnx: %w", err)
	}
	return t, nil
}

// NewFloat returns a float tensor of the given shape over data (not
// copied); its length must be the shape's product.
func NewFloat(dims []int64, data []float32) (*Tensor, error) {
	// dims come from a guest: no negative one, and the product is checked
	// against the data's length step by step, so it cannot overflow.
	n := int64(1)
	for _, d := range dims {
		if d < 0 || (d > 0 && n > int64(len(data))/d) {
			return nil, fmt.Errorf("onnx: %d values for shape %v", len(data), dims)
		}
		n *= d
	}
	if n != int64(len(data)) {
		return nil, fmt.Errorf("onnx: %d values for shape %v", len(data), dims)
	}
	return &Tensor{Type: Float, Dims: dims, F: data}, nil
}
