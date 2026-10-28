package onnx

import (
	"errors"
	"fmt"
	"math"
	"slices"
)

func softmax(n *Node, opset int64, args []*Tensor, b *budget) ([]*Tensor, error) {
	x, err := argF(args, 0)
	if err != nil {
		return nil, err
	}
	r := len(x.Dims)
	if r == 0 {
		return nil, errors.New("Softmax of a scalar")
	}
	y, err := b.alloc(x.Dims)
	if err != nil {
		return nil, err
	}
	if err := b.spend(3 * int64(len(x.F))); err != nil {
		return nil, err
	}
	// From opset 13 the softmax runs along one axis (default -1); before,
	// the input is coerced to 2-D at axis (default 1) and each row of the
	// flattened inner part is normalised.
	var outer, length, inner int64
	if opset >= 13 {
		a, err := axis(attrInt(n, "axis", -1), r, false)
		if err != nil {
			return nil, err
		}
		outer, length, inner = prod(x.Dims[:a]), x.Dims[a], prod(x.Dims[a+1:])
	} else {
		a, err := axis(attrInt(n, "axis", 1), r, true)
		if err != nil {
			return nil, err
		}
		outer, length, inner = prod(x.Dims[:a]), prod(x.Dims[a:]), 1
	}
	for o := int64(0); o < outer; o++ {
		for in := int64(0); in < inner; in++ {
			base := o*length*inner + in
			mx := float32(math.Inf(-1))
			for i := int64(0); i < length; i++ {
				mx = max(mx, x.F[base+i*inner])
			}
			var sum float64
			for i := int64(0); i < length; i++ {
				e := float32(math.Exp(float64(x.F[base+i*inner] - mx)))
				y.F[base+i*inner] = e
				sum += float64(e)
			}
			for i := int64(0); i < length; i++ {
				y.F[base+i*inner] = float32(float64(y.F[base+i*inner]) / sum)
			}
		}
	}
	return []*Tensor{y}, nil
}

// view returns t with new dimensions over the same values.
func view(t *Tensor, dims []int64) *Tensor {
	return &Tensor{Type: t.Type, Dims: dims, F: t.F, I: t.I}
}

func reshape(n *Node, opset int64, args []*Tensor) ([]*Tensor, error) {
	if len(args) < 2 || args[0] == nil || args[1] == nil {
		return nil, errors.New("Reshape wants data and shape")
	}
	x, s := args[0], args[1]
	if s.Type != Int64 || len(s.Dims) != 1 {
		return nil, fmt.Errorf("shape must be a 1-D int64 tensor, got %v %v", s.Type, s.Dims)
	}
	allowZero := opset >= 14 && attrInt(n, "allowzero", 0) != 0
	total := prod(x.Dims)
	dims := make([]int64, len(s.I))
	infer := -1
	known := int64(1)
	for i, d := range s.I {
		switch {
		case d == -1:
			if infer >= 0 {
				return nil, errors.New("more than one -1 in the shape")
			}
			infer = i
			continue
		case d == 0 && !allowZero:
			if i >= len(x.Dims) {
				return nil, fmt.Errorf("shape %v copies dimension %d of %v, which does not exist", s.I, i, x.Dims)
			}
			d = x.Dims[i]
		case d < 0:
			return nil, fmt.Errorf("negative dimension in shape %v", s.I)
		}
		dims[i] = d
		if d > 0 && known > math.MaxInt64/d {
			return nil, fmt.Errorf("shape %v does not hold %v", s.I, x.Dims)
		}
		known *= d
	}
	if infer >= 0 {
		if known == 0 || total%known != 0 {
			return nil, fmt.Errorf("cannot infer -1 in %v for %v", s.I, x.Dims)
		}
		dims[infer] = total / known
	} else if known != total {
		return nil, fmt.Errorf("shape %v does not hold %v", s.I, x.Dims)
	}
	return []*Tensor{view(x, dims)}, nil
}

func flatten(n *Node, args []*Tensor) ([]*Tensor, error) {
	if len(args) == 0 || args[0] == nil {
		return nil, errors.New("input 0 missing")
	}
	x := args[0]
	a, err := axis(attrInt(n, "axis", 1), len(x.Dims), true)
	if err != nil {
		return nil, err
	}
	return []*Tensor{view(x, []int64{prod(x.Dims[:a]), prod(x.Dims[a:])})}, nil
}

func transpose(n *Node, args []*Tensor, b *budget) ([]*Tensor, error) {
	x, err := argF(args, 0)
	if err != nil {
		return nil, err
	}
	r := len(x.Dims)
	perm := attrInts(n, "perm")
	if perm == nil {
		perm = make([]int64, r)
		for i := range perm {
			perm[i] = int64(r - 1 - i)
		}
	}
	if len(perm) != r {
		return nil, fmt.Errorf("perm %v for rank %d", perm, r)
	}
	seen := make([]bool, r)
	out := make([]int64, r)
	for i, p := range perm {
		if p < 0 || p >= int64(r) || seen[p] {
			return nil, fmt.Errorf("perm %v is not a permutation", perm)
		}
		seen[p] = true
		out[i] = x.Dims[p]
	}
	y, err := b.alloc(out)
	if err != nil {
		return nil, err
	}
	if err := b.spend(int64(len(y.F))); err != nil {
		return nil, err
	}
	inStride := make([]int64, r)
	s := int64(1)
	for i := r - 1; i >= 0; i-- {
		inStride[i] = s
		s *= x.Dims[i]
	}
	st := make([]int64, r) // input stride of each output axis
	for i, p := range perm {
		st[i] = inStride[p]
	}
	idx := make([]int64, r)
	var off int64
	for i := range y.F {
		y.F[i] = x.F[off]
		for d := r - 1; d >= 0; d-- {
			idx[d]++
			off += st[d]
			if idx[d] < out[d] {
				break
			}
			off -= st[d] * idx[d]
			idx[d] = 0
		}
	}
	return []*Tensor{y}, nil
}

func concat(n *Node, args []*Tensor, b *budget) ([]*Tensor, error) {
	if len(args) == 0 || args[0] == nil {
		return nil, errors.New("Concat wants inputs")
	}
	if n.Attrs["axis"] == nil {
		return nil, errors.New("Concat needs an axis")
	}
	first := args[0]
	r := len(first.Dims)
	a, err := axis(attrInt(n, "axis", 0), r, false)
	if err != nil {
		return nil, err
	}
	out := slices.Clone(first.Dims)
	out[a] = 0
	for i, t := range args {
		if t == nil || t.Type != first.Type || len(t.Dims) != r {
			return nil, fmt.Errorf("input %d does not match input 0 in type or rank", i)
		}
		for d := range t.Dims {
			if d != a && t.Dims[d] != first.Dims[d] {
				return nil, fmt.Errorf("input %d of shape %v does not match %v", i, t.Dims, first.Dims)
			}
		}
		if out[a] > math.MaxInt32 {
			return nil, errors.New("Concat axis too long")
		}
		out[a] += t.Dims[a]
	}
	n0, err := shapeLen(out, b.lim.TensorElems)
	if err != nil {
		return nil, err
	}
	if err := b.spend(n0); err != nil {
		return nil, err
	}
	outer := prod(out[:a])
	inner := prod(out[a+1:])
	y := &Tensor{Type: first.Type, Dims: out}
	if first.Type == Float {
		y.F = make([]float32, 0, n0)
	} else {
		y.I = make([]int64, 0, n0)
	}
	for o := int64(0); o < outer; o++ {
		for _, t := range args {
			chunk := t.Dims[a] * inner
			if first.Type == Float {
				y.F = append(y.F, t.F[o*chunk:(o+1)*chunk]...)
			} else {
				y.I = append(y.I, t.I[o*chunk:(o+1)*chunk]...)
			}
		}
	}
	return []*Tensor{y}, nil
}
