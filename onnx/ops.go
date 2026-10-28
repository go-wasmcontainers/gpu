package onnx

import (
	"errors"
	"fmt"
	"slices"
)

// Operator semantics follow docs/Operators.md and docs/Changelog.md of
// github.com/onnx/onnx; each kernel is checked against the node test data
// of onnx/backend/test/data/node (testdata/node, see node_test.go).

func attrInt(n *Node, name string, def int64) int64 {
	if a := n.Attrs[name]; a != nil {
		return a.I
	}
	return def
}

func attrFloat(n *Node, name string, def float32) float32 {
	if a := n.Attrs[name]; a != nil {
		return a.F
	}
	return def
}

func attrInts(n *Node, name string) []int64 {
	if a := n.Attrs[name]; a != nil {
		return a.Ints
	}
	return nil
}

func attrString(n *Node, name, def string) string {
	if a := n.Attrs[name]; a != nil {
		return a.S
	}
	return def
}

// arg returns input i, which must be present and a float tensor.
func argF(args []*Tensor, i int) (*Tensor, error) {
	if i >= len(args) || args[i] == nil {
		return nil, fmt.Errorf("input %d missing", i)
	}
	if args[i].Type != Float {
		return nil, fmt.Errorf("input %d is %v, want float", i, args[i].Type)
	}
	return args[i], nil
}

// axis normalises an axis in [-r, r) (or [-r, r] when inclusive).
func axis(a int64, r int, inclusive bool) (int, error) {
	hi := int64(r)
	if inclusive {
		hi++
	}
	if a < -int64(r) || a >= hi {
		return 0, fmt.Errorf("axis %d out of range for rank %d", a, r)
	}
	if a < 0 {
		a += int64(r)
	}
	return int(a), nil
}

func prod(d []int64) int64 {
	n := int64(1)
	for _, x := range d {
		n *= x
	}
	return n
}

// run executes one node.
func run(n *Node, opset int64, args []*Tensor, b *budget) ([]*Tensor, error) {
	switch n.Op {
	case "Identity", "Dropout":
		if len(args) == 0 || args[0] == nil {
			return nil, errors.New("input 0 missing")
		}
		return []*Tensor{args[0]}, nil
	case "Constant":
		return []*Tensor{n.Attrs["value"].T}, nil
	case "Relu":
		x, err := argF(args, 0)
		if err != nil {
			return nil, err
		}
		y, err := b.alloc(x.Dims)
		if err != nil {
			return nil, err
		}
		if err := b.spend(int64(len(x.F))); err != nil {
			return nil, err
		}
		for i, v := range x.F {
			if v > 0 {
				y.F[i] = v
			}
		}
		return []*Tensor{y}, nil
	case "Add", "Sub", "Mul", "Div":
		return elementwise(n.Op, args, b)
	case "MatMul":
		return matmul(args, b)
	case "Gemm":
		return gemm(n, args, b)
	case "Conv":
		return conv(n, args, b)
	case "MaxPool", "AveragePool":
		return pool(n, args, b)
	case "Softmax":
		return softmax(n, opset, args, b)
	case "Reshape":
		return reshape(n, opset, args)
	case "Flatten":
		return flatten(n, args)
	case "Transpose":
		return transpose(n, args, b)
	case "Concat":
		return concat(n, args, b)
	}
	return nil, fmt.Errorf("operator %s not implemented", n.Op)
}

// broadcastShape is the multidirectional broadcast of a and b (numpy).
func broadcastShape(a, b []int64) ([]int64, error) {
	r := max(len(a), len(b))
	out := make([]int64, r)
	for i := 0; i < r; i++ {
		da, db := int64(1), int64(1)
		if j := len(a) - r + i; j >= 0 {
			da = a[j]
		}
		if j := len(b) - r + i; j >= 0 {
			db = b[j]
		}
		switch {
		case da == db, db == 1:
			out[i] = da
		case da == 1:
			out[i] = db
		default:
			return nil, fmt.Errorf("shapes %v and %v do not broadcast", a, b)
		}
	}
	return out, nil
}

// broadcastStrides are the strides of a tensor of shape d read as shape
// out (0 along broadcast dimensions).
func broadcastStrides(d, out []int64) []int64 {
	s := make([]int64, len(out))
	stride := int64(1)
	for i := len(out) - 1; i >= 0; i-- {
		j := len(d) - len(out) + i
		if j >= 0 && d[j] != 1 {
			s[i] = stride
		}
		if j >= 0 {
			stride *= d[j]
		}
	}
	return s
}

func elementwise(op string, args []*Tensor, b *budget) ([]*Tensor, error) {
	x, err := argF(args, 0)
	if err != nil {
		return nil, err
	}
	z, err := argF(args, 1)
	if err != nil {
		return nil, err
	}
	shape, err := broadcastShape(x.Dims, z.Dims)
	if err != nil {
		return nil, err
	}
	y, err := b.alloc(shape)
	if err != nil {
		return nil, err
	}
	if err := b.spend(int64(len(y.F))); err != nil {
		return nil, err
	}
	f := map[string]func(p, q float32) float32{
		"Add": func(p, q float32) float32 { return p + q },
		"Sub": func(p, q float32) float32 { return p - q },
		"Mul": func(p, q float32) float32 { return p * q },
		"Div": func(p, q float32) float32 { return p / q },
	}[op]
	if slices.Equal(x.Dims, z.Dims) {
		for i := range y.F {
			y.F[i] = f(x.F[i], z.F[i])
		}
		return []*Tensor{y}, nil
	}
	sx, sz := broadcastStrides(x.Dims, shape), broadcastStrides(z.Dims, shape)
	idx := make([]int64, len(shape))
	var ix, iz int64
	for i := range y.F {
		y.F[i] = f(x.F[ix], z.F[iz])
		for d := len(shape) - 1; d >= 0; d-- {
			idx[d]++
			ix += sx[d]
			iz += sz[d]
			if idx[d] < shape[d] {
				break
			}
			ix -= sx[d] * idx[d]
			iz -= sz[d] * idx[d]
			idx[d] = 0
		}
	}
	return []*Tensor{y}, nil
}

// gemmKernel adds alpha*A*B to C (row-major, A m×k, B k×n), i-k-j order so
// the inner loop runs over contiguous rows.
func gemmKernel(m, k, n int, alpha float32, a []float32, as0, as1 int, bm []float32, bs0, bs1 int, c []float32) {
	for i := 0; i < m; i++ {
		row := c[i*n : i*n+n]
		for p := 0; p < k; p++ {
			av := alpha * a[i*as0+p*as1]
			if av == 0 {
				continue
			}
			if bs1 == 1 {
				br := bm[p*bs0 : p*bs0+n]
				for j, bv := range br {
					row[j] += av * bv
				}
				continue
			}
			for j := 0; j < n; j++ {
				row[j] += av * bm[p*bs0+j*bs1]
			}
		}
	}
}

func matmul(args []*Tensor, b *budget) ([]*Tensor, error) {
	x, err := argF(args, 0)
	if err != nil {
		return nil, err
	}
	w, err := argF(args, 1)
	if err != nil {
		return nil, err
	}
	if len(x.Dims) == 0 || len(w.Dims) == 0 {
		return nil, errors.New("MatMul of a scalar")
	}
	ad, bd := slices.Clone(x.Dims), slices.Clone(w.Dims)
	dropRow, dropCol := false, false
	if len(ad) == 1 {
		ad = []int64{1, ad[0]}
		dropRow = true
	}
	if len(bd) == 1 {
		bd = []int64{bd[0], 1}
		dropCol = true
	}
	m, k := ad[len(ad)-2], ad[len(ad)-1]
	k2, n := bd[len(bd)-2], bd[len(bd)-1]
	if k != k2 {
		return nil, fmt.Errorf("shapes %v and %v do not multiply", x.Dims, w.Dims)
	}
	batch, err := broadcastShape(ad[:len(ad)-2], bd[:len(bd)-2])
	if err != nil {
		return nil, err
	}
	outDims := append(slices.Clone(batch), m, n)
	y, err := b.alloc(outDims)
	if err != nil {
		return nil, err
	}
	nb := prod(batch)
	if err := b.spend(nb * m * n * k); err != nil {
		return nil, err
	}
	sa, sb := broadcastStrides(ad[:len(ad)-2], batch), broadcastStrides(bd[:len(bd)-2], batch)
	idx := make([]int64, len(batch))
	var ia, ib int64
	for bi := int64(0); bi < nb; bi++ {
		gemmKernel(int(m), int(k), int(n), 1, x.F[ia*m*k:], int(k), 1, w.F[ib*k*n:], int(n), 1, y.F[bi*m*n:])
		for d := len(batch) - 1; d >= 0; d-- {
			idx[d]++
			ia += sa[d]
			ib += sb[d]
			if idx[d] < batch[d] {
				break
			}
			ia -= sa[d] * idx[d]
			ib -= sb[d] * idx[d]
			idx[d] = 0
		}
	}
	switch {
	case dropRow && dropCol:
		y.Dims = batch
	case dropRow:
		y.Dims = append(slices.Clone(batch), n)
	case dropCol:
		y.Dims = append(slices.Clone(batch), m)
	}
	return []*Tensor{y}, nil
}

func gemm(n *Node, args []*Tensor, b *budget) ([]*Tensor, error) {
	a, err := argF(args, 0)
	if err != nil {
		return nil, err
	}
	w, err := argF(args, 1)
	if err != nil {
		return nil, err
	}
	if len(a.Dims) != 2 || len(w.Dims) != 2 {
		return nil, fmt.Errorf("Gemm wants 2-D inputs, got %v and %v", a.Dims, w.Dims)
	}
	alpha, beta := attrFloat(n, "alpha", 1), attrFloat(n, "beta", 1)
	m, k := a.Dims[0], a.Dims[1]
	as0, as1 := int(k), 1
	if attrInt(n, "transA", 0) != 0 {
		m, k = k, m
		as0, as1 = 1, int(m)
	}
	k2, nn := w.Dims[0], w.Dims[1]
	bs0, bs1 := int(nn), 1
	if attrInt(n, "transB", 0) != 0 {
		k2, nn = nn, k2
		bs0, bs1 = 1, int(k2)
	}
	if k != k2 {
		return nil, fmt.Errorf("shapes %v and %v do not multiply", a.Dims, w.Dims)
	}
	y, err := b.alloc([]int64{m, nn})
	if err != nil {
		return nil, err
	}
	if err := b.spend(m * nn * k); err != nil {
		return nil, err
	}
	if len(args) > 2 && args[2] != nil && beta != 0 {
		c, err := argF(args, 2)
		if err != nil {
			return nil, err
		}
		shape, err := broadcastShape(c.Dims, y.Dims)
		if err != nil || !slices.Equal(shape, y.Dims) {
			return nil, fmt.Errorf("C of shape %v does not broadcast to %v", c.Dims, y.Dims)
		}
		s := broadcastStrides(c.Dims, y.Dims)
		for i := int64(0); i < m; i++ {
			for j := int64(0); j < nn; j++ {
				y.F[i*nn+j] = beta * c.F[i*s[0]+j*s[1]]
			}
		}
	}
	gemmKernel(int(m), int(k), int(nn), alpha, a.F, as0, as1, w.F, bs0, bs1, y.F)
	return []*Tensor{y}, nil
}
