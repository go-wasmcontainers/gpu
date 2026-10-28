package onnx

import (
	"fmt"
	"math"
)

// window is the geometry of a convolution or pooling over the spatial
// dimensions (those after N and C).
type window struct {
	in, kernel, stride, dil, padB, padE, out []int64
}

// geometry reads kernel_shape, strides, dilations, pads and auto_pad, and
// computes the output size per spatial axis (docs/Operators.md, Conv and
// MaxPool/AveragePool).
func geometry(n *Node, in, kernel []int64, ceil bool) (*window, error) {
	r := len(in)
	w := &window{in: in, kernel: kernel}
	ones := func(name string) ([]int64, error) {
		v := attrInts(n, name)
		if v == nil {
			v = make([]int64, r)
			for i := range v {
				v[i] = 1
			}
		}
		if len(v) != r {
			return nil, fmt.Errorf("%s has %d values for %d spatial axes", name, len(v), r)
		}
		for _, x := range v {
			if x < 1 || x > 1<<16 {
				return nil, fmt.Errorf("%s value %d out of range", name, x)
			}
		}
		return v, nil
	}
	var err error
	if w.stride, err = ones("strides"); err != nil {
		return nil, err
	}
	if w.dil, err = ones("dilations"); err != nil {
		return nil, err
	}
	for _, k := range kernel {
		if k < 1 || k > 1<<16 {
			return nil, fmt.Errorf("kernel size %d out of range", k)
		}
	}
	w.padB, w.padE, w.out = make([]int64, r), make([]int64, r), make([]int64, r)
	pads := attrInts(n, "pads")
	autoPad := attrString(n, "auto_pad", "NOTSET")
	switch autoPad {
	case "NOTSET", "":
		if pads != nil {
			if len(pads) != 2*r {
				return nil, fmt.Errorf("pads has %d values for %d spatial axes", len(pads), r)
			}
			for i := 0; i < r; i++ {
				w.padB[i], w.padE[i] = pads[i], pads[i+r]
				if w.padB[i] < 0 || w.padE[i] < 0 || w.padB[i] > 1<<16 || w.padE[i] > 1<<16 {
					return nil, fmt.Errorf("pads %v out of range", pads)
				}
			}
		}
	case "VALID":
	case "SAME_UPPER", "SAME_LOWER":
		for i := 0; i < r; i++ {
			out := (in[i] + w.stride[i] - 1) / w.stride[i]
			total := max(0, (out-1)*w.stride[i]+(kernel[i]-1)*w.dil[i]+1-in[i])
			if autoPad == "SAME_UPPER" {
				w.padB[i] = total / 2
			} else {
				w.padB[i] = (total + 1) / 2
			}
			w.padE[i] = total - w.padB[i]
		}
	default:
		return nil, fmt.Errorf("auto_pad %q", autoPad)
	}
	for i := 0; i < r; i++ {
		span := (kernel[i]-1)*w.dil[i] + 1
		padded := in[i] + w.padB[i] + w.padE[i]
		if padded < span {
			return nil, fmt.Errorf("kernel %v larger than the padded input %v", kernel, in)
		}
		if ceil {
			w.out[i] = (padded-span+w.stride[i]-1)/w.stride[i] + 1
			// A window that would start in the end padding is dropped.
			if (w.out[i]-1)*w.stride[i] >= in[i]+w.padB[i] {
				w.out[i]--
			}
		} else {
			w.out[i] = (padded-span)/w.stride[i] + 1
		}
	}
	return w, nil
}

// spatialIter walks every output position and, for each, every kernel
// position, calling visit with the flat output index, the flat input
// offset (or -1 when the tap falls in the padding) and the flat kernel
// index. It keeps the coordinates in small arrays so N-D costs no more
// bookkeeping per tap than 2-D.
func spatialIter(w *window, visit func(o, in, k int64)) {
	r := len(w.in)
	inStride := make([]int64, r)
	s := int64(1)
	for i := r - 1; i >= 0; i-- {
		inStride[i] = s
		s *= w.in[i]
	}
	oc := make([]int64, r)
	kc := make([]int64, r)
	nOut, nK := prod(w.out), prod(w.kernel)
	for o := int64(0); o < nOut; o++ {
		for i := range kc {
			kc[i] = 0
		}
		for k := int64(0); k < nK; k++ {
			off := int64(0)
			for i := 0; i < r; i++ {
				c := oc[i]*w.stride[i] - w.padB[i] + kc[i]*w.dil[i]
				if c < 0 || c >= w.in[i] {
					off = -1
					break
				}
				off += c * inStride[i]
			}
			visit(o, off, k)
			for i := r - 1; i >= 0; i-- {
				kc[i]++
				if kc[i] < w.kernel[i] {
					break
				}
				kc[i] = 0
			}
		}
		for i := r - 1; i >= 0; i-- {
			oc[i]++
			if oc[i] < w.out[i] {
				break
			}
			oc[i] = 0
		}
	}
}

// taps precomputes spatialIter's (output, input offset, kernel) triples
// for one channel plane, so the channel loops reuse them.
type tap struct{ o, in, k int32 }

func taps(w *window, lim int64) ([]tap, error) {
	no, nk := prod(w.out), prod(w.kernel)
	if no > 0 && nk > lim/no || prod(w.in) > math.MaxInt32 || no > math.MaxInt32 {
		return nil, fmt.Errorf("window of %d×%d taps over %v exceeds the limits", no, nk, w.in)
	}
	n := no * nk
	t := make([]tap, 0, n)
	spatialIter(w, func(o, in, k int64) {
		t = append(t, tap{int32(o), int32(in), int32(k)})
	})
	return t, nil
}

func conv(n *Node, args []*Tensor, b *budget) ([]*Tensor, error) {
	x, err := argF(args, 0)
	if err != nil {
		return nil, err
	}
	wt, err := argF(args, 1)
	if err != nil {
		return nil, err
	}
	if len(x.Dims) < 3 || len(wt.Dims) != len(x.Dims) {
		return nil, fmt.Errorf("Conv wants X (N,C,...) and W of the same rank, got %v and %v", x.Dims, wt.Dims)
	}
	group := attrInt(n, "group", 1)
	N, C, M, Cg := x.Dims[0], x.Dims[1], wt.Dims[0], wt.Dims[1]
	if group < 1 || C != Cg*group || M%group != 0 {
		return nil, fmt.Errorf("channels: X %v, W %v, group %d", x.Dims, wt.Dims, group)
	}
	kernel := wt.Dims[2:]
	if ks := attrInts(n, "kernel_shape"); ks != nil && !equal(ks, kernel) {
		return nil, fmt.Errorf("kernel_shape %v differs from W %v", ks, wt.Dims)
	}
	w, err := geometry(n, x.Dims[2:], kernel, false)
	if err != nil {
		return nil, err
	}
	var bias []float32
	if len(args) > 2 && args[2] != nil {
		bt, err := argF(args, 2)
		if err != nil {
			return nil, err
		}
		if len(bt.F) != int(M) {
			return nil, fmt.Errorf("bias of %d values for %d feature maps", len(bt.F), M)
		}
		bias = bt.F
	}
	y, err := b.alloc(append([]int64{N, M}, w.out...))
	if err != nil {
		return nil, err
	}
	nOut, nK, nIn := prod(w.out), prod(kernel), prod(w.in)
	if err := b.spend(N * M * nOut * Cg * nK); err != nil {
		return nil, err
	}
	tp, err := taps(w, b.lim.TensorElems)
	if err != nil {
		return nil, err
	}
	mg := M / group
	for ni := int64(0); ni < N; ni++ {
		for m := int64(0); m < M; m++ {
			g := m / mg
			out := y.F[(ni*M+m)*nOut : (ni*M+m+1)*nOut]
			if bias != nil {
				for i := range out {
					out[i] = bias[m]
				}
			}
			for c := int64(0); c < Cg; c++ {
				plane := x.F[(ni*C+g*Cg+c)*nIn : (ni*C+g*Cg+c+1)*nIn]
				kern := wt.F[(m*Cg+c)*nK : (m*Cg+c+1)*nK]
				for _, t := range tp {
					if t.in >= 0 {
						out[t.o] += plane[t.in] * kern[t.k]
					}
				}
			}
		}
	}
	return []*Tensor{y}, nil
}

func equal(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func pool(n *Node, args []*Tensor, b *budget) ([]*Tensor, error) {
	x, err := argF(args, 0)
	if err != nil {
		return nil, err
	}
	if len(x.Dims) < 3 {
		return nil, fmt.Errorf("%s wants (N,C,...), got %v", n.Op, x.Dims)
	}
	kernel := attrInts(n, "kernel_shape")
	if len(kernel) != len(x.Dims)-2 {
		return nil, fmt.Errorf("kernel_shape %v for input %v", kernel, x.Dims)
	}
	if attrInt(n, "storage_order", 0) != 0 {
		return nil, fmt.Errorf("storage_order 1 is not supported")
	}
	w, err := geometry(n, x.Dims[2:], kernel, attrInt(n, "ceil_mode", 0) != 0)
	if err != nil {
		return nil, err
	}
	N, C := x.Dims[0], x.Dims[1]
	y, err := b.alloc(append([]int64{N, C}, w.out...))
	if err != nil {
		return nil, err
	}
	nOut, nK, nIn := prod(w.out), prod(kernel), prod(w.in)
	if err := b.spend(N * C * nOut * nK); err != nil {
		return nil, err
	}
	tp, err := taps(w, b.lim.TensorElems)
	if err != nil {
		return nil, err
	}
	isMax := n.Op == "MaxPool"
	includePad := attrInt(n, "count_include_pad", 0) != 0
	// With count_include_pad, the divisor counts the taps inside the
	// padded input (explicit pads), not those past it (ceil_mode overhang).
	var padDiv []float32
	if !isMax && includePad {
		padDiv = make([]float32, nOut)
		pw := *w
		pw.in = make([]int64, len(w.in))
		for i := range w.in {
			pw.in[i] = w.in[i] + w.padB[i] + w.padE[i]
		}
		pw.padB = make([]int64, len(w.in))
		spatialIter(&pw, func(o, in, _ int64) {
			if in >= 0 {
				padDiv[o]++
			}
		})
	}
	cnt := make([]float32, nOut)
	for p := int64(0); p < N*C; p++ {
		plane := x.F[p*nIn : (p+1)*nIn]
		out := y.F[p*nOut : (p+1)*nOut]
		if isMax {
			for i := range out {
				out[i] = float32(math.Inf(-1))
			}
		} else {
			clear(cnt)
		}
		for _, t := range tp {
			if t.in < 0 {
				continue
			}
			v := plane[t.in]
			if isMax {
				if v > out[t.o] || v != v {
					out[t.o] = v
				}
			} else {
				out[t.o] += v
				cnt[t.o]++
			}
		}
		if !isMax {
			for i := range out {
				d := cnt[i]
				if padDiv != nil {
					d = padDiv[i]
				}
				if d > 0 {
					out[i] /= d
				}
			}
		}
	}
	return []*Tensor{y}, nil
}
