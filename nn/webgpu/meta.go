// Package webgpu is a wasi-nn backend that runs ONNX graphs on the
// browser's GPU through WebGPU (WGSL compute kernels), for the "gpu"
// execution target. It runs where the engine's WASI host runs in the
// browser: the playground's engine worker.
//
// The graph is laid out by onnx.Session.Plan (shapes from a CPU dry run,
// constants folded), then each node is one kernel dispatch. A fallback
// (software) adapter is refused: GPU work is never claimed from a CPU.
package webgpu

import (
	"fmt"
	"math"

	"github.com/go-wasmcontainers/gpu/onnx"
)

// The meta buffers the kernels read, computed in Go (testable natively).

func contiguous(dims []int64) []int64 {
	s := make([]int64, len(dims))
	n := int64(1)
	for i := len(dims) - 1; i >= 0; i-- {
		s[i] = n
		n *= dims[i]
	}
	return s
}

// broadcastStrides are the strides of shape d read as shape out.
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

func prod(d []int64) int64 {
	n := int64(1)
	for _, x := range d {
		n *= x
	}
	return n
}

// meta is a builder of u32 words, refusing values past 32 bits.
type meta struct {
	w   []uint32
	err error
}

func (m *meta) u(v int64) {
	if v < 0 || v > math.MaxUint32 {
		m.err = fmt.Errorf("value %d does not fit the kernels' 32-bit indices", v)
	}
	m.w = append(m.w, uint32(v))
}

func (m *meta) f(v float32) { m.w = append(m.w, math.Float32bits(v)) }

// pad8 writes v padded to MaxRank entries with fill.
func (m *meta) pad8(v []int64, fill int64) {
	for i := 0; i < MaxRank; i++ {
		if i < len(v) {
			m.u(v[i])
		} else {
			m.u(fill)
		}
	}
}

func (m *meta) done() ([]uint32, error) { return m.w, m.err }

func binaryMeta(out, a, b []int64) ([]uint32, error) {
	if len(out) > MaxRank {
		return nil, fmt.Errorf("rank %d", len(out))
	}
	var m meta
	m.u(prod(out))
	m.u(int64(len(out)))
	m.pad8(out, 1)
	m.pad8(broadcastStrides(a, out), 0)
	m.pad8(broadcastStrides(b, out), 0)
	return m.done()
}

// matmulMeta lays out MatMul (numpy semantics) or Gemm; hasCIn says Gemm
// has a C (which may be a scalar, of no dimensions).
func matmulMeta(n *onnx.Node, a, b, c, out []int64, hasCIn bool) ([]uint32, error) {
	var m meta
	if n.Op == "Gemm" {
		M, N := out[0], out[1]
		K := a[1]
		as0, as1 := a[1], int64(1)
		if n.AttrInt("transA", 0) != 0 {
			K = a[0]
			as0, as1 = 1, a[1]
		}
		bs0, bs1 := b[1], int64(1)
		if n.AttrInt("transB", 0) != 0 {
			bs0, bs1 = 1, b[1]
		}
		var cs0, cs1, hasC int64
		beta := n.AttrFloat("beta", 1)
		if hasCIn && beta != 0 {
			s := broadcastStrides(c, out)
			cs0, cs1, hasC = s[0], s[1], 1
		}
		m.u(M * N)
		for _, v := range []int64{M, N, K, as0, as1, bs0, bs1, cs0, cs1, hasC} {
			m.u(v)
		}
		m.f(n.AttrFloat("alpha", 1))
		m.f(beta)
		m.u(0)
		m.pad8(nil, 1)
		m.pad8(nil, 0)
		m.pad8(nil, 0)
		return m.done()
	}
	ad, bd := a, b
	if len(ad) == 1 {
		ad = []int64{1, ad[0]}
	}
	if len(bd) == 1 {
		bd = []int64{bd[0], 1}
	}
	M, K := ad[len(ad)-2], ad[len(ad)-1]
	N := bd[len(bd)-1]
	batch := make([]int64, 0, MaxRank)
	// The output's batch dimensions: out minus its matrix part (which
	// MatMul drops for 1-D operands).
	matrix := 2
	if len(a) == 1 {
		matrix--
	}
	if len(b) == 1 {
		matrix--
	}
	batch = append(batch, out[:len(out)-matrix]...)
	if len(batch) > MaxRank {
		return nil, fmt.Errorf("batch rank %d", len(batch))
	}
	sa := broadcastStrides(ad[:len(ad)-2], batch)
	sb := broadcastStrides(bd[:len(bd)-2], batch)
	for i := range sa {
		sa[i] *= M * K
		sb[i] *= K * N
	}
	m.u(prod(batch) * M * N)
	for _, v := range []int64{M, N, K, K, 1, N, 1, 0, 0, 0} {
		m.u(v)
	}
	m.f(1)
	m.f(0)
	m.u(int64(len(batch)))
	m.pad8(batch, 1)
	m.pad8(sa, 0)
	m.pad8(sb, 0)
	return m.done()
}

// window3 writes a window's seven fields, each padded to three spatial
// axes at the front (in, kernel, stride, dil as 1; pads as 0; out as 1).
func window3(m *meta, w *onnx.Window) error {
	if len(w.In) > MaxSpatial {
		return fmt.Errorf("%d spatial axes (at most %d)", len(w.In), MaxSpatial)
	}
	field := func(v []int64, fill int64) {
		lead := MaxSpatial - len(v)
		for i := 0; i < MaxSpatial; i++ {
			if i < lead {
				m.u(fill)
			} else {
				m.u(v[i-lead])
			}
		}
	}
	field(w.In, 1)
	field(w.Kernel, 1)
	field(w.Stride, 1)
	field(w.Dil, 1)
	field(w.PadB, 0)
	field(w.PadE, 0)
	field(w.Out, 1)
	return nil
}

func convMeta(n *onnx.Node, x, wt, out []int64, hasBias bool) ([]uint32, error) {
	w, err := onnx.NodeWindow(n, x, wt)
	if err != nil {
		return nil, err
	}
	group := n.AttrInt("group", 1)
	var m meta
	m.u(prod(out))
	hb := int64(0)
	if hasBias {
		hb = 1
	}
	for _, v := range []int64{x[0], x[1], wt[0], wt[1], wt[0] / group, hb} {
		m.u(v)
	}
	if err := window3(&m, w); err != nil {
		return nil, err
	}
	return m.done()
}

func poolMeta(n *onnx.Node, x, out []int64) ([]uint32, error) {
	w, err := onnx.NodeWindow(n, x, nil)
	if err != nil {
		return nil, err
	}
	var m meta
	m.u(prod(out))
	m.u(x[0] * x[1])
	isMax, incl := int64(0), int64(0)
	if n.Op == "MaxPool" {
		isMax = 1
	}
	if n.AttrInt("count_include_pad", 0) != 0 {
		incl = 1
	}
	m.u(isMax)
	m.u(incl)
	if err := window3(&m, w); err != nil {
		return nil, err
	}
	return m.done()
}

// softmaxMeta: rows, len, inner, by the opset's semantics.
func softmaxMeta(n *onnx.Node, opset int64, x []int64) ([]uint32, int64, error) {
	var outer, length, inner int64
	if opset >= 13 {
		a, err := onnx.NormAxis(n.AttrInt("axis", -1), len(x), false)
		if err != nil {
			return nil, 0, err
		}
		outer, length, inner = prod(x[:a]), x[a], prod(x[a+1:])
	} else {
		a, err := onnx.NormAxis(n.AttrInt("axis", 1), len(x), true)
		if err != nil {
			return nil, 0, err
		}
		outer, length, inner = prod(x[:a]), prod(x[a:]), 1
	}
	var m meta
	rows := outer * inner
	m.u(rows)
	m.u(length)
	m.u(max(inner, 1))
	w, err := m.done()
	return w, rows, err
}

// copyMeta: an index space of shape, read at srcStride from srcOff and
// written at dstStride from dstOff.
func copyMeta(shape, src, dst []int64, srcOff, dstOff int64) ([]uint32, error) {
	if len(shape) > MaxRank {
		return nil, fmt.Errorf("rank %d", len(shape))
	}
	var m meta
	m.u(prod(shape))
	m.u(int64(len(shape)))
	m.pad8(shape, 1)
	m.pad8(src, 0)
	m.pad8(dst, 0)
	m.u(srcOff)
	m.u(dstOff)
	return m.done()
}

// transposeMeta copies x into the permuted output.
func transposeMeta(n *onnx.Node, x, out []int64) ([]uint32, error) {
	perm := n.AttrInts("perm")
	if perm == nil {
		perm = make([]int64, len(x))
		for i := range perm {
			perm[i] = int64(len(x) - 1 - i)
		}
	}
	in := contiguous(x)
	src := make([]int64, len(perm))
	for i, p := range perm {
		src[i] = in[p]
	}
	return copyMeta(out, src, contiguous(out), 0, 0)
}

// concatParts lays out one copy per input of a Concat.
func concatParts(n *onnx.Node, ins [][]int64, out []int64) ([][]uint32, error) {
	a, err := onnx.NormAxis(n.AttrInt("axis", 0), len(out), false)
	if err != nil {
		return nil, err
	}
	dst := contiguous(out)
	var parts [][]uint32
	off := int64(0)
	for _, s := range ins {
		w, err := copyMeta(s, contiguous(s), dst, 0, off*dst[a])
		if err != nil {
			return nil, err
		}
		parts = append(parts, w)
		off += s[a]
	}
	return parts, nil
}
