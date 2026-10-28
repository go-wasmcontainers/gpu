package webgpu

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/go-wasmcontainers/gpu/onnx"
)

// emu is a device that runs each kernel's arithmetic in Go, statement for
// statement as the WGSL does (u32 index math, the same loops), so the meta
// layouts the executor builds are checked natively against package onnx's
// CPU operators. The browser e2e then checks WGSL against the same tests.
type emu struct{ live int }

type ebuf struct{ f []float32 }

func (d *emu) upload(f []float32) (buf, error) {
	d.live++
	return &ebuf{f: slices.Clone(f)}, nil
}
func (d *emu) alloc(n int64) (buf, error) {
	d.live++
	return &ebuf{f: make([]float32, max(n, 1))}, nil
}
func (d *emu) read(b buf, n int64) ([]float32, error) { return slices.Clone(b.(*ebuf).f[:n]), nil }
func (d *emu) release(buf)                            { d.live-- }

func (d *emu) run(kernel string, n int64, prm []uint32, bufs ...buf) error {
	if _, ok := kernelSource(kernel); !ok {
		return fmt.Errorf("no kernel %q", kernel)
	}
	f := make([][]float32, len(bufs))
	for i, b := range bufs {
		f[i] = b.(*ebuf).f
	}
	for i := uint32(0); i < uint32(n); i++ {
		switch {
		case strings.HasPrefix(kernel, "binary:"):
			emuBinary(kernel[7:], prm, f, i)
		case kernel == kRelu:
			if i < prm[0] {
				f[1][i] = max(f[0][i], 0)
			}
		case kernel == kMatmul:
			emuMatmul(prm, f, i)
		case kernel == kConv:
			emuConv(prm, f, i)
		case kernel == kPool:
			emuPool(prm, f, i)
		case kernel == kSoftmax:
			emuSoftmax(prm, f, i)
		case kernel == kCopy:
			emuCopy(prm, f, i)
		}
	}
	return nil
}

func emuBinary(op string, prm []uint32, f [][]float32, i uint32) {
	if i >= prm[0] {
		return
	}
	rem, ia, ib := i, uint32(0), uint32(0)
	for d := int(prm[1]) - 1; d >= 0; d-- {
		dim := prm[2+d]
		c := rem % dim
		rem /= dim
		ia += c * prm[10+d]
		ib += c * prm[18+d]
	}
	p, q := f[0][ia], f[1][ib]
	var v float32
	switch op {
	case "Add":
		v = p + q
	case "Sub":
		v = p - q
	case "Mul":
		v = p * q
	case "Div":
		v = p / q
	}
	f[2][i] = v
}

func emuMatmul(prm []uint32, f [][]float32, i uint32) {
	if i >= prm[0] {
		return
	}
	M, N, K := prm[1], prm[2], prm[3]
	n, m := i%N, (i/N)%M
	rem, ab, bb := i/(M*N), uint32(0), uint32(0)
	for d := int(prm[13]) - 1; d >= 0; d-- {
		dim := prm[14+d]
		co := rem % dim
		rem /= dim
		ab += co * prm[22+d]
		bb += co * prm[30+d]
	}
	var acc float32
	for k := uint32(0); k < K; k++ {
		acc += f[0][ab+m*prm[4]+k*prm[5]] * f[1][bb+k*prm[6]+n*prm[7]]
	}
	v := math.Float32frombits(prm[11]) * acc
	if prm[10] != 0 {
		v += math.Float32frombits(prm[12]) * f[2][m*prm[8]+n*prm[9]]
	}
	f[3][i] = v
}

func emuConv(prm []uint32, f [][]float32, i uint32) {
	if i >= prm[0] {
		return
	}
	wm := func(fl, d uint32) uint32 { return prm[7+fl*3+d] }
	C, M, Cg, Mg := prm[2], prm[3], prm[4], prm[5]
	o2 := i % wm(6, 2)
	o1 := (i / wm(6, 2)) % wm(6, 1)
	o0 := (i / (wm(6, 2) * wm(6, 1))) % wm(6, 0)
	outPlane := wm(6, 0) * wm(6, 1) * wm(6, 2)
	m := (i / outPlane) % M
	ni := i / (outPlane * M)
	grp := m / Mg
	inPlane := wm(0, 0) * wm(0, 1) * wm(0, 2)
	kPlane := wm(1, 0) * wm(1, 1) * wm(1, 2)
	var acc float32
	if prm[6] != 0 {
		acc = f[2][m]
	}
	for c := uint32(0); c < Cg; c++ {
		xb := (ni*C + grp*Cg + c) * inPlane
		wb := (m*Cg + c) * kPlane
		for k0 := uint32(0); k0 < wm(1, 0); k0++ {
			p0 := int32(o0*wm(2, 0)+k0*wm(3, 0)) - int32(wm(4, 0))
			if p0 < 0 || p0 >= int32(wm(0, 0)) {
				continue
			}
			for k1 := uint32(0); k1 < wm(1, 1); k1++ {
				p1 := int32(o1*wm(2, 1)+k1*wm(3, 1)) - int32(wm(4, 1))
				if p1 < 0 || p1 >= int32(wm(0, 1)) {
					continue
				}
				for k2 := uint32(0); k2 < wm(1, 2); k2++ {
					p2 := int32(o2*wm(2, 2)+k2*wm(3, 2)) - int32(wm(4, 2))
					if p2 < 0 || p2 >= int32(wm(0, 2)) {
						continue
					}
					xi := xb + (uint32(p0)*wm(0, 1)+uint32(p1))*wm(0, 2) + uint32(p2)
					wi := wb + (k0*wm(1, 1)+k1)*wm(1, 2) + k2
					acc += f[0][xi] * f[1][wi]
				}
			}
		}
	}
	f[3][i] = acc
}

func emuPool(prm []uint32, f [][]float32, i uint32) {
	if i >= prm[0] {
		return
	}
	wm := func(fl, d uint32) uint32 { return prm[4+fl*3+d] }
	isMax, incl := prm[2] != 0, prm[3] != 0
	o2 := i % wm(6, 2)
	o1 := (i / wm(6, 2)) % wm(6, 1)
	o0 := (i / (wm(6, 2) * wm(6, 1))) % wm(6, 0)
	outPlane := wm(6, 0) * wm(6, 1) * wm(6, 2)
	plane := i / outPlane
	xb := plane * wm(0, 0) * wm(0, 1) * wm(0, 2)
	best := float32(math.Inf(-1))
	var sum, cnt, padCnt float32
	for k0 := uint32(0); k0 < wm(1, 0); k0++ {
		p0 := int32(o0*wm(2, 0)+k0*wm(3, 0)) - int32(wm(4, 0))
		in0 := p0 >= 0 && p0 < int32(wm(0, 0))
		pad0 := p0 < int32(wm(0, 0)+wm(5, 0))
		for k1 := uint32(0); k1 < wm(1, 1); k1++ {
			p1 := int32(o1*wm(2, 1)+k1*wm(3, 1)) - int32(wm(4, 1))
			in1 := p1 >= 0 && p1 < int32(wm(0, 1))
			pad1 := p1 < int32(wm(0, 1)+wm(5, 1))
			for k2 := uint32(0); k2 < wm(1, 2); k2++ {
				p2 := int32(o2*wm(2, 2)+k2*wm(3, 2)) - int32(wm(4, 2))
				in2 := p2 >= 0 && p2 < int32(wm(0, 2))
				pad2 := p2 < int32(wm(0, 2)+wm(5, 2))
				if pad0 && pad1 && pad2 {
					padCnt++
				}
				if !(in0 && in1 && in2) {
					continue
				}
				v := f[0][xb+(uint32(p0)*wm(0, 1)+uint32(p1))*wm(0, 2)+uint32(p2)]
				if v > best || v != v {
					best = v
				}
				sum += v
				cnt++
			}
		}
	}
	if isMax {
		f[1][i] = best
		return
	}
	d := cnt
	if incl {
		d = padCnt
	}
	if d > 0 {
		f[1][i] = sum / d
	} else {
		f[1][i] = 0
	}
}

func emuSoftmax(prm []uint32, f [][]float32, r uint32) {
	if r >= prm[0] {
		return
	}
	L, inner := prm[1], prm[2]
	base := (r/inner)*L*inner + r%inner
	mx := float32(math.Inf(-1))
	for j := uint32(0); j < L; j++ {
		mx = max(mx, f[0][base+j*inner])
	}
	var s float32
	for j := uint32(0); j < L; j++ {
		e := float32(math.Exp(float64(f[0][base+j*inner] - mx)))
		f[1][base+j*inner] = e
		s += e
	}
	for j := uint32(0); j < L; j++ {
		f[1][base+j*inner] /= s
	}
}

func emuCopy(prm []uint32, f [][]float32, i uint32) {
	if i >= prm[0] {
		return
	}
	rem, s, t := i, prm[26], prm[27]
	for d := int(prm[1]) - 1; d >= 0; d-- {
		dim := prm[2+d]
		c := rem % dim
		rem /= dim
		s += c * prm[10+d]
		t += c * prm[18+d]
	}
	f[1][t] = f[0][s]
}

func readT(t *testing.T, p string) *onnx.Tensor {
	t.Helper()
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	x, err := onnx.DecodeTensor(b, onnx.DefaultLimits)
	if err != nil {
		t.Fatal(err)
	}
	return x
}

// runEmu runs one node test on the emulated device; skip reports a test
// the GPU path refuses (errUnsupported), which is listed, not hidden.
func runEmu(t *testing.T, dir string) (skip string) {
	b, err := os.ReadFile(filepath.Join(dir, "model.onnx"))
	if err != nil {
		t.Fatal(err)
	}
	s, err := onnx.Load(b, onnx.DefaultLimits)
	if err != nil {
		t.Fatal(err)
	}
	set := filepath.Join(dir, "test_data_set_0")
	in := map[string]*onnx.Tensor{}
	for i, vi := range s.Inputs() {
		in[vi.Name] = readT(t, filepath.Join(set, fmt.Sprintf("input_%d.pb", i)))
	}
	p, err := s.Plan(in)
	if err != nil {
		t.Fatal(err)
	}
	dev := &emu{}
	e, err := newExecutor(dev, p)
	if err != nil {
		t.Fatal(err)
	}
	floats := map[string]*onnx.Tensor{}
	for k, v := range in {
		if v.Type == onnx.Float {
			floats[k] = v
		}
	}
	var names []string
	for _, o := range s.Outputs() {
		names = append(names, o.Name)
	}
	got, shapes, err := e.run(floats, names)
	if err != nil {
		if strings.Contains(err.Error(), errUnsupported.Error()) {
			return err.Error()
		}
		t.Fatal(err)
	}
	for i, o := range s.Outputs() {
		want := readT(t, filepath.Join(set, fmt.Sprintf("output_%d.pb", i)))
		if !slices.Equal(shapes[o.Name], want.Dims) {
			t.Fatalf("output %q shape %v, want %v", o.Name, shapes[o.Name], want.Dims)
		}
		for j, w := range want.F {
			g := got[o.Name][j]
			if math.IsNaN(float64(w)) && math.IsNaN(float64(g)) {
				continue
			}
			if math.Abs(float64(g-w)) > 1e-7+1e-3*math.Abs(float64(w)) {
				t.Fatalf("output %q[%d] = %g, want %g", o.Name, j, g, w)
			}
		}
	}
	if dev.live != len(e.consts) {
		t.Errorf("%d buffers alive after the run, want the %d constants", dev.live, len(e.consts))
	}
	return ""
}

// TestKernelsEmulated runs every vendored ONNX node test through the
// executor on the emulated device. Tests the GPU path refuses (int64
// outputs) are listed and must stay few.
func TestKernelsEmulated(t *testing.T) {
	dirs, _ := filepath.Glob("../../onnx/testdata/node/test_*")
	if len(dirs) < 100 {
		t.Fatalf("%d node tests", len(dirs))
	}
	var skipped []string
	for _, dir := range dirs {
		t.Run(filepath.Base(dir), func(t *testing.T) {
			if s := runEmu(t, dir); s != "" {
				skipped = append(skipped, filepath.Base(dir)+": "+s)
			}
		})
	}
	t.Logf("%d tests, %d refused by the GPU path:\n%s", len(dirs), len(skipped), strings.Join(skipped, "\n"))
	if len(skipped) > 10 {
		t.Errorf("%d tests refused by the GPU path", len(skipped))
	}
}

func TestSupportedMatchesCPU(t *testing.T) {
	for op := range onnx.Supported {
		if !supportedOp(op) {
			t.Errorf("%s has no GPU path", op)
		}
	}
	if len(Supported) != len(onnx.Supported) {
		t.Errorf("GPU %d ops, CPU %d", len(Supported), len(onnx.Supported))
	}
}
