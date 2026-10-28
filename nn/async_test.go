package nn

import (
	"encoding/binary"
	"os"
	"testing"

	"github.com/go-wasmcontainers/engine/rootfs"
	"github.com/go-wasmcontainers/engine/wasi"
	"github.com/go-wasmcontainers/gpu/onnx"
)

// asyncGPU is a "gpu" backend running on the CPU that declares itself
// asynchronous, to drive nn's off-the-call path natively. It exists only
// in tests: a real GPU backend never runs on the CPU.
type asyncGPU struct {
	CPUBackend
	closed *int
}

func (asyncGPU) Runs(t Target) bool { return t == GPU }
func (asyncGPU) Async() bool        { return true }
func (a asyncGPU) Load(b [][]byte, _ Target) (Graph, Errno) {
	g, e := a.CPUBackend.Load(b, CPU)
	if e != Success {
		return nil, e
	}
	return closingGraph{g, a.closed}, e
}

type closingGraph struct {
	Graph
	closed *int
}

func (c closingGraph) Close() { *c.closed++ }

func resume(t *testing.T, res wasi.Result, m wasi.Memory) wasi.Result {
	t.Helper()
	for res.Wait != nil {
		<-res.Wait
		res = res.Resume(m)
	}
	return res
}

// TestAsyncBackend: load and compute of an asynchronous backend return a
// Wait and finish on Resume; a graph is closed with the instance.
func TestAsyncBackend(t *testing.T) {
	model, err := os.ReadFile("../onnx/testdata/models/mnist-12/mnist-12.onnx")
	if err != nil {
		t.Fatal(err)
	}
	closed := 0
	mod := &Module{Backends: []Backend{CPUBackend{Limits: onnx.DefaultLimits}, asyncGPU{CPUBackend{Limits: onnx.DefaultLimits}, &closed}}, Limits: DefaultLimits}
	in := mod.Open(wasi.New(wasi.Config{FS: rootfs.New(), Caps: []string{"gpu"}}))
	m := make(mem, 1<<20)
	copy(m[100000:], model)
	le(m, 0, 100000)
	le(m, 4, uint32(len(model)))
	res := in.Call(0, []uint64{0, 1, uint64(ONNX), uint64(GPU), 16}, m)
	if res.Wait == nil {
		t.Fatal("an async load did not wait")
	}
	if res = resume(t, res, m); res.Ret != uint64(Success) {
		t.Fatalf("load: %+v", res)
	}
	g := binary.LittleEndian.Uint32(m[16:])
	if r := in.Call(2, []uint64{uint64(g), 20}, m); r.Ret != uint64(Success) {
		t.Fatal(r)
	}
	ctx := binary.LittleEndian.Uint32(m[20:])
	le(m, 500, 600)
	le(m, 504, 4)
	for i, d := range []uint32{1, 1, 28, 28} {
		le(m, 600+uint32(4*i), d)
	}
	m[508] = TypeF32
	le(m, 512, 700)
	le(m, 516, 28*28*4)
	if r := in.Call(3, []uint64{uint64(ctx), 0, 500}, m); r.Ret != uint64(Success) {
		t.Fatal(r)
	}
	res = in.Call(5, []uint64{uint64(ctx)}, m)
	if res.Wait == nil {
		t.Fatal("an async compute did not wait")
	}
	if res = resume(t, res, m); res.Ret != uint64(Success) {
		t.Fatalf("compute: %+v", res)
	}
	if r := in.Call(4, []uint64{uint64(ctx), 0, 300, 40, 24}, m); r.Ret != uint64(Success) {
		t.Fatal(r)
	}
	// A cpu target still goes to the CPU backend, synchronously.
	if r := in.Call(0, []uint64{0, 1, uint64(ONNX), uint64(CPU), 16}, m); r.Wait != nil || r.Ret != uint64(Success) {
		t.Fatalf("cpu load: %+v", r)
	}
	in.Close()
	if closed != 1 {
		t.Errorf("graph closed %d times", closed)
	}
	// A load still running when the process ends closes its graph.
	in2 := mod.Open(wasi.New(wasi.Config{FS: rootfs.New(), Caps: []string{"gpu"}}))
	res = in2.Call(0, []uint64{0, 1, uint64(ONNX), uint64(GPU), 16}, m)
	in2.Close()
	if res = resume(t, res, m); res.Ret != uint64(RuntimeError) || closed != 2 {
		t.Errorf("load after close: %+v, closed %d", res, closed)
	}
}
