//go:build js

package webgpu

import (
	"errors"
	"fmt"
	"os"
	"slices"
	"strings"

	wg "github.com/go-gfx/webgpu"
	"github.com/go-gfx/webgpu/batch"
	"github.com/go-wasmcontainers/gpu/nn"
	"github.com/go-wasmcontainers/gpu/onnx"
)

// Backend is the wasi-nn "gpu" backend for ONNX graphs. Its calls block
// on WebGPU promises, so nn runs them off the host call (Async).
type Backend struct {
	Limits onnx.Limits
	// MemoryBytes is each graph's GPU memory quota (default 256 MiB).
	MemoryBytes int64
	// AllowFallback accepts a software adapter; off by default, since its
	// work is not GPU work.
	AllowFallback bool
	// NoLoopBudget turns off the loop budget (go-gfx/webgpu batch.LoopBudget,
	// 2^36 work): by default every kernel is rewritten so that its loops
	// stop, whatever a model's shapes ask for, and a cut kernel fails the run.
	NoLoopBudget bool
	// LoopWork and LoopMin set the loop budget's work (default 2^36) and
	// its floor (default 1024); for tests of the cut.
	LoopWork uint64
	LoopMin  uint32
}

// NewModule is the wasi-nn host module for the browser: the pure-Go CPU
// backend for the "cpu" target and this backend for "gpu" (with the
// container's gpu capability).
func NewModule() *nn.Module {
	m := nn.NewModule()
	m.Backends = append(m.Backends, Backend{Limits: onnx.DefaultLimits})
	return m
}

// Encoding implements nn.Backend.
func (Backend) Encoding() nn.Encoding { return nn.ONNX }

// Runs implements nn.Backend.
func (Backend) Runs(t nn.Target) bool { return t == nn.GPU }

// Async implements nn.AsyncBackend.
func (Backend) Async() bool { return true }

// Load implements nn.Backend: the model checked and laid out on a device
// of its own, destroyed with the graph.
func (b Backend) Load(builders [][]byte, _ nn.Target) (nn.Graph, nn.Errno) {
	if len(builders) != 1 {
		return nil, nn.InvalidArgument
	}
	s, err := onnx.Load(builders[0], b.Limits)
	if err != nil {
		var u *onnx.UnsupportedError
		if errors.As(err, &u) {
			return nil, nn.UnsupportedOperation
		}
		return nil, nn.InvalidEncoding
	}
	mem := b.MemoryBytes
	if mem <= 0 {
		mem = 256 << 20
	}
	var lb *batch.LoopBudget
	if !b.NoLoopBudget {
		w := b.LoopWork
		if w == 0 {
			w = 1 << 36
		}
		lb = &batch.LoopBudget{Work: w, MinIterations: b.LoopMin, Group: 3}
	}
	d, err := newGPUDevice(b.AllowFallback, mem, lb)
	if err != nil {
		// The errno cannot carry why; stderr reaches the worker's console.
		fmt.Fprintf(os.Stderr, "nn/webgpu: no GPU device: %v\n", err)
		return nil, nn.UnsupportedOperation // no GPU here: never the CPU instead
	}
	return &graph{s: s, dev: d}, nn.Success
}

type graph struct {
	s   *onnx.Session
	dev *gpuDevice
}

// Info is the adapter the graph runs on (for logs and tests).
func (g *graph) Info() wg.AdapterInfo { return g.dev.info }

// Close implements nn's optional closer: the device and all it holds go.
func (g *graph) Close() { g.dev.d.Destroy() }

func (g *graph) NewContext() (nn.Context, nn.Errno) {
	return &context{g: g, in: map[string]*onnx.Tensor{}}, nn.Success
}

type context struct {
	g    *graph
	in   map[string]*onnx.Tensor
	out  map[string]*onnx.Tensor
	key  string
	exec *executor
	gd   *gpuDevice
}

func (c *context) SetInput(i uint32, t *onnx.Tensor) nn.Errno {
	ins := c.g.s.Inputs()
	if int(i) >= len(ins) {
		return nn.InvalidArgument
	}
	c.in[ins[i].Name] = t
	return nn.Success
}

// planKey identifies a plan: float shapes and int64 values of the inputs.
func planKey(in map[string]*onnx.Tensor) string {
	names := make([]string, 0, len(in))
	for k := range in {
		names = append(names, k)
	}
	slices.Sort(names)
	var b strings.Builder
	for _, k := range names {
		t := in[k]
		fmt.Fprintf(&b, "%s%v", k, t.Dims)
		if t.Type != onnx.Float {
			fmt.Fprintf(&b, "=%v", t.I)
		}
		b.WriteByte(';')
	}
	return b.String()
}

func (c *context) Compute() nn.Errno {
	if c.g.dev.lost() {
		return nn.RuntimeError
	}
	key := planKey(c.in)
	if c.exec == nil || key != c.key {
		p, err := c.g.s.Plan(c.in)
		if err != nil {
			return nn.InvalidArgument
		}
		if c.gd == nil {
			c.gd = c.g.dev
		}
		if c.exec != nil {
			for _, k := range c.exec.consts {
				c.gd.release(k.b)
			}
		}
		e, err := newExecutor(c.gd, p)
		if err != nil {
			return nn.TooLarge
		}
		c.exec, c.key = e, key
	}
	floats := map[string]*onnx.Tensor{}
	for k, v := range c.in {
		if v.Type == onnx.Float {
			floats[k] = v
		}
	}
	var names []string
	for _, o := range c.g.s.Outputs() {
		names = append(names, o.Name)
	}
	got, shapes, err := c.exec.run(floats, names)
	if err != nil {
		// The errno cannot carry why; stderr reaches the worker's console.
		fmt.Fprintf(os.Stderr, "nn/webgpu: compute failed: %v\n", err)
		if errors.Is(err, errUnsupported) {
			return nn.UnsupportedOperation
		}
		return nn.RuntimeError
	}
	c.out = map[string]*onnx.Tensor{}
	for _, n := range names {
		c.out[n] = &onnx.Tensor{Type: onnx.Float, Dims: shapes[n], F: got[n]}
	}
	return nn.Success
}

func (c *context) Output(i uint32) (*onnx.Tensor, nn.Errno) {
	outs := c.g.s.Outputs()
	if c.out == nil || int(i) >= len(outs) {
		return nil, nn.InvalidArgument
	}
	return c.out[outs[i].Name], nn.Success
}
