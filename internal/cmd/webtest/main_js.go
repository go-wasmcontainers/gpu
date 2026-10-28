//go:build js

// Command webtest runs in a browser Web Worker (as the playground's engine
// worker does) and checks the WebGPU wasi-nn backend against the ONNX node
// tests and MNIST-12: for each case it fetches the model and its test data
// from the page's origin, runs it through webgpu.Backend, compares with
// the reference (rtol 1e-3, atol 1e-7, the ONNX backend's tolerances) and
// posts a JSON report: the adapter, and every case's result.
//
// e2e/webgpu_test.go builds it, serves it and drives it with Playwright.
package main

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"slices"
	"syscall/js"
	"time"

	"github.com/go-wasmcontainers/gpu/internal/jsgpu"
	"github.com/go-wasmcontainers/gpu/nn"
	"github.com/go-wasmcontainers/gpu/nn/webgpu"
	"github.com/go-wasmcontainers/gpu/onnx"
)

// Case is one test: a model and its data set, by URL.
type Case struct {
	Name    string   `json:"name"`
	Model   string   `json:"model"`
	Inputs  []string `json:"inputs"`
	Outputs []string `json:"outputs"`
}

// Result is one case's outcome.
type Result struct {
	Name   string  `json:"name"`
	OK     bool    `json:"ok"`
	Error  string  `json:"error,omitempty"`
	MaxErr float64 `json:"maxErr"`
	Ms     float64 `json:"ms"`
}

// Report is what the worker posts.
type Report struct {
	Adapter jsgpu.AdapterInfo `json:"adapter"`
	Results []Result          `json:"results"`
	Error   string            `json:"error,omitempty"`
}

func fetch(url string) ([]byte, error) {
	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("%s: %s", url, resp.Status)
	}
	return io.ReadAll(io.LimitReader(resp.Body, 64<<20))
}

func tensor(url string) (*onnx.Tensor, error) {
	b, err := fetch(url)
	if err != nil {
		return nil, err
	}
	return onnx.DecodeTensor(b, onnx.DefaultLimits)
}

func runCase(be webgpu.Backend, c Case) Result {
	res := Result{Name: c.Name}
	fail := func(err error) Result { res.Error = err.Error(); return res }
	model, err := fetch(c.Model)
	if err != nil {
		return fail(err)
	}
	g, e := be.Load([][]byte{model}, nn.GPU)
	if e != nn.Success {
		return fail(fmt.Errorf("load: nn errno %d", e))
	}
	defer g.(interface{ Close() }).Close()
	ctx, _ := g.NewContext()
	for i, u := range c.Inputs {
		t, err := tensor(u)
		if err != nil {
			return fail(err)
		}
		if e := ctx.SetInput(uint32(i), t); e != nn.Success {
			return fail(fmt.Errorf("input %d: nn errno %d", i, e))
		}
	}
	t0 := time.Now()
	if e := ctx.Compute(); e != nn.Success {
		return fail(fmt.Errorf("compute: nn errno %d", e))
	}
	res.Ms = float64(time.Since(t0).Microseconds()) / 1000
	for i, u := range c.Outputs {
		want, err := tensor(u)
		if err != nil {
			return fail(err)
		}
		got, e := ctx.Output(uint32(i))
		if e != nn.Success {
			return fail(fmt.Errorf("output %d: nn errno %d", i, e))
		}
		if !slices.Equal(got.Dims, want.Dims) {
			return fail(fmt.Errorf("output %d: shape %v, want %v", i, got.Dims, want.Dims))
		}
		for j, w := range want.F {
			gv := got.F[j]
			if math.IsNaN(float64(w)) && math.IsNaN(float64(gv)) {
				continue
			}
			d := math.Abs(float64(gv - w))
			res.MaxErr = max(res.MaxErr, d)
			if d > 1e-7+1e-3*math.Abs(float64(w)) || math.IsNaN(float64(gv)) != math.IsNaN(float64(w)) {
				return fail(fmt.Errorf("output %d[%d] = %g, want %g", i, j, gv, w))
			}
		}
	}
	res.OK = true
	return res
}

func main() {
	var rep Report
	post := func() {
		b, _ := json.Marshal(rep)
		js.Global().Call("postMessage", string(b))
	}
	b, err := fetch("cases.json")
	if err != nil {
		rep.Error = err.Error()
		post()
		return
	}
	var cases []Case
	if err := json.Unmarshal(b, &cases); err != nil {
		rep.Error = err.Error()
		post()
		return
	}
	allow := js.Global().Get("location").Get("search").String() == "?fallback=1"
	d, err := jsgpu.NewDevice(true) // report the adapter whatever it is
	if err != nil {
		rep.Error = "no WebGPU adapter: " + err.Error()
		post()
		return
	}
	rep.Adapter = d.Info
	d.Destroy()
	be := webgpu.Backend{Limits: onnx.DefaultLimits, AllowFallback: allow}
	for _, c := range cases {
		rep.Results = append(rep.Results, runCase(be, c))
	}
	post()
}
