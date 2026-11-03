//go:build js

// Command enginetest is a minimal engine worker, as the playground's: the
// engine's browser executor (one Web Worker per container, the SAB bridge)
// with this module's host modules, wasi:webgpu and wasi-nn (CPU and
// WebGPU backends). It runs the containers listed in cases.json (a module
// URL, its arguments, whether it gets the gpu capability) and posts a JSON
// report of each one's output and exit code, and the adapter it ran on.
//
// e2e/engine_test.go builds it, serves it with the isolation headers
// shared memory needs, and drives it with Playwright.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"sort"
	"strconv"
	"sync"
	"syscall/js"
	"time"

	wg "github.com/go-gfx/webgpu"
	"github.com/go-wasmcontainers/engine/models"
	"github.com/go-wasmcontainers/engine/rootfs"
	"github.com/go-wasmcontainers/engine/runtime"
	"github.com/go-wasmcontainers/engine/runtime/browser"
	"github.com/go-wasmcontainers/engine/wasi"
	"github.com/go-wasmcontainers/gpu/gpubatch"
	"github.com/go-wasmcontainers/gpu/nn"
	nnwebgpu "github.com/go-wasmcontainers/gpu/nn/webgpu"
	"github.com/go-wasmcontainers/gpu/onnx"
	"github.com/go-wasmcontainers/gpu/webgpu"
)

// Case is one container run.
type Case struct {
	Name   string   `json:"name"`
	Module string   `json:"module"`
	Args   []string `json:"args"`
	GPU    bool     `json:"gpu"`
	// GPUMemory is run --gpu-memory in bytes (0: none, the default quota).
	GPUMemory int64 `json:"gpuMemory"`
	// Models are mounted read-only at /models (the engine's model mount,
	// models.Mount): each is fetched by ranges from URL (same origin), its
	// size and sha256 verified before the container starts.
	Models []ModelRef `json:"models"`
	// Parallel starts the container without waiting for it to end before the
	// next case starts; DelayMs waits that long before starting it.
	Parallel bool `json:"parallel"`
	DelayMs  int  `json:"delayMs"`
}

// ModelRef is a model of a case.
type ModelRef struct {
	Name   string `json:"name"`
	URL    string `json:"url"`
	Size   int64  `json:"size"`
	SHA256 string `json:"sha256"`
}

// Result is how it ended.
type Result struct {
	Name   string  `json:"name"`
	Exit   int     `json:"exit"`
	Stdout string  `json:"stdout"`
	Stderr string  `json:"stderr"`
	Error  string  `json:"error,omitempty"`
	Ms     float64 `json:"ms"`
	// Events are what the container's host modules reported (the engine's
	// gpu-sanction event, as "module/action k=v ...").
	Events []string `json:"events,omitempty"`
}

// Report is what the worker posts.
type Report struct {
	Adapter wg.AdapterInfo `json:"adapter"`
	// AdapterError is why no adapter was obtained (no navigator.gpu, or
	// requestAdapter refused), empty when Adapter is set.
	AdapterError string   `json:"adapterError,omitempty"`
	Results      []Result `json:"results"`
	Error        string   `json:"error,omitempty"`
	// ConfiguredSurfaces is how many surfaces still hold a WebGPU context
	// once every container has ended: they all dropped theirs, so 0.
	ConfiguredSurfaces int64 `json:"configuredSurfaces"`
	// GPU is what the wasi_webgpu_batch module measured: submits, sanctions,
	// the longest submit within budget.
	GPU gpubatch.Stats `json:"gpu"`
}

type buf struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (w *buf) Write(p []byte) (int, error) { w.mu.Lock(); defer w.mu.Unlock(); return w.b.Write(p) }
func (w *buf) String() string              { w.mu.Lock(); defer w.mu.Unlock(); return w.b.String() }

func fetch(url string) ([]byte, error) {
	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	return io.ReadAll(io.LimitReader(resp.Body, 64<<20))
}

func main() {
	// Never exit: the page keeps sending frame acknowledgements and input
	// after the report, and wasm_exec.js throws "Go program has already
	// exited" for any js.FuncOf callback once main has returned (Go's
	// _makeFuncWrapper calls _resume, which throws when exited). That
	// race made the e2e flaky (the error replaced the report); the
	// playground's engine worker lives as long as its page, and so does
	// this one.
	defer func() { select {} }()
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
	// Query: fallback=1 lets the host accept a software adapter; budget and
	// gputime (ms) set the module's steady-state budget and hard cap.
	params := js.Global().Get("URLSearchParams").New(js.Global().Get("location").Get("search").String())
	param := func(k string) int {
		if v := params.Call("get", k); v.Type() == js.TypeString {
			n, _ := strconv.Atoi(v.String())
			return n
		}
		return 0
	}
	allow := param("fallback") == 1
	if g, err := wg.Navigator(); err != nil {
		rep.AdapterError = "navigator.gpu: " + err.Error()
	} else if a, err := g.RequestAdapter(context.Background(), &wg.RequestAdapterOptions{AllowFallback: true}); err != nil {
		rep.AdapterError = "requestAdapter: " + err.Error()
	} else {
		rep.Adapter = a.Info()
	}
	nnMod := nn.NewModule()
	nnMod.Backends = append(nnMod.Backends, nnwebgpu.Backend{Limits: onnx.DefaultLimits, AllowFallback: allow})
	// Surfaces go to the page as the playground's engine worker sends them
	// to the dashboard (docs/DESIGN.md, "Surfaces"); the page acknowledges
	// the frames it drew.
	onSurface := func(ev webgpu.SurfaceEvent) {
		m := js.Global().Get("Object").New()
		m.Set("t", "surface")
		m.Set("op", ev.Op)
		m.Set("surface", ev.Surface)
		m.Set("container", ev.Container)
		m.Set("width", ev.Width)
		m.Set("height", ev.Height)
		if ev.Op != "frame" {
			js.Global().Call("postMessage", m)
			return
		}
		m.Set("t", "frame")
		m.Set("seq", float64(ev.Seq))
		m.Set("bitmap", ev.Bitmap)
		js.Global().Call("postMessage", m, []any{ev.Bitmap})
	}
	js.Global().Call("addEventListener", "message", js.FuncOf(func(_ js.Value, a []js.Value) any {
		d := a[0].Get("data")
		if d.Type() != js.TypeObject {
			return nil
		}
		switch d.Get("t").String() {
		case "frame-ack":
			webgpu.AckFrame(uint32(d.Get("surface").Int()), uint64(d.Get("seq").Float()))
		case "input":
			// The page's pointer and key events on a surface's pane
			// (docs/DESIGN.md, "Surfaces"), as the dashboard sends them.
			str := func(k string) string {
				if v := d.Get(k); v.Type() == js.TypeString {
					return v.String()
				}
				return ""
			}
			num := func(k string) float64 {
				if v := d.Get(k); v.Type() == js.TypeNumber {
					return v.Float()
				}
				return 0
			}
			webgpu.SurfaceInput(uint32(num("surface")), webgpu.InputEvent{
				Kind: str("kind"), X: num("x"), Y: num("y"), Code: str("code"), Text: str("text"),
				Alt: d.Get("alt").Truthy(), Ctrl: d.Get("ctrl").Truthy(), Meta: d.Get("meta").Truthy(), Shift: d.Get("shift").Truthy(),
			})
		}
		return nil
	}))
	mods := webgpu.HostModules(webgpu.Browser{AllowFallback: allow, OnSurface: onSurface})
	batchMod := gpubatch.NewModule(gpubatch.Browser{AllowFallback: allow})
	batchMod.Limits.Budget = time.Duration(param("budget")) * time.Millisecond
	batchMod.Limits.GPUTime = time.Duration(param("gputime")) * time.Millisecond
	exec := browser.New(browser.Options{
		WorkerURL:   "container-worker.js",
		HostModules: append(append(mods, wasi.HostModule(nnMod)), batchMod),
	})
	rep.Results = make([]Result, len(cases))
	var running sync.WaitGroup
	for i, c := range cases {
		i, c := i, c
		if c.DelayMs > 0 {
			time.Sleep(time.Duration(c.DelayMs) * time.Millisecond)
		}
		if c.Parallel {
			running.Add(1)
			go func() { defer running.Done(); rep.Results[i] = runCase(exec, c) }()
			continue
		}
		rep.Results[i] = runCase(exec, c)
	}
	running.Wait()
	rep.GPU = batchMod.Stats()
	rep.ConfiguredSurfaces = webgpu.ConfiguredSurfaces()
	post()
}

// runCase runs one container to its end.
func runCase(exec runtime.Executor, c Case) Result {
	{
		r := Result{Name: c.Name}
		wasm, err := fetch(c.Module)
		if err != nil {
			r.Error = err.Error()
			return r
		}
		fsys := rootfs.New()
		fsys.MkdirAll("/bin", 0o755)
		fsys.WriteFile("/bin/guest.wasm", wasm, 0o755)
		var so, se buf
		var mounts []runtime.Mount
		if len(c.Models) > 0 {
			mfs, err := modelMount(c.Models)
			if err != nil {
				r.Error = err.Error()
				return r
			}
			mounts = append(mounts, runtime.Mount{Path: "/models", FS: mfs, ReadOnly: true})
		}
		spec := runtime.Spec{ID: "c-" + c.Name, FS: fsys, Path: "/bin/guest.wasm", Args: append([]string{c.Name}, c.Args...), Stdout: &so, Stderr: &se, Mounts: mounts,
			MemoryLimitBytes: 512 << 20, Timeout: 2 * time.Minute}
		var evMu sync.Mutex
		spec.OnEvent = func(module, action string, attrs map[string]string) {
			keys := make([]string, 0, len(attrs))
			for k := range attrs {
				keys = append(keys, k)
			}
			sort.Strings(keys)
			s := module + "/" + action
			for _, k := range keys {
				s += " " + k + "=" + attrs[k]
			}
			evMu.Lock()
			r.Events = append(r.Events, s)
			evMu.Unlock()
		}
		if c.GPU {
			spec.Caps = []string{"gpu"}
			if c.GPUMemory > 0 {
				spec.Caps = append(spec.Caps, "gpu-memory="+strconv.FormatInt(c.GPUMemory, 10))
			}
		}
		t0 := time.Now()
		p, err := exec.Start(context.Background(), spec)
		if err != nil {
			r.Error = err.Error()
			return r
		}
		res := p.Wait()
		r.Ms = float64(time.Since(t0).Microseconds()) / 1000
		r.Exit, r.Stdout, r.Stderr = res.ExitCode, so.String(), se.String()
		if res.Error != "" {
			r.Error = res.Error
		}
		evMu.Lock()
		r.Events = append([]string(nil), r.Events...)
		evMu.Unlock()
		return r
	}
}

// modelMount verifies the models of a case and returns the tree mounted at
// /models: the size and sha256 are checked before the container exists.
func modelMount(refs []ModelRef) (*rootfs.FS, error) {
	host := js.Global().Get("location").Get("hostname").String()
	srcs := map[string]models.Source{}
	var es []models.Entry
	for _, m := range refs {
		e := models.Entry{Name: m.Name, Repo: "test/model", Revision: "0000000000000000000000000000000000000000", File: m.Name + ".gguf", Size: m.Size, SHA256: m.SHA256}
		// Fetched by ranges into the origin's OPFS cache, verified whole,
		// then read synchronously by the guest (models.OpenOPFS).
		src, err := models.OpenOPFS(context.Background(), e, func() (models.Source, error) {
			return models.NewHTTPSource(m.URL, m.Size, []string{host}, nil)
		})
		if err != nil {
			return nil, err
		}
		srcs[m.Name] = src
		es = append(es, e)
	}
	return models.Mount(srcs, es)
}
