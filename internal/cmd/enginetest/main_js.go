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
	"sync"
	"syscall/js"
	"time"

	"github.com/go-wasmcontainers/engine/rootfs"
	"github.com/go-wasmcontainers/engine/runtime"
	"github.com/go-wasmcontainers/engine/runtime/browser"
	"github.com/go-wasmcontainers/engine/wasi"
	"github.com/go-wasmcontainers/gpu/internal/jsgpu"
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
}

// Result is how it ended.
type Result struct {
	Name   string  `json:"name"`
	Exit   int     `json:"exit"`
	Stdout string  `json:"stdout"`
	Stderr string  `json:"stderr"`
	Error  string  `json:"error,omitempty"`
	Ms     float64 `json:"ms"`
}

// Report is what the worker posts.
type Report struct {
	Adapter jsgpu.AdapterInfo `json:"adapter"`
	Results []Result          `json:"results"`
	Error   string            `json:"error,omitempty"`
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
	if d, err := jsgpu.NewDevice(true); err == nil {
		rep.Adapter = d.Info
		d.Destroy()
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
	exec := browser.New(browser.Options{
		WorkerURL:   "container-worker.js",
		HostModules: append(mods, wasi.HostModule(nnMod)),
	})
	for _, c := range cases {
		r := Result{Name: c.Name}
		wasm, err := fetch(c.Module)
		if err != nil {
			r.Error = err.Error()
			rep.Results = append(rep.Results, r)
			continue
		}
		fsys := rootfs.New()
		fsys.MkdirAll("/bin", 0o755)
		fsys.WriteFile("/bin/guest.wasm", wasm, 0o755)
		var so, se buf
		spec := runtime.Spec{ID: "c-" + c.Name, FS: fsys, Path: "/bin/guest.wasm", Args: append([]string{c.Name}, c.Args...), Stdout: &so, Stderr: &se,
			MemoryLimitBytes: 512 << 20, Timeout: 2 * time.Minute}
		if c.GPU {
			spec.Caps = []string{"gpu"}
		}
		t0 := time.Now()
		p, err := exec.Start(context.Background(), spec)
		if err != nil {
			r.Error = err.Error()
			rep.Results = append(rep.Results, r)
			continue
		}
		res := p.Wait()
		r.Ms = float64(time.Since(t0).Microseconds()) / 1000
		r.Exit, r.Stdout, r.Stderr = res.ExitCode, so.String(), se.String()
		if res.Error != "" {
			r.Error = res.Error
		}
		rep.Results = append(rep.Results, r)
	}
	post()
}
