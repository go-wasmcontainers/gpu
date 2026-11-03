//go:build e2e

package e2e

import (
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"

	"github.com/mxschmitt/playwright-go"

	"github.com/go-wasmcontainers/engine/runtime/browser"
)

// runBatchCases runs containers built from examples/batchcompute in the
// engine's browser executor (enginetest, with the host module
// wasi_webgpu_batch) and returns the worker's report. launch is extra
// Chromium flags; query is the page's query string ("?fallback=1" lets the
// host accept a software adapter).
func runBatchCases(t *testing.T, cases []engineCase, launch []string, query string) engineReport {
	t.Helper()
	root, err := filepath.Abs("..")
	if err != nil {
		t.Fatal(err)
	}
	site := t.TempDir()
	build(t, root, filepath.Join(site, "enginetest.wasm"), "./internal/cmd/enginetest", "js")
	build(t, root, filepath.Join(site, "batchcompute.wasm"), "./examples/batchcompute", "wasip1")
	wasmExec, err := os.ReadFile(filepath.Join(runtime.GOROOT(), "lib", "wasm", "wasm_exec.js"))
	if err != nil {
		t.Fatal(err)
	}
	cs, _ := json.Marshal(cases)
	files := map[string][]byte{"index.html": []byte(enginePage), "engine.js": []byte(engineWorker), "wasm_exec.js": wasmExec,
		"container-worker.js": browser.WorkerJS, "cases.json": cs}
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(site, name), body, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	fs := http.FileServer(http.Dir(site))
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	srv := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Cross-Origin-Opener-Policy", "same-origin")
		w.Header().Set("Cross-Origin-Embedder-Policy", "require-corp")
		fs.ServeHTTP(w, r)
	})}
	go srv.Serve(ln)
	defer srv.Close()

	pw, err := playwright.Run()
	if err != nil {
		t.Fatal(err)
	}
	defer pw.Stop()
	channel := os.Getenv("GPU_E2E_CHANNEL")
	if channel == "" {
		channel = "chromium"
	}
	opts := playwright.BrowserTypeLaunchOptions{Headless: playwright.Bool(true), Channel: playwright.String(channel)}
	opts.Args = append(append(opts.Args, strings.Fields(os.Getenv("GPU_E2E_ARGS"))...), launch...)
	b, err := pw.Chromium.Launch(opts)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	p, err := b.NewPage()
	if err != nil {
		t.Fatal(err)
	}
	var console []string
	p.On("console", func(m playwright.ConsoleMessage) { console = append(console, m.Text()) })
	if _, err := p.Goto(fmt.Sprintf("http://localhost:%d/index.html%s", ln.Addr().(*net.TCPAddr).Port, query)); err != nil {
		t.Fatal(err)
	}
	if _, err := p.WaitForFunction("() => window.__report !== undefined", nil, playwright.PageWaitForFunctionOptions{Timeout: playwright.Float(600000)}); err != nil {
		t.Fatalf("%v\nconsole: %s", err, strings.Join(console, "\n"))
	}
	v, _ := p.Evaluate("window.__report")
	var rep engineReport
	if err := json.Unmarshal([]byte(v.(string)), &rep); err != nil {
		t.Fatal(err)
	}
	if rep.Error != "" {
		t.Fatalf("engine worker: %s", rep.Error)
	}
	if len(rep.Results) != len(cases) {
		t.Fatalf("%d results for %d cases", len(rep.Results), len(cases))
	}
	return rep
}

// requireGPU fails the run when the adapter is a software one and the test
// did not allow it: the run could not prove the GPU path.
func requireGPU(t *testing.T, rep engineReport) {
	t.Helper()
	a := rep.Adapter
	fallback := a.Fallback || strings.Contains(strings.ToLower(a.Architecture+a.Vendor), "swiftshader")
	t.Logf("adapter %s %s, fallback %v", a.Vendor, a.Architecture, fallback)
	if fallback && os.Getenv("GPU_E2E_ALLOW_FALLBACK") != "1" {
		t.Fatalf("adapter %s %s is a fallback adapter: this run cannot prove the GPU path", a.Vendor, a.Architecture)
	}
}

func allowFallback() string {
	if os.Getenv("GPU_E2E_ALLOW_FALLBACK") == "1" {
		return "?fallback=1"
	}
	return ""
}

func launchFallbackArgs() []string {
	if os.Getenv("GPU_E2E_ALLOW_FALLBACK") == "1" {
		return []string{"--enable-unsafe-webgpu"}
	}
	return nil
}

// TestBatchComputeContainers: a wasip1 container records a compute shader
// that writes known values, copies them to a MAP_READ buffer and reads them
// back through wasi_webgpu_batch, in a real container in the browser
// executor on the machine's GPU; every element matches. Refusals: no
// --gpu, frames the host refuses (an out-of-range write caught before
// WebGPU, a read outside the window, malformed and oversized frames, an
// allocation past the quota), and an allocation past a smaller
// --gpu-memory.
func TestBatchComputeContainers(t *testing.T) {
	cases := []engineCase{
		{Name: "batch", Module: "batchcompute.wasm", GPU: true},
		{Name: "batch-no-gpu", Module: "batchcompute.wasm"},
		{Name: "batch-refusals", Module: "batchcompute.wasm", Args: []string{"refusals"}, GPU: true},
		{Name: "batch-quota", Module: "batchcompute.wasm", Args: []string{"quota"}, GPU: true, GPUMemory: 128 << 20},
		{Name: "batch-grant", Module: "batchcompute.wasm", GPU: true, GPUMemory: 1 << 30},
	}
	rep := runBatchCases(t, cases, launchFallbackArgs(), allowFallback())
	requireGPU(t, rep)
	want := map[string]struct {
		exit int
		out  []string
	}{
		"batch": {0, []string{"1024 KiB: 262144 elements match the CPU, 1 host crossing(s)", "4 KiB: 1024 elements match the CPU, 1 host crossing(s)",
			"max_frame 33554432", "max_memory 268435456"}},
		"batch-no-gpu": {2, []string{"no GPU: wasi_webgpu_batch: caps: denied"}},
		"batch-refusals": {0, []string{
			"out-of-range write: wasi_webgpu_batch: submit: refused (as expected: true)",
			"read outside the window: wasi_webgpu_batch: submit: refused (as expected: true)",
			"dispatch above the limit: wasi_webgpu_batch: submit: refused (as expected: true)",
			"id of another context: wasi_webgpu_batch: submit: refused (as expected: true)",
			"bad magic: wasi_webgpu_batch: submit: refused (as expected: true)",
			"truncated frame: wasi_webgpu_batch: submit: refused (as expected: true)",
			"frame above max_frame: wasi_webgpu_batch: submit: too large (as expected: true)",
			"allocation past the quota: wasi_webgpu_batch: submit: quota exceeded (as expected: true)",
			"8 frames refused"}},
		"batch-quota": {0, []string{"buffer 0 ok", "buffer 1 ok", "buffer 2 refused"}},
		"batch-grant": {0, []string{"max_memory 1073741824", "1024 KiB: 262144 elements match the CPU"}},
	}
	for _, r := range rep.Results {
		w := want[r.Name]
		t.Logf("%s (%.0f ms): exit %d\n%s%s", r.Name, r.Ms, r.Exit, r.Stdout, r.Stderr)
		ok := r.Exit == w.exit
		for _, s := range w.out {
			if !strings.Contains(r.Stdout, s) {
				t.Errorf("%s: stdout lacks %q", r.Name, s)
				ok = false
			}
		}
		if !ok {
			t.Errorf("%s: exit %d (%s), want %d; stderr %q", r.Name, r.Exit, r.Error, w.exit, r.Stderr)
		}
	}
}

// TestBatchFallbackRefused: on a software adapter, which the host module
// refuses unless told otherwise, the container is told there is no GPU
// (Unsupported), not given the software adapter.
func TestBatchFallbackRefused(t *testing.T) {
	// SwiftShader forced on any machine; the page does not pass ?fallback=1,
	// so the host module's Browser{} keeps AllowFallback off.
	rep := runBatchCases(t, []engineCase{{Name: "batch-fallback", Module: "batchcompute.wasm", GPU: true}},
		[]string{"--enable-unsafe-webgpu", "--use-webgpu-adapter=swiftshader"}, "")
	if !rep.Adapter.Fallback {
		t.Skipf("the browser still offers a hardware adapter (%s %s): the flag did not force SwiftShader", rep.Adapter.Vendor, rep.Adapter.Architecture)
	}
	r := rep.Results[0]
	t.Logf("%s: exit %d\n%s%s", r.Name, r.Exit, r.Stdout, r.Stderr)
	if r.Exit != 2 || !strings.Contains(r.Stdout, "no GPU: wasi_webgpu_batch: caps: unsupported") {
		t.Errorf("exit %d, stdout %q", r.Exit, r.Stdout)
	}
}

// loadAverage is the 1-minute load average (uptime), or -1.
func loadAverage() float64 {
	out, err := exec.Command("uptime").Output()
	if err != nil {
		return -1
	}
	s := string(out)
	i := strings.LastIndex(s, "load average")
	if i < 0 {
		return -1
	}
	f := strings.FieldsFunc(s[i:], func(r rune) bool { return r == ' ' || r == ',' || r == ':' })
	for _, x := range f {
		if v, err := strconv.ParseFloat(strings.ReplaceAll(x, ",", "."), 64); err == nil {
			return v
		}
	}
	return -1
}

// TestBatchRoundTrip measures the per-submit round trip seen from the guest
// at 4 KiB and 1 MiB, interleaved, and logs it next to the host-only figures
// of go-gfx/webgpu docs/readback.md (124 and 155 us for a read, 270 and
// 320 us for compute + copy + read). It is skipped above a 1-minute load of
// 8: a time measured under load is not a measurement.
func TestBatchRoundTrip(t *testing.T) {
	if load := loadAverage(); load > 8 {
		t.Skipf("1-minute load average %.1f above 8: not measuring", load)
	} else {
		t.Logf("1-minute load average before: %.2f", load)
	}
	rep := runBatchCases(t, []engineCase{{Name: "batch-bench", Module: "batchcompute.wasm", Args: []string{"bench", "30"}, GPU: true}},
		launchFallbackArgs(), allowFallback())
	requireGPU(t, rep)
	r := rep.Results[0]
	t.Logf("load average after: %.2f\n%s%s", loadAverage(), r.Stdout, r.Stderr)
	if r.Exit != 0 || strings.Count(r.Stdout, "bench ") < 5 {
		t.Fatalf("exit %d (%s), stdout %q", r.Exit, r.Error, r.Stdout)
	}
	if !strings.Contains(r.Stdout, "bench crossings: 120 for 120 submits") {
		t.Errorf("one crossing per submit expected: %q", r.Stdout)
	}
}
