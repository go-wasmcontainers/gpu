//go:build e2e

package e2e

import (
	"encoding/json"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

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
	build(t, root, filepath.Join(site, "hang.wasm"), "./e2e/testdata/hang", "wasip1")
	wasmExec, err := os.ReadFile(filepath.Join(runtime.GOROOT(), "lib", "wasm", "wasm_exec.js"))
	if err != nil {
		t.Fatal(err)
	}
	cs, _ := json.Marshal(cases)
	files := map[string][]byte{"index.html": []byte(enginePage + reportPoster), "engine.js": []byte(engineWorker), "wasm_exec.js": wasmExec,
		"container-worker.js": browser.WorkerJS, "cases.json": cs}
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(site, name), body, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	fs := http.FileServer(http.Dir(site))
	port, reports := serve(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Cross-Origin-Opener-Policy", "same-origin")
		w.Header().Set("Cross-Origin-Embedder-Policy", "require-corp")
		fs.ServeHTTP(w, r)
	}))
	v, version := fetchReport(t, port, reports, query, launch, 10*time.Minute)
	t.Logf("browser %s", version)
	var rep engineReport
	if err := json.Unmarshal([]byte(v), &rep); err != nil {
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
	if rep.AdapterError != "" {
		t.Fatalf("no WebGPU adapter in %s: %s", browserName(), rep.AdapterError)
	}
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
		{Name: "batch-shaders", Module: "batchcompute.wasm", Args: []string{"shaders"}, GPU: true},
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
			"out-of-range write: wasi_webgpu_batch: submit: refused: replay: range refused",
			"read outside the window: wasi_webgpu_batch: submit: refused: batch: read-back refused",
			"dispatch above the limit: wasi_webgpu_batch: submit: refused: replay: range refused",
			"id of another context: wasi_webgpu_batch: submit: refused: replay: unknown object 77",
			"bad magic: wasi_webgpu_batch: submit: refused (as expected: true)",
			"truncated frame: wasi_webgpu_batch: submit: refused (as expected: true)",
			"frame above max_frame: wasi_webgpu_batch: submit: too large (as expected: true)",
			"allocation past the quota: wasi_webgpu_batch: submit: quota exceeded: webgpu: quota: memory",
			"8 frames refused"}},
		"batch-quota": {0, []string{"buffer 0 ok", "buffer 1 ok", "buffer 2 refused"}},
		"batch-grant": {0, []string{"max_memory 1073741824", "1024 KiB: 262144 elements match the CPU"}},
		"batch-shaders": {0, []string{
			"valid control: <nil> (as expected: true)",
			"reserved word: wasi_webgpu_batch: submit: invalid shader or object (the GPU refused it): batch: the GPU refused the object",
			"syntax error: wasi_webgpu_batch: submit: invalid shader or object (the GPU refused it): batch: the GPU refused the object",
			"3 shader cases as expected"}},
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

// TestBatchDeadline: a guest submits a compute shader sized from the
// adapter's limits to run far past the host's 2 s of GPU time. The host must
// destroy the device at about 2 s and answer Timeout; the next submit of that
// container is ContextLost; and the engine keeps serving: containers started
// afterwards (compute read-back, refusals, a no-GPU container) pass.
func TestBatchDeadline(t *testing.T) {
	cases := []engineCase{
		{Name: "before", Module: "batchcompute.wasm", GPU: true},
		{Name: "hang", Module: "hang.wasm", GPU: true},
		{Name: "after", Module: "batchcompute.wasm", GPU: true},
		{Name: "after-refusals", Module: "batchcompute.wasm", Args: []string{"refusals"}, GPU: true},
		{Name: "after-no-gpu", Module: "batchcompute.wasm"},
	}
	rep := runBatchCases(t, cases, launchFallbackArgs(), allowFallback())
	requireGPU(t, rep)
	byName := map[string]int{}
	for i, r := range rep.Results {
		byName[r.Name] = i
		t.Logf("%s (%.0f ms): exit %d\n%s%s", r.Name, r.Ms, r.Exit, r.Stdout, r.Stderr)
	}
	h := rep.Results[byName["hang"]]
	m := deadlineMs.FindStringSubmatch(h.Stdout)
	if h.Exit != 0 || m == nil {
		t.Fatalf("hang: exit %d (%s), stdout %q", h.Exit, h.Error, h.Stdout)
	}
	ms, _ := strconv.Atoi(m[1])
	// 2 s timer, plus the round trip: at about 2 s, never well under, never
	// the several seconds the shader needs to finish.
	if ms < 1900 || ms > 3500 {
		t.Errorf("the host answered after %d ms, want about 2000", ms)
	}
	for _, s := range []string{"(timeout as expected: true)", "(device gone as expected: true)"} {
		if !strings.Contains(h.Stdout, s) {
			t.Errorf("hang stdout lacks %q", s)
		}
	}
	for name, out := range map[string]string{"before": "262144 elements match the CPU", "after": "262144 elements match the CPU",
		"after-refusals": "8 frames refused", "after-no-gpu": "no GPU: wasi_webgpu_batch: caps: denied"} {
		r := rep.Results[byName[name]]
		want := 0
		if name == "after-no-gpu" {
			want = 2
		}
		if r.Exit != want || !strings.Contains(r.Stdout, out) {
			t.Errorf("%s after the hang: exit %d (%s), stdout %q", name, r.Exit, r.Error, r.Stdout)
		}
	}
}

var deadlineMs = regexp.MustCompile(`long submit: .* after (\d+) ms`)

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
