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
	"slices"
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
	rep := runBatchCasesSanctions(t, cases, launch, query, 0)
	return rep
}

// pageQuery joins the page's query parameters: fallback=1 from the
// environment, and the budget and hard cap in ms (0: the module's defaults),
// GPU_E2E_BUDGET_MS and GPU_E2E_GPUTIME_MS override them.
func pageQuery(budgetMs, gpuTimeMs int) string {
	var q []string
	if allowFallback() != "" {
		q = append(q, "fallback=1")
	}
	for k, env := range map[string]string{"budget": "GPU_E2E_BUDGET_MS", "gputime": "GPU_E2E_GPUTIME_MS",
		// layer 4: the loop budget's work, its floor, and 1 to switch it off
		"loopwork": "GPU_E2E_LOOP_WORK", "loopmin": "GPU_E2E_LOOP_MIN", "noloop": "GPU_E2E_NO_LOOP_BUDGET"} {
		v := os.Getenv(env)
		if v == "" {
			switch k {
			case "budget":
				v = strconv.Itoa(budgetMs)
			case "gputime":
				v = strconv.Itoa(gpuTimeMs)
			}
		}
		if n, _ := strconv.Atoi(v); n > 0 {
			q = append(q, k+"="+v)
		}
	}
	if len(q) == 0 {
		return ""
	}
	slices.Sort(q)
	return "?" + strings.Join(q, "&")
}

// runBatchCasesSanctions is runBatchCases for a test that expects exactly
// `sanctions` GPU sanctions; any other number fails it (a sanction of
// legitimate work is a false positive). The module's measured submit times
// are logged either way.
func runBatchCasesSanctions(t *testing.T, cases []engineCase, launch []string, query string, sanctions int64) engineReport {
	t.Helper()
	rep := runBatchCasesRaw(t, cases, launch, query)
	t.Logf("gpu time budget: %s", rep.GPU)
	if rep.GPU.Sanctions != sanctions {
		t.Errorf("%d GPU sanction(s), want %d", rep.GPU.Sanctions, sanctions)
	}
	if rep.GPU.LoopRefused != 0 {
		t.Errorf("%d WGSL module(s) the browser refused after the loop-budget rewrite", rep.GPU.LoopRefused)
	}
	return rep
}

func runBatchCasesRaw(t *testing.T, cases []engineCase, launch []string, query string) engineReport {
	t.Helper()
	root, err := filepath.Abs("..")
	if err != nil {
		t.Fatal(err)
	}
	site := t.TempDir()
	build(t, root, filepath.Join(site, "enginetest.wasm"), "./internal/cmd/enginetest", "js")
	build(t, root, filepath.Join(site, "batchcompute.wasm"), "./examples/batchcompute", "wasip1")
	build(t, root, filepath.Join(site, "hang.wasm"), "./e2e/testdata/hang", "wasip1")
	build(t, root, filepath.Join(site, "victim.wasm"), "./e2e/testdata/victim", "wasip1")
	build(t, root, filepath.Join(site, "hangwasi.wasm"), "./e2e/testdata/hangwasi", "wasip1")
	build(t, root, filepath.Join(site, "mnist.wasm"), "./examples/mnist", "wasip1")
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
	rep := runBatchCases(t, cases, launchFallbackArgs(), pageQuery(0, 0))
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
	rep := runBatchCasesSanctions(t, cases, launchFallbackArgs(), pageQuery(0, 0), 1)
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
	// A software adapter (SwiftShader) runs the same loop about a thousand
	// times slower than the GPU the default Work was measured on, so the
	// hostile shader fits under the budget and the 2 s cap cuts it instead:
	// the loop budget's Work is per GPU (docs/gpu-scheduling.md).
	soft := rep.Adapter.Fallback || strings.Contains(strings.ToLower(rep.Adapter.Architecture+rep.Adapter.Vendor), "swiftshader")
	if loopBudgetOn() && !soft {
		// Layer 4: the loop is cut, the host answers Truncated, well before
		// the 2 s cap.
		if ms > 1500 {
			t.Errorf("the host answered after %d ms, want a loop-budget cut well under the 2000 ms cap", ms)
		}
		if !strings.Contains(h.Stdout, "by the loop budget: true") {
			t.Errorf("hang stdout lacks a loop-budget cut: %q", h.Stdout)
		}
	} else if ms < 1900 || ms > 3500 {
		// 2 s timer, plus the round trip: at about 2 s, never well under,
		// never the several seconds the shader needs to finish.
		t.Errorf("the host answered after %d ms, want about 2000", ms)
	}
	if len(h.Events) != 1 || !strings.HasPrefix(h.Events[0], "gpu/sanction ") {
		t.Errorf("engine events of the hostile container: %q", h.Events)
	}
	for _, s := range []string{"(cut as expected: true", "(device gone as expected: true)", "(revoked as expected: true)"} {
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

// loopBudgetOn is whether the run has layer 4 on (the default); the A/B
// measurements and the pre-layer-4 baseline set GPU_E2E_NO_LOOP_BUDGET=1.
func loopBudgetOn() bool { return os.Getenv("GPU_E2E_NO_LOOP_BUDGET") != "1" }

var deadlineMs = regexp.MustCompile(`long submit: .* after (\d+) ms`)

// TestBatchComputeQuota: a dispatch of 65535 x 16 workgroups, which the
// host's compute quota (2^26 invocations per dispatch) refuses, never reaches
// the GPU: the container gets Refused, not Timeout, and nobody is sanctioned.
func TestBatchComputeQuota(t *testing.T) {
	rep := runBatchCasesSanctions(t, []engineCase{{Name: "wide", Module: "hang.wasm", Args: []string{"3000", "wide"}, GPU: true}},
		launchFallbackArgs(), pageQuery(0, 0), 0)
	requireGPU(t, rep)
	r := rep.Results[0]
	t.Logf("%s (%.0f ms): exit %d\n%s%s", r.Name, r.Ms, r.Exit, r.Stdout, r.Stderr)
	if r.Exit != 0 || !strings.Contains(r.Stdout, "(refused as expected: true)") {
		t.Errorf("exit %d (%s), stdout %q", r.Exit, r.Error, r.Stdout)
	}
}

var victimStats = regexp.MustCompile(`victim: (\d+) submits, median ([0-9.]+) ms, max ([0-9.]+) ms, errors (\d+)`)

// TestBatchSanction (go-wasmcontainers/gpu#26): a hostile container runs a
// shader calibrated to about 3 s while a victim container makes short submits
// throughout. The hostile container is sanctioned once (its device destroyed,
// its grant revoked: a new context answers Revoked, the engine event is
// reported); the victim is never sanctioned and its submits keep succeeding
// and reading back correctly; its worst latency is measured and bounded by
// what the GPU still held when the hostile's device was destroyed (the
// remaining work of the shader, which no browser lets the host cancel).
// "combined" puts the shader's creation and its dispatch in one frame, which
// the host holds to the hard cap; "split" gives the dispatch a frame of its
// own, held to the steady-state budget.
func TestBatchSanction(t *testing.T) {
	if load := loadAverage(); load > 8 {
		t.Skipf("1-minute load average %.1f above 8: not measuring", load)
	}
	// "control" runs the same pair with the budget and the cap raised to 60 s:
	// nobody is sanctioned, the hostile shader runs to its end and the victim
	// waits for all of it. That is what the victim pays with no sanction at
	// all, on this machine at this moment, and what the sanctioned runs must
	// not exceed by more than a margin: sanctioning cannot shorten the
	// shader (no browser cancels submitted work), it must not lengthen the
	// wait either.
	for _, mode := range []string{"control", "combined", "split"} {
		t.Run(mode, func(t *testing.T) {
			// The combined frame is held to the 2 s cap, and the calibration is
			// coarse (Firefox resolves a submit in 200 ms steps: a "3 s" shader
			// ran 1.3 s once), so it aims higher to be sure to pass the cap.
			args := []string{"3000"}
			if mode != "split" {
				args = []string{"4500"}
			}
			if mode == "split" {
				args = append(args, "split")
			}
			cases := []engineCase{
				{Name: "victim", Module: "victim.wasm", Args: []string{"25000", "20"}, GPU: true, Parallel: true},
				{Name: "hostile", Module: "hang.wasm", Args: args, GPU: true, DelayMs: 1000},
			}
			query, want := pageQuery(0, 0), int64(1)
			if mode == "control" {
				query, want = "?budget=60000&gputime=60000&noloop=1", 0 // no loop budget either: nothing bounds the shader
				if allowFallback() != "" {
					query += "&fallback=1"
				}
			}
			rep := runBatchCasesSanctions(t, cases, launchFallbackArgs(), query, want)
			requireGPU(t, rep)
			byName := map[string]int{}
			for i, r := range rep.Results {
				byName[r.Name] = i
				t.Logf("%s (%.0f ms): exit %d\n%s%s events %q", r.Name, r.Ms, r.Exit, r.Stdout, r.Stderr, r.Events)
			}
			h, v := rep.Results[byName["hostile"]], rep.Results[byName["victim"]]
			if mode != "control" {
				if h.Exit != 0 || !strings.Contains(h.Stdout, "(cut as expected: true") || !strings.Contains(h.Stdout, "(revoked as expected: true)") {
					t.Errorf("hostile: exit %d (%s), stdout %q", h.Exit, h.Error, h.Stdout)
				}
				if len(h.Events) != 1 || !strings.HasPrefix(h.Events[0], "gpu/sanction ") || len(v.Events) != 0 {
					t.Errorf("engine events: hostile %q, victim %q", h.Events, v.Events)
				}
			}
			if mode == "control" && browserName() == "safari" {
				// Safari destroys the hostile's device itself after about 3 s
				// and the victim's queued submit never completes; then it
				// answers "no adapter" for a minute or more (#32). What the host
				// owes the victim is not to sanction it for that: the device is
				// reset (an event), the grant kept. Whether Safari offers an
				// adapter again within the run is Safari's.
				resets := 0
				for _, e := range v.Events {
					if strings.HasPrefix(e, "gpu/device-reset ") {
						resets++
					}
					if strings.HasPrefix(e, "gpu/sanction ") {
						t.Errorf("the victim was sanctioned: %q", v.Events)
					}
				}
				if resets != 1 {
					t.Errorf("victim events %q: want one gpu/device-reset", v.Events)
				}
				return
			}
			m := victimStats.FindStringSubmatch(v.Stdout)
			if v.Exit != 0 || m == nil || m[4] != "0" {
				t.Fatalf("victim: exit %d (%s), stdout %q", v.Exit, v.Error, v.Stdout)
			}
			median, _ := strconv.ParseFloat(m[2], 64)
			worst, _ := strconv.ParseFloat(m[3], 64)
			t.Logf("MEASURED %s: the victim's median submit %.1f ms, worst case %.1f ms (the hostile's shader aims at 3000 ms)", mode, median, worst)
			if load := loadAverage(); load > 8 {
				t.Skipf("1-minute load average %.1f above 8 after the run: the latency above is not a measurement", load)
			}
			if mode == "control" {
				return
			}
			// The worst case is the work the GPU still held after the
			// destroy: the hostile shader's own remaining length, which the
			// host cannot shorten and the test cannot fix (the shader is
			// calibrated on whatever else uses the GPU, and on Firefox a
			// submit resolves in 200 ms steps, so the aim is coarse). It is
			// logged, not compared; the assertion is that the victim waited
			// for the shader and not for the whole run.
			if worst > 20000 {
				t.Errorf("the victim's worst submit took %.1f ms of a 25 s run", worst)
			}
		})
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
		launchFallbackArgs(), pageQuery(0, 0))
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

// TestLoopBudgetRefusals: what the host refuses for its loop budget, as seen
// from a container: errno 13 (the host's bind group) and errno 15 (a module
// the host cannot bound), and the bounded control passes.
func TestLoopBudgetRefusals(t *testing.T) {
	if !loopBudgetOn() {
		t.Skip("the loop budget is off")
	}
	rep := runBatchCasesRaw(t, []engineCase{{Name: "unbounded", Module: "batchcompute.wasm", Args: []string{"unbounded"}, GPU: true}},
		launchFallbackArgs(), pageQuery(0, 0))
	requireGPU(t, rep)
	r := rep.Results[0]
	t.Logf("%s (%.0f ms): exit %d\n%s%s", r.Name, r.Ms, r.Exit, r.Stdout, r.Stderr)
	if r.Exit != 0 || !strings.Contains(r.Stdout, "7 unbounded cases as expected") {
		t.Errorf("exit %d (%s), stdout %q", r.Exit, r.Error, r.Stdout)
	}
	if rep.GPU.Sanctions != 0 || rep.GPU.LoopRefused < 3 {
		t.Errorf("sanctions %d (a refusal is not a sanction), refused %d (want at least 3)", rep.GPU.Sanctions, rep.GPU.LoopRefused)
	}
}

// TestLoopBudgetWasiWebGPU: the wasi:webgpu host module (no scheduler) is not
// a way around the loop budget: a hostile container's loop is cut, the
// read-back traps with the truncation, and the guest never sees a partial
// result.
func TestLoopBudgetWasiWebGPU(t *testing.T) {
	if !loopBudgetOn() {
		t.Skip("the loop budget is off")
	}
	rep := runBatchCasesRaw(t, []engineCase{{Name: "hangwasi", Module: "hangwasi.wasm", GPU: true}}, launchFallbackArgs(), pageQuery(0, 0))
	requireGPU(t, rep)
	r := rep.Results[0]
	t.Logf("%s (%.0f ms): exit %d (%s)\n%s%s", r.Name, r.Ms, r.Exit, r.Error, r.Stdout, r.Stderr)
	if r.Exit == 0 || strings.Contains(r.Stdout, "NOT CUT") || !strings.Contains(r.Error+r.Stderr, "loop budget") {
		t.Errorf("the hostile wasi:webgpu container was not cut by the loop budget: exit %d (%s), stdout %q", r.Exit, r.Error, r.Stdout)
	}
	if r.Ms > 2000 {
		t.Errorf("the cut took %.0f ms", r.Ms)
	}
}

// TestLoopBudgetWasiNN: the wasi-nn GPU kernels are rewritten too. With the
// default work the MNIST model runs; with the work and the floor at 1 its
// loops are cut and the run fails with the truncation.
func TestLoopBudgetWasiNN(t *testing.T) {
	if !loopBudgetOn() {
		t.Skip("the loop budget is off")
	}
	mnist := []engineCase{{Name: "mnist-gpu", Module: "mnist.wasm", Args: []string{"gpu"}, GPU: true}}
	ok := runBatchCasesRaw(t, mnist, launchFallbackArgs(), pageQuery(0, 0))
	requireGPU(t, ok)
	if r := ok.Results[0]; r.Exit != 0 || !strings.Contains(r.Stdout, "digit 7") {
		t.Errorf("default work: exit %d (%s), stdout %q", r.Exit, r.Error, r.Stdout)
	}
	cut := runBatchCasesRaw(t, mnist, launchFallbackArgs(), "?nnloopwork=1&nnloopmin=1")
	r := cut.Results[0]
	t.Logf("work 1: exit %d (%s)\n%s%s", r.Exit, r.Error, r.Stdout, r.Stderr)
	// wasi-nn reports a failed kernel as a runtime error at compute; the
	// default run above is the control that the loops are what was cut.
	if r.Exit == 0 || strings.Contains(r.Stdout, "digit 7") || !strings.Contains(r.Error+r.Stderr+r.Stdout, "compute: wasi-nn: runtime error") {
		t.Errorf("work 1: the kernels were not cut: exit %d (%s), stdout %q", r.Exit, r.Error, r.Stdout)
	}
}
