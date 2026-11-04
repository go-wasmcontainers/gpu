//go:build e2e

package e2e

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net"
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

	"github.com/mxschmitt/playwright-go"

	"github.com/go-wasmcontainers/engine/runtime/browser"
)

// The LLM end-to-end tests need three things outside this repository:
//
//	LLM_MODELS  a directory holding SmolLM2-135M-Instruct-Q4_0.gguf (pinned in
//	            go-wasmcontainers-llm-corpus; model files are never committed)
//	LLM_REPO    a clone of go-wasmcontainers/llm (the container is built from it)
//
// and skip cleanly without them.

const llmDefaultModelFile = "SmolLM2-135M-Instruct-Q4_0.gguf"

// The model under test: LLM_MODEL_FILE (the GGUF in LLM_MODELS), LLM_REGISTRY_NAME
// (the name under /models), LLM_ORACLE_FILE (testdata/ of the llm repository)
// and LLM_GPU_MEMORY (MiB) default to SmolLM2-135M Q4_0, so the SmolLM2 runs
// are unchanged; set them for another model, e.g. Qwen2.5-0.5B:
//
//	LLM_MODEL_FILE=qwen2.5-0.5b-instruct-q4_0.gguf LLM_REGISTRY_NAME=qwen2.5-0.5b-instruct-q4_0
//	LLM_ORACLE_FILE=oracle-qwen2.5-0.5b-dequant-f32.json LLM_GPU_MEMORY=1024
func llmModelFile() string {
	if v := os.Getenv("LLM_MODEL_FILE"); v != "" {
		return v
	}
	return llmDefaultModelFile
}

func llmRegistryName() string {
	if v := os.Getenv("LLM_REGISTRY_NAME"); v != "" {
		return v
	}
	return "smollm2-135m-instruct-q4_0"
}

func llmOracleFile() string {
	if v := os.Getenv("LLM_ORACLE_FILE"); v != "" {
		return v
	}
	return "oracle-smollm2-135m-dequant-f32.json"
}

// llmGPUMemory is the container's GPU memory grant, in bytes.
func llmGPUMemory() int64 {
	if v, _ := strconv.Atoi(os.Getenv("LLM_GPU_MEMORY")); v > 0 {
		return int64(v) << 20
	}
	return 512 << 20
}

type llmEnv struct {
	model string
	repo  string
	size  int64
	sum   string
}

func needLLM(t *testing.T) llmEnv {
	t.Helper()
	dir, repo := os.Getenv("LLM_MODELS"), os.Getenv("LLM_REPO")
	if dir == "" || repo == "" {
		t.Skip("LLM_MODELS and LLM_REPO are not set")
	}
	p := filepath.Join(dir, llmModelFile())
	f, err := os.Open(p)
	if err != nil {
		t.Skip(err)
	}
	defer f.Close()
	h := sha256.New()
	n, err := io.Copy(h, f)
	if err != nil {
		t.Fatal(err)
	}
	return llmEnv{model: p, repo: repo, size: n, sum: hex.EncodeToString(h.Sum(nil))}
}

// runLLMCases runs cases in the browser executor with the model served by
// the test's own server (ranges, same origin) and mounted at /models.
func runLLMCases(t *testing.T, env llmEnv, cases []engineCase, extraArgs []string) engineReport {
	t.Helper()
	root, err := filepath.Abs("..")
	if err != nil {
		t.Fatal(err)
	}
	site := t.TempDir()
	build(t, root, filepath.Join(site, "enginetest.wasm"), "./internal/cmd/enginetest", "js")
	cmd := exec.Command("go", "build", "-o", filepath.Join(site, "generate.wasm"), "./examples/generate")
	cmd.Dir = env.repo
	// GOFLAGS= : a -modfile of this repository's must not leak into llm's own module.
	cmd.Env = append(os.Environ(), "GOOS=wasip1", "GOARCH=wasm", "CGO_ENABLED=0", "GOFLAGS=")
	if b, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build the llm container: %v\n%s", err, b)
	}
	if err := os.Symlink(env.model, filepath.Join(site, "model.gguf")); err != nil {
		t.Fatal(err)
	}
	wasmExec, err := os.ReadFile(filepath.Join(runtime.GOROOT(), "lib", "wasm", "wasm_exec.js"))
	if err != nil {
		t.Fatal(err)
	}
	files := map[string][]byte{"index.html": []byte(enginePage + reportPoster), "engine.js": []byte(engineWorker), "wasm_exec.js": wasmExec,
		"container-worker.js": browser.WorkerJS}
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
	origin := fmt.Sprintf("http://localhost:%d", port)
	for i := range cases {
		cases[i].Models = []engineModel{{Name: llmRegistryName(), URL: origin + "/model.gguf", Size: env.size, SHA256: env.sum}}
		if n := os.Getenv("LLM_REGISTRY_NAME"); n != "" && len(cases[i].Args) > 0 { // a model other than the default of the example
			cases[i].Args = append([]string{cases[i].Args[0], "-model", "/models/" + n + ".gguf"}, cases[i].Args[1:]...)
		}
	}
	cs, _ := json.Marshal(cases)
	if err := os.WriteFile(filepath.Join(site, "cases.json"), cs, 0o644); err != nil {
		t.Fatal(err)
	}
	// Chromium (Playwright), real Firefox or Safari, as in the batch lanes
	// (GPU_E2E_BROWSER); the page posts its report to the server.
	v, version := fetchReport(t, port, reports, pageQuery(0, 0), extraArgs, 30*time.Minute)
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
	// The GPU time budget (go-wasmcontainers/gpu#26) must never trip on the
	// LLM kernels: no false positive, and the longest submit is on record.
	t.Logf("gpu time budget: %s", rep.GPU)
	if rep.GPU.Sanctions != 0 {
		t.Errorf("%d GPU sanction(s) on legitimate LLM work", rep.GPU.Sanctions)
	}
	return rep
}

// oracleRecord is testdata/oracle-smollm2-135m-dequant-f32.json of go-wasmcontainers/llm:
// llama.cpp's greedy tokens and top-10 log-probabilities per step.
type oracleRecord struct {
	Prompt    string  `json:"prompt"`
	PromptIDs []int32 `json:"prompt_ids"`
	Tokens    []int32 `json:"tokens"`
	Steps     [][]struct {
		ID      int32   `json:"id"`
		Logprob float64 `json:"logprob"`
	} `json:"steps"`
}

func loadOracle(t *testing.T, env llmEnv) []oracleRecord {
	b, err := os.ReadFile(filepath.Join(env.repo, "testdata", llmOracleFile()))
	if err != nil {
		t.Fatal(err)
	}
	var o []oracleRecord
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatal(err)
	}
	return o
}

var idList = regexp.MustCompile(`^(\w+( \w+)?): \[([-0-9 ]*)\]`)

func intsAfter(out, key string) []int32 {
	for _, l := range strings.Split(out, "\n") {
		if strings.HasPrefix(l, key+": [") {
			var ids []int32
			for _, f := range strings.Fields(strings.Trim(strings.TrimPrefix(l, key+": "), "[]")) {
				n, _ := strconv.Atoi(f)
				ids = append(ids, int32(n))
			}
			return ids
		}
	}
	return nil
}

// Vendor-sensitive tolerances: this runs on whatever GPU the browser offers
// (only Apple Metal has been run). The f32 kernels sum in a fixed order that
// depends on the workgroup size, not on the hardware, but a different
// compiler may fuse multiply-adds and exp() differs by a few ulp.
const (
	gpuCPULogitRel = 1e-3 // max |gpu-cpu| over the logits, relative to max |logit|
	// A token differing from the f32 reference is accepted only at a gap of
	// llama.cpp's own log-probabilities this small (on Apple Metal none
	// differs; another vendor's rounding can flip only an almost exact tie).
	llamaNearTie  = 1e-3
	llamaMinEqual = 0
)

// TestLLMOracle: SmolLM2-135M Q4_0 generated on the GPU in a container, from a
// read-only /models mount, equals llama.cpp's greedy tokens on the fixed
// prompts (a divergence is accepted only at a near-tie of llama.cpp's own
// distribution), and the logits of the GPU path agree with the Go CPU path.
func TestLLMOracle(t *testing.T) {
	env := needLLM(t)
	oracle := loadOracle(t, env)
	var cases []engineCase
	for i, o := range oracle {
		cases = append(cases, engineCase{Name: fmt.Sprintf("run%d", i), Module: "generate.wasm", GPU: true, GPUMemory: llmGPUMemory(),
			Args: []string{"run", "-n", "32", o.Prompt}})
	}
	cases = append(cases, engineCase{Name: "check", Module: "generate.wasm", GPU: true, GPUMemory: llmGPUMemory(),
		Args: []string{"check", "-n", "16", oracle[0].Prompt}})
	rep := runLLMCases(t, env, cases, launchFallbackArgs())
	requireGPU(t, rep)
	for i, o := range oracle {
		r := rep.Results[i]
		t.Logf("%s (%.0f ms): exit %d\n%s%s events %q", r.Name, r.Ms, r.Exit, r.Stdout, r.Stderr, r.Events)
		if r.Exit != 0 {
			t.Errorf("%s: exit %d (%s)", r.Name, r.Exit, r.Error)
			continue
		}
		if p := intsAfter(r.Stdout, "prompt"); !slices.Equal(p, o.PromptIDs) {
			t.Errorf("%s: prompt ids %v, llama.cpp %v", r.Name, p, o.PromptIDs)
		}
		got := intsAfter(r.Stdout, "tokens")
		eq := 0
		for eq < len(got) && eq < len(o.Tokens) && got[eq] == o.Tokens[eq] {
			eq++
		}
		t.Logf("%s: %d of %d tokens equal to llama.cpp's", r.Name, eq, len(o.Tokens))
		if eq < len(o.Tokens) && eq < len(got) {
			gap := math.Inf(1)
			for _, c := range o.Steps[eq] {
				if c.ID == got[eq] {
					gap = o.Steps[eq][0].Logprob - c.Logprob
				}
			}
			t.Logf("%s: token %d differs (gpu %d, llama.cpp %d) at llama.cpp's own gap %.3f nats", r.Name, eq, got[eq], o.Tokens[eq], gap)
			if gap > llamaNearTie || eq < llamaMinEqual {
				t.Errorf("%s: divergence at token %d with gap %.3f (limit %.2f, at least %d equal tokens)", r.Name, eq, gap, llamaNearTie, llamaMinEqual)
			}
		}
	}
	c := rep.Results[len(oracle)]
	t.Logf("check (%.0f ms): exit %d\n%s%s", c.Ms, c.Exit, c.Stdout, c.Stderr)
	if c.Exit != 0 {
		t.Fatalf("check: exit %d (%s)", c.Exit, c.Error)
	}
	m := regexp.MustCompile(`relative ([0-9.e+-]+),`).FindStringSubmatch(c.Stdout)
	if m == nil {
		t.Fatalf("no logits line in %q", c.Stdout)
	}
	rel, _ := strconv.ParseFloat(m[1], 64)
	if rel > gpuCPULogitRel {
		t.Errorf("GPU logits differ from the CPU path by %.3e relative, tolerance %.0e", rel, gpuCPULogitRel)
	}
	if !strings.Contains(c.Stdout, "equal prefix: 16 of 16") {
		t.Errorf("the GPU and CPU greedy tokens differ")
	}
}

// TestLLMTrace prints where the residual stream of the GPU path first leaves
// the CPU path's (a debugging aid; it fails only if the container fails).
func TestLLMTrace(t *testing.T) {
	env := needLLM(t)
	rep := runLLMCases(t, env, []engineCase{{Name: "trace", Module: "generate.wasm", GPU: true, GPUMemory: llmGPUMemory(),
		Args: []string{"trace", "The capital of France is"}}}, launchFallbackArgs())
	r := rep.Results[0]
	t.Logf("trace (%.0f ms): exit %d\n%s%s", r.Ms, r.Exit, r.Stdout, r.Stderr)
	if r.Exit != 0 {
		t.Fatalf("exit %d (%s)", r.Exit, r.Error)
	}
}

// TestLLMShaders compiles every WGSL kernel of the llm package in the
// browser and fails on any compilation message of type error, at the
// workgroup sizes 64 and 32 (an adapter with a lower limit gets the latter).
func TestLLMShaders(t *testing.T) {
	repo := os.Getenv("LLM_REPO")
	if repo == "" {
		t.Skip("LLM_REPO is not set")
	}
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	srv := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		io.WriteString(w, "<!doctype html><title>wgsl</title>")
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
	opts.Args = append(strings.Fields(os.Getenv("GPU_E2E_ARGS")), launchFallbackArgs()...)
	b, err := pw.Chromium.Launch(opts)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	p, err := b.NewPage()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := p.Goto(fmt.Sprintf("http://localhost:%d/", ln.Addr().(*net.TCPAddr).Port)); err != nil {
		t.Fatal(err)
	}
	for _, wg := range []string{"64", "32"} {
		cmd := exec.Command("go", "run", "./cmd/wgsldump", wg)
		cmd.Dir = repo
		out, err := cmd.Output()
		if err != nil {
			t.Fatal(err)
		}
		var shaders map[string]string
		if err := json.Unmarshal(out, &shaders); err != nil {
			t.Fatal(err)
		}
		v, err := p.Evaluate(`async (shaders) => {
			const adapter = await navigator.gpu.requestAdapter();
			const device = await adapter.requestDevice();
			const res = {};
			for (const [n, code] of Object.entries(shaders)) {
				const m = device.createShaderModule({code});
				const info = await m.getCompilationInfo();
				res[n] = info.messages.map(x => x.type + " " + x.lineNum + ":" + x.linePos + " " + x.message);
			}
			return res;
		}`, anyMap(shaders))
		if err != nil {
			t.Fatal(err)
		}
		for n, msgs := range v.(map[string]any) {
			for _, m := range msgs.([]any) {
				t.Logf("wg %s %s: %v", wg, n, m)
				if strings.HasPrefix(m.(string), "error") {
					t.Errorf("wg %s kernel %s: %v", wg, n, m)
				}
			}
		}
	}
}

func anyMap(m map[string]string) map[string]any {
	out := make(map[string]any, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

// TestLLMBench: decode and prefill tokens per second of SmolLM2-135M Q4_0 in a
// container on the browser's GPU, interleaved with llama.cpp on one CPU core
// (llama-bench -t 1 -ngl 0, the go-pkgx bottle), 128-token prompt and 128
// generated tokens, at a 1-minute load of at most 8.
func TestLLMBench(t *testing.T) {
	env := needLLM(t)
	if os.Getenv("LLM_BENCH") == "" {
		t.Skip("LLM_BENCH is not set")
	}
	if load := loadAverage(); load > 8 {
		t.Skipf("1-minute load average %.1f above 8: not measuring", load)
	}
	bench, err := exec.LookPath("llama-bench")
	if err != nil {
		t.Skip("llama-bench not in PATH")
	}
	ver, _ := exec.Command("llama-cli", "--version").CombinedOutput()
	t.Logf("llama.cpp: %s", strings.TrimSpace(string(ver)))
	rounds := 5
	var gpuPre, gpuDec, cpuPre, cpuDec []float64
	maxLoad := 0.0
	for i := 0; i < rounds; i++ {
		rep := runLLMCases(t, env, []engineCase{{Name: "bench", Module: "generate.wasm", GPU: true, GPUMemory: llmGPUMemory(),
			Args: []string{"bench", "-rounds", "2", "-p", "128", "-n", "128", "-ctx", "512"}}}, launchFallbackArgs())
		r := rep.Results[0]
		if r.Exit != 0 {
			t.Fatalf("bench container: exit %d (%s)\n%s%s", r.Exit, r.Error, r.Stdout, r.Stderr)
		}
		re := regexp.MustCompile(`round \d+: prefill ([0-9.]+) tok/s, decode ([0-9.]+) tok/s`)
		for _, m := range re.FindAllStringSubmatch(r.Stdout, -1) {
			a, _ := strconv.ParseFloat(m[1], 64)
			b, _ := strconv.ParseFloat(m[2], 64)
			gpuPre, gpuDec = append(gpuPre, a), append(gpuDec, b)
		}
		out, err := exec.Command(bench, "-m", env.model, "-t", "1", "-ngl", "0", "-p", "128", "-n", "128", "-r", "1", "-o", "json").Output()
		if err != nil {
			t.Fatal(err)
		}
		var rows []struct {
			NPrompt int     `json:"n_prompt"`
			NGen    int     `json:"n_gen"`
			AvgTS   float64 `json:"avg_ts"`
		}
		if err := json.Unmarshal(out, &rows); err != nil {
			t.Fatalf("%v: %s", err, out)
		}
		for _, row := range rows {
			if row.NPrompt == 128 && row.NGen == 0 {
				cpuPre = append(cpuPre, row.AvgTS)
			}
			if row.NGen == 128 && row.NPrompt == 0 {
				cpuDec = append(cpuDec, row.AvgTS)
			}
		}
		maxLoad = math.Max(maxLoad, loadAverage())
		t.Logf("round %d: gpu prefill %v decode %v; llama.cpp prefill %v decode %v; load %.2f", i+1, gpuPre, gpuDec, cpuPre, cpuDec, loadAverage())
	}
	med := func(v []float64) float64 { s := slices.Clone(v); slices.Sort(s); return s[len(s)/2] }
	t.Logf("| | prefill tok/s (p50, min-max) | decode tok/s (p50, min-max) |\n|---|---|---|\n| container, WebGPU Apple GPU, f32 | %.1f (%.1f-%.1f) | %.1f (%.1f-%.1f) |\n| llama.cpp CPU -t 1 | %.1f (%.1f-%.1f) | %.1f (%.1f-%.1f) |\n| n | %d, %d | max 1-min load %.2f |",
		med(gpuPre), slices.Min(gpuPre), slices.Max(gpuPre), med(gpuDec), slices.Min(gpuDec), slices.Max(gpuDec),
		med(cpuPre), slices.Min(cpuPre), slices.Max(cpuPre), med(cpuDec), slices.Min(cpuDec), slices.Max(cpuDec), len(gpuPre), len(cpuPre), maxLoad)
}

// TestLLMDecodeBreakdown prints the per-token split of the decode time (guest
// recording against submit) at a short and a 128-token context.
func TestLLMDecodeBreakdown(t *testing.T) {
	env := needLLM(t)
	var cases []engineCase
	for _, p := range []string{"8", "128"} {
		cases = append(cases, engineCase{Name: "p" + p, Module: "generate.wasm", GPU: true, GPUMemory: llmGPUMemory(),
			Args: []string{"bench", "-rounds", "2", "-p", p, "-n", "64", "-ctx", "512"}})
	}
	rep := runLLMCases(t, env, cases, launchFallbackArgs())
	for _, r := range rep.Results {
		t.Logf("%s: exit %d\n%s%s", r.Name, r.Exit, r.Stdout, r.Stderr)
	}
}
