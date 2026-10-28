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
	"strings"
	"testing"

	"github.com/playwright-community/playwright-go"

	"github.com/go-wasmcontainers/engine/runtime/browser"
)

type engineCase struct {
	Name   string   `json:"name"`
	Module string   `json:"module"`
	Args   []string `json:"args"`
	GPU    bool     `json:"gpu"`
}

type engineReport struct {
	Adapter struct {
		Vendor, Architecture string
		Fallback             bool
	} `json:"adapter"`
	Results []struct {
		Name   string  `json:"name"`
		Exit   int     `json:"exit"`
		Stdout string  `json:"stdout"`
		Stderr string  `json:"stderr"`
		Error  string  `json:"error"`
		Ms     float64 `json:"ms"`
	} `json:"results"`
	Error string `json:"error"`
}

const enginePage = `<!doctype html><meta charset=utf-8><title>engine e2e</title><script>
const w = new Worker("engine.js" + location.search);
w.onmessage = (e) => { window.__report = e.data; };
w.onerror = (e) => { window.__report = JSON.stringify({error: "worker: " + e.message}); };
</script>`

const engineWorker = `importScripts("wasm_exec.js");
const go = new Go();
WebAssembly.instantiateStreaming(fetch("enginetest.wasm"), go.importObject).then((r) => go.run(r.instance));`

func build(t *testing.T, root, out, pkg, goos string) {
	t.Helper()
	cmd := exec.Command("go", "build", "-o", out, pkg)
	cmd.Dir = root
	cmd.Env = append(os.Environ(), "GOOS="+goos, "GOARCH=wasm", "CGO_ENABLED=0")
	if b, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build %s: %v\n%s", pkg, err, b)
	}
}

// TestContainersOnWebGPU runs real containers in the engine's browser
// executor with this module's host modules: examples/compute (wasi:webgpu:
// vector add and matmul checked against the CPU inside the guest) and
// examples/mnist (wasi-nn on the CPU and on the GPU), with and without the
// gpu capability.
func TestContainersOnWebGPU(t *testing.T) {
	root, err := filepath.Abs("..")
	if err != nil {
		t.Fatal(err)
	}
	site := t.TempDir()
	build(t, root, filepath.Join(site, "enginetest.wasm"), "./internal/cmd/enginetest", "js")
	build(t, root, filepath.Join(site, "compute.wasm"), "./examples/compute", "wasip1")
	build(t, root, filepath.Join(site, "mnist.wasm"), "./examples/mnist", "wasip1")
	wasmExec, err := os.ReadFile(filepath.Join(runtime.GOROOT(), "lib", "wasm", "wasm_exec.js"))
	if err != nil {
		t.Fatal(err)
	}
	cases := []engineCase{
		{Name: "compute", Module: "compute.wasm", GPU: true},
		{Name: "compute-no-gpu", Module: "compute.wasm"},
		{Name: "compute-quota", Module: "compute.wasm", Args: []string{"quota"}, GPU: true},
		{Name: "mnist-cpu", Module: "mnist.wasm"},
		{Name: "mnist-gpu", Module: "mnist.wasm", Args: []string{"gpu"}, GPU: true},
		{Name: "mnist-gpu-no-cap", Module: "mnist.wasm", Args: []string{"gpu"}},
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
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Shared memory (the SAB bridge) needs cross-origin isolation.
		w.Header().Set("Cross-Origin-Opener-Policy", "same-origin")
		w.Header().Set("Cross-Origin-Embedder-Policy", "require-corp")
		fs.ServeHTTP(w, r)
	})
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	srv := &http.Server{Handler: handler}
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
	allowFallback := os.Getenv("GPU_E2E_ALLOW_FALLBACK") == "1"
	q := ""
	if allowFallback {
		opts.Args = []string{"--enable-unsafe-webgpu"}
		q = "?fallback=1"
	}
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
	if _, err := p.Goto(fmt.Sprintf("http://localhost:%d/index.html%s", ln.Addr().(*net.TCPAddr).Port, q)); err != nil {
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
	a := rep.Adapter
	what := "GPU"
	if a.Fallback || strings.Contains(strings.ToLower(a.Architecture+a.Vendor), "swiftshader") {
		what = "FALLBACK (software, not a GPU)"
		if !allowFallback {
			t.Fatalf("adapter %s %s is a fallback adapter: this run cannot prove the GPU path", a.Vendor, a.Architecture)
		}
	}
	t.Logf("adapter %s %s: %s", a.Vendor, a.Architecture, what)
	want := map[string]struct {
		exit int
		out  []string
	}{
		"compute":          {0, []string{"vector add: 1048576 elements match the CPU", "matmul 256x256: 65536 elements match the CPU"}},
		"compute-no-gpu":   {2, []string{"no GPU adapter"}},
		"compute-quota":    {134, []string{"buffer 3\n"}}, // the 5th 64 MiB buffer traps (ExitTrap)
		"mnist-cpu":        {0, []string{"digit 7 (p=1.000)"}},
		"mnist-gpu":        {0, []string{"digit 7 (p=1.000)"}},
		"mnist-gpu-no-cap": {1, nil},
	}
	for _, r := range rep.Results {
		w := want[r.Name]
		ok := r.Exit == w.exit
		for _, s := range w.out {
			ok = ok && strings.Contains(r.Stdout, s)
		}
		switch r.Name {
		case "mnist-gpu-no-cap":
			ok = ok && strings.Contains(r.Stderr, "unsupported operation")
		case "compute-quota":
			ok = ok && strings.Contains(r.Error, "GPU memory quota") && !strings.Contains(r.Stdout, "buffer 4")
		}
		t.Logf("%s (%.0f ms): exit %d\n%s%s", r.Name, r.Ms, r.Exit, r.Stdout, r.Stderr)
		if !ok {
			t.Errorf("%s: exit %d (%s), stdout %q, stderr %q; want exit %d with %q", r.Name, r.Exit, r.Error, r.Stdout, r.Stderr, w.exit, w.out)
		}
	}
	if len(rep.Results) != len(cases) {
		t.Errorf("%d results for %d cases", len(rep.Results), len(cases))
	}
}
