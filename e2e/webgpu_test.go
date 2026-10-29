//go:build e2e

// Package e2e checks the WebGPU paths in real browsers, through
// Playwright: go test -tags e2e ./e2e/
//
// The browser needs Playwright's driver (PLAYWRIGHT_DRIVER_PATH) and
// Chromium. GPU_E2E_CHANNEL picks the Chromium channel ("chromium", the
// full browser in new headless mode, is the default: the headless shell
// offers no GPU adapter). A fallback (software) adapter fails the test
// unless GPU_E2E_ALLOW_FALLBACK=1, and then the report says the run
// proved the WGSL against SwiftShader, not on a GPU.
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
	"time"

	"github.com/mxschmitt/playwright-go"
)

type caseSpec struct {
	Name    string   `json:"name"`
	Model   string   `json:"model"`
	Inputs  []string `json:"inputs"`
	Outputs []string `json:"outputs"`
}

type report struct {
	Adapter struct {
		Vendor, Architecture string
		Fallback             bool
	} `json:"adapter"`
	Results []struct {
		Name   string  `json:"name"`
		OK     bool    `json:"ok"`
		Error  string  `json:"error"`
		MaxErr float64 `json:"maxErr"`
		Ms     float64 `json:"ms"`
	} `json:"results"`
	Error string `json:"error"`
}

const page = `<!doctype html><meta charset=utf-8><title>webgpu e2e</title><script>
const w = new Worker("worker.js" + location.search);
w.onmessage = (e) => { window.__report = e.data; };
w.onerror = (e) => { window.__report = JSON.stringify({error: "worker: " + e.message}); };
</script>`

const worker = `importScripts("wasm_exec.js");
const go = new Go();
WebAssembly.instantiateStreaming(fetch("webtest.wasm"), go.importObject).then((r) => go.run(r.instance));`

// cases lists the vendored node tests and MNIST-12, served under /data.
func cases(t *testing.T, root string) []caseSpec {
	dirs, err := filepath.Glob(filepath.Join(root, "onnx/testdata/node/test_*"))
	if err != nil || len(dirs) < 100 {
		t.Fatalf("%d node tests: %v", len(dirs), err)
	}
	var out []caseSpec
	add := func(name, model, set string) {
		c := caseSpec{Name: name, Model: model}
		for i := 0; ; i++ {
			p := filepath.Join(root, set, fmt.Sprintf("input_%d.pb", i))
			if _, err := os.Stat(p); err != nil {
				break
			}
			c.Inputs = append(c.Inputs, "data/"+filepath.ToSlash(filepath.Join(set, fmt.Sprintf("input_%d.pb", i))))
		}
		for i := 0; ; i++ {
			p := filepath.Join(root, set, fmt.Sprintf("output_%d.pb", i))
			if _, err := os.Stat(p); err != nil {
				break
			}
			c.Outputs = append(c.Outputs, "data/"+filepath.ToSlash(filepath.Join(set, fmt.Sprintf("output_%d.pb", i))))
		}
		out = append(out, c)
	}
	for _, d := range dirs {
		rel, _ := filepath.Rel(root, d)
		add(filepath.Base(d), "data/"+filepath.ToSlash(rel)+"/model.onnx", filepath.Join(rel, "test_data_set_0"))
	}
	add("mnist-12", "data/onnx/testdata/models/mnist-12/mnist-12.onnx", "onnx/testdata/models/mnist-12/test_data_set_0")
	return out
}

func TestWebGPUBackend(t *testing.T) {
	root, err := filepath.Abs("..")
	if err != nil {
		t.Fatal(err)
	}
	site := t.TempDir()
	build := exec.Command("go", "build", "-o", filepath.Join(site, "webtest.wasm"), "./internal/cmd/webtest")
	build.Dir = root
	build.Env = append(os.Environ(), "GOOS=js", "GOARCH=wasm", "CGO_ENABLED=0")
	if b, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build: %v\n%s", err, b)
	}
	wasmExec, err := os.ReadFile(filepath.Join(runtime.GOROOT(), "lib", "wasm", "wasm_exec.js"))
	if err != nil {
		t.Fatal(err)
	}
	cs, _ := json.Marshal(cases(t, root))
	for name, body := range map[string][]byte{"index.html": []byte(page), "worker.js": []byte(worker), "wasm_exec.js": wasmExec, "cases.json": cs} {
		if err := os.WriteFile(filepath.Join(site, name), body, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	mux := http.NewServeMux()
	mux.Handle("/data/", http.StripPrefix("/data/", http.FileServer(http.Dir(root))))
	mux.Handle("/", http.FileServer(http.Dir(site)))
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	srv := &http.Server{Handler: mux}
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
	if allowFallback {
		opts.Args = []string{"--enable-unsafe-webgpu"}
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
	q := ""
	if allowFallback {
		q = "?fallback=1"
	}
	url := fmt.Sprintf("http://localhost:%d/index.html%s", ln.Addr().(*net.TCPAddr).Port, q)
	if _, err := p.Goto(url); err != nil {
		t.Fatal(err)
	}
	if _, err := p.WaitForFunction("() => window.__report !== undefined", nil, playwright.PageWaitForFunctionOptions{Timeout: playwright.Float(300000)}); err != nil {
		t.Fatal(err)
	}
	v, err := p.Evaluate("window.__report")
	if err != nil {
		t.Fatal(err)
	}
	var rep report
	if err := json.Unmarshal([]byte(v.(string)), &rep); err != nil {
		t.Fatal(err)
	}
	if rep.Error != "" {
		t.Fatalf("worker: %s", rep.Error)
	}
	a := rep.Adapter
	what := "GPU"
	if a.Fallback || strings.Contains(strings.ToLower(a.Architecture+a.Vendor), "swiftshader") {
		what = "FALLBACK (software, not a GPU)"
		if !allowFallback {
			t.Fatalf("adapter %s %s is a fallback adapter: this run cannot prove the GPU path", a.Vendor, a.Architecture)
		}
	}
	t.Logf("browser %s, adapter %s %s: %s", b.Version(), a.Vendor, a.Architecture, what)
	pass, total := 0, 0.0
	for _, r := range rep.Results {
		total += r.Ms
		if r.OK {
			pass++
			continue
		}
		t.Errorf("%s: %s", r.Name, r.Error)
	}
	t.Logf("%d/%d cases pass on the %s; compute time %.1f ms in all", pass, len(rep.Results), what, total)
	for _, r := range rep.Results {
		if r.Name == "mnist-12" {
			t.Logf("mnist-12: compute %.2f ms, max |error| %.2g", r.Ms, r.MaxErr)
		}
	}
	_ = time.Second
}
