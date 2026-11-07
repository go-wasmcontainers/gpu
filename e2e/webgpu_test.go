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
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
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
w.onmessage = (e) => {
  if (typeof e.data === "string") window.__report = e.data;
  else if (e.data && e.data.fail && window.__report === undefined) window.__report = JSON.stringify({error: e.data.fail + "\n" + (window.__log || []).join("\n")});
};
// A worker error must not replace a report that already arrived: a device's lost
// promise can resolve after the Go program exited (Firefox), which throws "Go
// program has already exited" late.
w.onerror = (e) => { if (window.__report === undefined) window.__report = JSON.stringify({error: "worker: " + e.message + "\n" + (window.__log || []).join("\n")}); };
// The worker's lines also go to this page's console: an engine that captures
// the console (go-webengine's JSLog) then shows how far a silent run got.
w.addEventListener("message", (e) => { if (e.data && e.data.log) { (window.__log = window.__log || []).push(e.data.log); console.log("worker: " + e.data.log); } });
</script>`

const worker = `importScripts("wasm_exec.js");
// the Go program's stderr (a fatal error) reaches the page: the Firefox and Safari lanes have no console
const oldLog = console.log;
console.log = (...a) => { postMessage({log: a.join(" ")}); oldLog(...a); };
const go = new Go();
// Milestones, and a failed fetch or compile reported instead of swallowed.
// A failure is posted as {fail}: the page takes it as the report only if the
// program's own report has not arrived (a late rejection must not replace it).
const t0 = performance.now(), ms = () => Math.round(performance.now() - t0) + " ms";
console.log("fetching webtest.wasm");
WebAssembly.instantiateStreaming(fetch("webtest.wasm"), go.importObject)
  .then((r) => { console.log("instantiated after " + ms() + ", running"); return go.run(r.instance); })
  .catch((e) => { console.log("failed after " + ms() + ": " + e + "\n" + (e && e.stack)); postMessage({fail: "worker: " + e}); });`

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
	for name, body := range map[string][]byte{"index.html": []byte(page + reportPoster), "worker.js": []byte(worker), "wasm_exec.js": wasmExec, "cases.json": cs} {
		if err := os.WriteFile(filepath.Join(site, name), body, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	mux := http.NewServeMux()
	mux.Handle("/data/", http.StripPrefix("/data/", http.FileServer(http.Dir(root))))
	mux.Handle("/", http.FileServer(http.Dir(site)))
	port, reports := serve(t, mux)
	v, version := fetchReport(t, port, reports, allowFallback(), launchFallbackArgs(), 5*time.Minute)
	var rep report
	if err := json.Unmarshal([]byte(v), &rep); err != nil {
		t.Fatal(err)
	}
	if rep.Error != "" {
		t.Fatalf("worker: %s", rep.Error)
	}
	a := rep.Adapter
	what := "GPU"
	if a.Fallback || strings.Contains(strings.ToLower(a.Architecture+a.Vendor), "swiftshader") {
		what = "FALLBACK (software, not a GPU)"
		if os.Getenv("GPU_E2E_ALLOW_FALLBACK") != "1" {
			t.Fatalf("adapter %s %s is a fallback adapter: this run cannot prove the GPU path", a.Vendor, a.Architecture)
		}
	}
	t.Logf("browser %s, adapter %s %s: %s", version, a.Vendor, a.Architecture, what)
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
