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

	"github.com/mxschmitt/playwright-go"

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

// enginePage plays the dashboard's part for surfaces: it draws each frame
// into a canvas, samples it, closes the bitmap and acknowledges it.
const enginePage = `<!doctype html><meta charset=utf-8><title>engine e2e</title>
<canvas id=c width=320 height=240></canvas><script>
const w = new Worker("engine.js" + location.search);
const g = document.getElementById("c").getContext("2d", {willReadFrequently: true});
window.__surfaces = {events: [], frames: []};
w.onmessage = (e) => {
  const d = e.data;
  if (typeof d === "string") { window.__report = d; return; }
  if (d.t === "surface") { window.__surfaces.events.push({op: d.op, surface: d.surface, container: d.container, width: d.width, height: d.height}); return; }
  if (d.t === "frame") {
    g.drawImage(d.bitmap, 0, 0);
    const px = (x, y) => Array.from(g.getImageData(x, y, 1, 1).data);
    const all = g.getImageData(0, 0, d.width, d.height).data;
    let sum = 0;
    for (let i = 0; i < all.length; i += 7) sum = (sum + all[i] * (i % 251)) >>> 0;
    window.__surfaces.frames.push({surface: d.surface, seq: d.seq, t: performance.now(), w: d.width, h: d.height,
      center: px(d.width >> 1, d.height >> 1), corner: px(2, 2), sum});
    d.bitmap.close();
    w.postMessage({t: "frame-ack", surface: d.surface, seq: d.seq});
    // A click and a key on the pane, as the dashboard forwards them.
    if (d.seq === 5) {
      w.postMessage({t: "input", surface: d.surface, kind: "pointer-down", x: 10, y: 20});
      w.postMessage({t: "input", surface: d.surface, kind: "key-down", code: "KeyA", text: "A", shift: true});
    }
  }
};
// A worker error is recorded on its own: it must not replace a report that
// already arrived (as when a late message hit a Go program that had
// exited), and it fails the test either way.
w.onerror = (e) => {
  window.__workerErrors = (window.__workerErrors || []).concat(e.message);
  if (window.__report === undefined) window.__report = JSON.stringify({error: "worker: " + e.message});
};
window.__worker = w;
</script>`

type surfaceLog struct {
	Events []struct {
		Op            string `json:"op"`
		Surface       int    `json:"surface"`
		Container     string `json:"container"`
		Width, Height int
	} `json:"events"`
	Frames []struct {
		Surface int     `json:"surface"`
		Seq     int     `json:"seq"`
		T       float64 `json:"t"`
		W, H    int
		Center  []int  `json:"center"`
		Corner  []int  `json:"corner"`
		Sum     uint32 `json:"sum"`
	} `json:"frames"`
}

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
	build(t, root, filepath.Join(site, "triangle.wasm"), "./examples/triangle", "wasip1")
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
		{Name: "triangle", Module: "triangle.wasm", Args: []string{"60"}, GPU: true},
		// The page clicks and types on every surface at its 5th frame.
		{Name: "triangle-input", Module: "triangle.wasm", Args: []string{"40", "input"}, GPU: true},
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
	// GPU_E2E_ARGS adds browser flags, e.g. --use-webgpu-adapter=swiftshader
	// to reproduce the Linux lane on another machine.
	opts.Args = append(opts.Args, strings.Fields(os.Getenv("GPU_E2E_ARGS"))...)
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
	// The page keeps talking to the worker after the report (a frame
	// acknowledgement or an input event can still be in flight, as from a
	// dashboard): the worker must still answer, not throw "Go program has
	// already exited". A late message is sent on purpose.
	late, err := p.Evaluate(`new Promise((done) => {
	  window.__worker.postMessage({t: "frame-ack", surface: 1, seq: 1e6});
	  window.__worker.postMessage({t: "input", surface: 1, kind: "pointer-move", x: 1, y: 1});
	  setTimeout(() => done(JSON.stringify(window.__workerErrors || [])), 500);
	})`)
	if err != nil {
		t.Fatal(err)
	}
	if late != "[]" {
		t.Errorf("worker errors after its report: %v", late)
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
		"triangle":         {0, []string{"60 frames in"}},
		"triangle-input":   {0, []string{"pointer-down 10,20\n", "key-down key-a \"A\" shift=true\n", "40 frames in"}},
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
	checkSurfaces(t, p)
}

// checkSurfaces checks what the page (playing the dashboard) received:
// two surfaces (triangle, then triangle-input), each opened and closed and
// named after its container; for the first, 60 frames in order, the
// triangle over the clear colour, frames that differ (it spins), and a
// rate within the 60 per second pacing.
func checkSurfaces(t *testing.T, p playwright.Page) {
	v, err := p.Evaluate("JSON.stringify(window.__surfaces)")
	if err != nil {
		t.Fatal(err)
	}
	var all surfaceLog
	if err := json.Unmarshal([]byte(v.(string)), &all); err != nil {
		t.Fatal(err)
	}
	var ops []string
	for _, e := range all.Events {
		ops = append(ops, e.Op)
	}
	if strings.Join(ops, ",") != "open,close,open,close" || all.Events[0].Width != 320 || all.Events[0].Height != 240 {
		t.Fatalf("surface events %+v", all.Events)
	}
	// The open event names the container (the engine's wasi.Config.Label,
	// set from runtime.Spec.ID).
	if a, b := all.Events[0].Container, all.Events[2].Container; a != "c-triangle" || b != "c-triangle-input" {
		t.Errorf("open events name containers %q, %q", a, b)
	}
	// The rest is about the first surface.
	s := all
	s.Frames = nil
	for _, f := range all.Frames {
		if f.Surface == all.Events[0].Surface {
			s.Frames = append(s.Frames, f)
		}
	}
	if len(s.Frames) != 60 {
		t.Fatalf("%d frames, want 60", len(s.Frames))
	}
	sums := map[uint32]bool{}
	for i, f := range s.Frames {
		if f.Seq != i+1 {
			t.Fatalf("frame %d has seq %d", i, f.Seq)
		}
		sums[f.Sum] = true
	}
	first, last := s.Frames[0], s.Frames[59]
	fps := 59 / ((last.T - first.T) / 1000)
	t.Logf("triangle: 60 frames, %.1f per second at the page; corner %v, centre %v", fps, first.Corner, first.Center)
	if fps > 66 {
		t.Errorf("%.1f frames per second: pacing to 60 not enforced", fps)
	}
	// Every lane checks pixels. (Fully transparent frames mean the browser
	// could not back a WebGPU canvas: on headless Linux it needs Skia on
	// Vulkan, docs/DESIGN.md "Surfaces".)
	// Clear colour (0.05, 0.05, 0.1) in the corner, the triangle (its
	// colours are bright) in the middle.
	if c := first.Corner; c[0] > 30 || c[1] > 30 || c[2] > 45 || c[2] < 10 {
		t.Errorf("corner pixel %v is not the clear colour", c)
	}
	if c := first.Center; c[0]+c[1]+c[2] < 200 {
		t.Errorf("centre pixel %v: no triangle drawn", c)
	}
	if len(sums) < 30 {
		t.Errorf("only %d distinct frames of 60: the triangle does not spin", len(sums))
	}
}
