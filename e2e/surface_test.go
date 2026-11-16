//go:build e2e

package e2e

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/go-wasmcontainers/engine/runtime/browser"
)

// TestSurfaceFrames runs the triangle container (60 frames on a wasi-gfx
// surface) in every lane's browser, Firefox included, and checks the frames the
// page received: 60 of them, in order, pixels drawn, and changing. A surface
// frame is configure, getCurrentTexture, a draw, then transferToImageBitmap;
// Firefox drops a canvas context's configuration at that last step, so the
// second getCurrentTexture used to throw "Canvas not configured" and the
// container trapped at its second frame (go-wasmcontainers/playground#56;
// fixed in v0.14.1, the host configures the canvas again). The Chromium lane's
// TestContainersOnWebGPU cannot see this: Chromium keeps the configuration.
func TestSurfaceFrames(t *testing.T) {
	if b := browserName(); b == "webengine" || b == "safari" {
		t.Skip("Chromium and Firefox lanes: the other engines have their own surface coverage")
	}
	root, err := filepath.Abs("..")
	if err != nil {
		t.Fatal(err)
	}
	site := t.TempDir()
	build(t, root, filepath.Join(site, "enginetest.wasm"), "./internal/cmd/enginetest", "js")
	build(t, root, filepath.Join(site, "triangle.wasm"), "./examples/triangle", "wasip1")
	wasmExec, err := os.ReadFile(filepath.Join(runtime.GOROOT(), "lib", "wasm", "wasm_exec.js"))
	if err != nil {
		t.Fatal(err)
	}
	cs, _ := json.Marshal([]engineCase{{Name: "triangle", Module: "triangle.wasm", Args: []string{"60"}, GPU: true}})
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
	// The software-adapter lane (ubuntu, SwiftShader) lets the page accept it, as
	// TestContainersOnWebGPU does.
	query, launch := "", []string(nil)
	if os.Getenv("GPU_E2E_ALLOW_FALLBACK") == "1" {
		query, launch = "?fallback=1", []string{"--enable-unsafe-webgpu"}
	}
	v, version := fetchReport(t, port, reports, query, launch, 5*time.Minute)
	var rep struct {
		engineReport
		Surfaces surfaceLog `json:"surfaces"`
	}
	if err := json.Unmarshal([]byte(v), &rep); err != nil {
		t.Fatal(err)
	}
	if rep.Error != "" {
		t.Fatalf("%s: engine worker: %s", version, rep.Error)
	}
	if len(rep.Results) != 1 {
		t.Fatalf("%s: %d results", version, len(rep.Results))
	}
	r := rep.Results[0]
	t.Logf("%s: triangle exit %d (%s), stdout %q", version, r.Exit, r.Error, r.Stdout)
	if r.Exit != 0 || !strings.Contains(r.Stdout, "60 frames in") {
		t.Fatalf("%s: triangle exit %d (%s), stdout %q, stderr %q", version, r.Exit, r.Error, r.Stdout, r.Stderr)
	}
	fr := rep.Surfaces.Frames
	if len(fr) != 60 {
		t.Fatalf("%s: %d frames reached the page, want 60", version, len(fr))
	}
	sums := map[uint32]bool{}
	for i, f := range fr {
		if f.Seq != i+1 {
			t.Fatalf("frame %d has seq %d", i, f.Seq)
		}
		sums[f.Sum] = true
	}
	if c := fr[0].Center; c[0]+c[1]+c[2] < 200 {
		t.Errorf("centre pixel %v: no triangle drawn", c)
	}
	if len(sums) < 30 {
		t.Errorf("only %d distinct frames of 60: the triangle does not spin", len(sums))
	}
}
