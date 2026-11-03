//go:build e2e

package e2e

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"
)

// probePage reports each capability the e2e pages need, separately, so that
// a browser (or engine) that lacks some is told precisely what.
const probePage = `<!doctype html><meta charset=utf-8><title>capabilities</title><script>
(async () => {
  const caps = [];
  const add = (name, ok, detail) => caps.push({name, ok: !!ok, detail: detail || ""});
  const t = (name, f) => { try { add(name, f()); } catch (e) { add(name, false, String(e)); } };
  t("Worker", () => typeof Worker === "function");
  t("SharedArrayBuffer", () => typeof SharedArrayBuffer === "function");
  t("Atomics.wait", () => typeof Atomics === "object" && typeof Atomics.wait === "function");
  t("crossOriginIsolated (COOP/COEP)", () => self.crossOriginIsolated === true);
  t("fetch", () => typeof fetch === "function");
  t("WebAssembly", () => typeof WebAssembly === "object" && typeof WebAssembly.instantiate === "function");
  t("WebAssembly.instantiateStreaming", () => typeof WebAssembly === "object" && typeof WebAssembly.instantiateStreaming === "function");
  t("OPFS (navigator.storage.getDirectory)", () => typeof navigator.storage === "object" && typeof navigator.storage.getDirectory === "function");
  t("OffscreenCanvas", () => typeof OffscreenCanvas === "function");
  t("navigator.gpu", () => typeof navigator.gpu === "object" && navigator.gpu !== null);
  try {
    const m = await WebAssembly.instantiate(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0]));
    add("WebAssembly.instantiate runs", !!m);
  } catch (e) { add("WebAssembly.instantiate runs", false, String(e)); }
  try {
    // A software adapter counts here: the probe lists capabilities, the
    // other tests refuse a fallback adapter.
    let a = await navigator.gpu.requestAdapter();
    if (!a) a = await navigator.gpu.requestAdapter({forceFallbackAdapter: true});
    add("navigator.gpu.requestAdapter returns an adapter", a !== null && a !== undefined);
  } catch (e) { add("navigator.gpu.requestAdapter returns an adapter", false, String(e)); }
  window.__report = JSON.stringify({caps});
})();
</script>`

type capReport struct {
	Caps []struct {
		Name   string `json:"name"`
		OK     bool   `json:"ok"`
		Detail string `json:"detail"`
	} `json:"caps"`
}

func (c capReport) missing() []string {
	var m []string
	for _, x := range c.Caps {
		if !x.OK {
			m = append(m, "no "+x.Name)
		}
	}
	return m
}

type engineRun struct {
	Report  string   `json:"report"`
	Log     []string `json:"log"`
	Title   string   `json:"title"`
	Error   string   `json:"error"`
	Version string   `json:"version"`
}

// renderInEngine runs e2e/webengine (its own module, pinned to
// go-webengine/engine v0.25.0) on url.
func renderInEngine(t *testing.T, url string, timeout time.Duration) engineRun {
	t.Helper()
	cmd := exec.Command("go", "run", ".", "-url", url, "-timeout", timeout.String())
	cmd.Dir = "webengine"
	var stderr strings.Builder
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("go-webengine driver: %v\n%s", err, stderr.String())
	}
	var r engineRun
	if err := json.Unmarshal(out, &r); err != nil {
		t.Fatalf("go-webengine driver output: %v\n%.500s", err, out)
	}
	return r
}

// runWebengine is the go-webengine lane (GPU_E2E_BROWSER=webengine). It first
// renders the capability probe and FAILS naming every capability the engine
// lacks ("no Worker", "no navigator.gpu", ...): each go-webengine milestone
// then shows up as a capability turning green. It never skips. If the engine
// has them all, it renders the real page and returns its report.
func runWebengine(t *testing.T, port int, reports <-chan string, url string, timeout time.Duration) (string, string) {
	t.Helper()
	probeURL := fmt.Sprintf("http://localhost:%d/__probe.html", port)
	probing := url == probeURL
	pr := renderInEngine(t, probeURL, 60*time.Second)
	if pr.Error != "" || pr.Report == "" {
		t.Fatalf("%s: the capability probe did not report: error %q, log %q", pr.Version, pr.Error, pr.Log)
	}
	if probing {
		return pr.Report, pr.Version
	}
	var caps capReport
	if err := json.Unmarshal([]byte(pr.Report), &caps); err != nil {
		t.Fatal(err)
	}
	if m := caps.missing(); len(m) > 0 {
		t.Fatalf("%s lacks: %s", pr.Version, strings.Join(m, ", "))
	}
	r := renderInEngine(t, url, timeout)
	if r.Error != "" {
		t.Fatalf("%s: %s\nconsole: %q", r.Version, r.Error, r.Log)
	}
	if r.Report == "" {
		t.Fatalf("%s: the page did not call __webengineDone within %v\nconsole: %q", r.Version, timeout, r.Log)
	}
	return r.Report, r.Version
}

// TestBrowserCapabilities lists, one by one, what the browser under test (or
// go-webengine) offers of what the e2e pages need. It fails on any missing
// capability, naming each; the CI job for go-webengine publishes that list.
func TestBrowserCapabilities(t *testing.T) {
	port, reports := serve(t, nopHandler{})
	v, version := fetchPage(t, port, reports, "/__probe.html", launchFallbackArgs(), 2*time.Minute)
	var caps capReport
	if err := json.Unmarshal([]byte(v), &caps); err != nil {
		t.Fatal(err)
	}
	t.Logf("%s", version)
	var md strings.Builder
	fmt.Fprintf(&md, "### Capabilities of %s\n\n| capability | offered |\n|---|---|\n", version)
	for _, c := range caps.Caps {
		state := "yes"
		if !c.OK {
			state = "NO"
		}
		t.Logf("capability %-52s %s %s", c.Name, state, c.Detail)
		fmt.Fprintf(&md, "| %s | %s |\n", c.Name, state)
	}
	if m := caps.missing(); len(m) > 0 {
		fmt.Fprintf(&md, "\nMissing: %s\n", strings.Join(m, ", "))
	}
	// The CI job publishes the table in the run's summary.
	if p := os.Getenv("GITHUB_STEP_SUMMARY"); p != "" {
		if f, err := os.OpenFile(p, os.O_APPEND|os.O_WRONLY|os.O_CREATE, 0o644); err == nil {
			f.WriteString(md.String())
			f.Close()
		}
	}
	if m := caps.missing(); len(m) > 0 {
		t.Errorf("%s lacks: %s", version, strings.Join(m, ", "))
	}
}
