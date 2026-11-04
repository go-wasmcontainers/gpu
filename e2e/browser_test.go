//go:build e2e

package e2e

import (
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/mxschmitt/playwright-go"
)

// browserName is "chromium" (default) or "firefox" (GPU_E2E_BROWSER). The
// Firefox lane runs the same tests; it fails, never skips, when the browser
// offers no WebGPU or only a fallback adapter.
func browserName() string {
	if v := os.Getenv("GPU_E2E_BROWSER"); v != "" {
		return v
	}
	return "chromium"
}

// reportPoster is appended to every e2e page: it posts window.__report to the
// test server, which is how the Firefox lane reads it (Playwright's own
// Firefox build does not start on every macOS release, so the lane drives the
// real Firefox binary and needs no automation protocol).
const reportPoster = `<script>
const __poll = setInterval(() => {
  if (window.__report === undefined) return;
  clearInterval(__poll);
  // go-webengine: the report goes through the console and the done signal
  // (Engine.EnableDoneSignal); elsewhere, through a POST.
  if (typeof __webengineDone === "function") {
    console.log("__REPORT__" + window.__report);
    __webengineDone(window.__report);
    return;
  }
  fetch("/__report", {method: "POST", body: window.__report});
}, 200);
</script>`

// serve starts the page server on a free port; reports receives what the
// page posts to /__report.
func serve(t *testing.T, h http.Handler) (port int, reports <-chan string) {
	t.Helper()
	ch := make(chan string, 1)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	srv := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/__report" && r.Method == http.MethodPost {
			b, _ := io.ReadAll(http.MaxBytesReader(w, r.Body, 64<<20))
			select {
			case ch <- string(b):
			default:
			}
			return
		}
		if r.URL.Path == "/__stream-worker.js" {
			w.Header().Set("Content-Type", "text/javascript")
			w.Header().Set("Cross-Origin-Resource-Policy", "same-origin")
			w.Header().Set("Cross-Origin-Embedder-Policy", "require-corp")
			io.WriteString(w, "let i = 0; (function tick() { postMessage(i++); if (i < 100) setTimeout(tick, 5); })();")
			return
		}
		if r.URL.Path == "/__sab-worker.js" {
			w.Header().Set("Content-Type", "text/javascript")
			w.Header().Set("Cross-Origin-Resource-Policy", "same-origin")
			w.Header().Set("Cross-Origin-Embedder-Policy", "require-corp")
			io.WriteString(w, `onmessage = (e) => { const d = e.data; const sab = d instanceof SharedArrayBuffer; const a = sab ? new Uint8Array(d) : d; a[0] = 42; postMessage((sab || a.buffer instanceof SharedArrayBuffer) ? "shared" : "copied"); };`)
			return
		}
		if r.URL.Path == "/__probe.html" {
			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			w.Header().Set("Cross-Origin-Opener-Policy", "same-origin")
			w.Header().Set("Cross-Origin-Embedder-Policy", "require-corp")
			io.WriteString(w, probePage+reportPoster)
			return
		}
		h.ServeHTTP(w, r)
	})}
	go srv.Serve(ln)
	t.Cleanup(func() { srv.Close() })
	return ln.Addr().(*net.TCPAddr).Port, ch
}

// fetchReport opens /index.html<query> in the browser under test and returns
// the page's window.__report and the browser's version. launch is extra
// Chromium flags (ignored by Firefox).
func fetchReport(t *testing.T, port int, reports <-chan string, query string, launch []string, timeout time.Duration) (report, version string) {
	t.Helper()
	return fetchPage(t, port, reports, "/index.html"+query, launch, timeout)
}

// fetchPage is fetchReport for any path of the page server.
func fetchPage(t *testing.T, port int, reports <-chan string, path string, launch []string, timeout time.Duration) (report, version string) {
	t.Helper()
	url := fmt.Sprintf("http://localhost:%d%s", port, path)
	switch browserName() {
	case "webengine":
		return runWebengine(t, port, reports, url, timeout)
	case "firefox":
		return runFirefox(t, url, reports, timeout)
	case "safari":
		return runSafari(t, url, reports, timeout)
	}
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
	if exe := os.Getenv("GPU_E2E_EXECUTABLE"); exe != "" {
		opts.ExecutablePath, opts.Channel = playwright.String(exe), nil
	}
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
	if _, err := p.Goto(url); err != nil {
		t.Fatal(err)
	}
	if _, err := p.WaitForFunction("() => window.__report !== undefined", nil, playwright.PageWaitForFunctionOptions{Timeout: playwright.Float(float64(timeout.Milliseconds()))}); err != nil {
		t.Fatalf("%v\nconsole: %s", err, strings.Join(console, "\n"))
	}
	v, err := p.Evaluate("window.__report")
	if err != nil {
		t.Fatal(err)
	}
	return v.(string), "chromium " + b.Version()
}

// runFirefox starts the Firefox binary (GPU_E2E_EXECUTABLE, default the macOS
// application) headless on a fresh profile with WebGPU on, and waits for the
// page to post its report. The browser is killed by PID at the end.
func runFirefox(t *testing.T, url string, reports <-chan string, timeout time.Duration) (string, string) {
	t.Helper()
	exe := os.Getenv("GPU_E2E_EXECUTABLE")
	if exe == "" {
		exe = "/Applications/Firefox.app/Contents/MacOS/firefox"
	}
	ver, err := exec.Command(exe, "--version").Output()
	if err != nil {
		t.Fatalf("firefox %s: %v", exe, err)
	}
	profile := t.TempDir()
	// dom.webgpu.enabled: WebGPU ships on in Firefox 141 (Windows), 145
	// (macOS Tahoe, Apple silicon), 147 (older macOS, Apple silicon); the pref
	// makes the lane independent of the rollout's own switch. COOP/COEP pages
	// need no other pref.
	prefs := `user_pref("dom.webgpu.enabled", true);
` + os.Getenv("GPU_E2E_FIREFOX_PREFS") + `
user_pref("browser.shell.checkDefaultBrowser", false);
user_pref("datareporting.policy.dataSubmissionEnabled", false);
user_pref("app.update.auto", false);
user_pref("app.update.enabled", false);
`
	if err := os.WriteFile(filepath.Join(profile, "user.js"), []byte(prefs), 0o644); err != nil {
		t.Fatal(err)
	}
	// Measured 2026-11-03, Firefox 159 on macOS 27 / Apple M4 Max: --headless
	// answers requestAdapter with null (no adapter); a normal window gets the
	// Apple GPU. So the lane is headed; GPU_E2E_HEADLESS=1 shows the null.
	args := []string{"--no-remote", "--profile", profile, url}
	if os.Getenv("GPU_E2E_HEADLESS") == "1" {
		args = append([]string{"--headless"}, args...)
	}
	cmd := exec.Command(exe, args...)
	logf, _ := os.Create(filepath.Join(profile, "stdout.log"))
	cmd.Stdout, cmd.Stderr = logf, logf
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() { cmd.Process.Kill(); cmd.Wait() }()
	select {
	case r := <-reports:
		return r, strings.TrimSpace(string(ver))
	case <-time.After(timeout):
		b, _ := os.ReadFile(filepath.Join(profile, "stdout.log"))
		t.Fatalf("%s: no report in %v (no WebGPU, or the page hung)\nbrowser output: %.2000s", ver, timeout, b)
	}
	return "", ""
}
