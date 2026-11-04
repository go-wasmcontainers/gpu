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
  // After each item the results so far go to the console: an engine that
  // ends the page before the last item completes still yields a list.
  const add = (name, ok, detail) => { caps.push({name, ok: !!ok, detail: detail || ""}); console.log("__PARTIAL__" + JSON.stringify({caps})); };
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
  // SharedArrayBuffer details, each its own named case.
  t("Atomics.wait/notify exist and notify works", () => {
    const i = new Int32Array(new SharedArrayBuffer(16));
    return typeof Atomics.wait === "function" && typeof Atomics.notify === "function" && Atomics.notify(i, 0, 1) === 0 && Atomics.add(i, 1, 5) === 0 && Atomics.load(i, 1) === 5;
  });
  t("Uint8Array/Int32Array/DataView over the same SAB agree", () => {
    const sab = new SharedArrayBuffer(16), u8 = new Uint8Array(sab), i32 = new Int32Array(sab), dv = new DataView(sab);
    u8[0] = 1; u8[1] = 2; u8[2] = 3; u8[3] = 4;
    if (i32[0] !== 0x04030201) throw new Error("Int32Array read " + i32[0].toString(16));
    if (dv.getUint32(0, true) !== 0x04030201) throw new Error("DataView read " + dv.getUint32(0, true).toString(16));
    dv.setUint8(8, 9);
    if (u8[8] !== 9) throw new Error("write through DataView not seen by Uint8Array");
    return u8.buffer === sab && i32.buffer === sab && dv.buffer === sab;
  });
  // Posting a SAB (or a view of one) to a worker: it must arrive shared (the
  // worker's write is seen here), not copied. Bounded by a 15 s timer.
  const post = (what) => new Promise((resolve) => {
    if (typeof Worker !== "function" || typeof SharedArrayBuffer !== "function") return resolve("no Worker or SharedArrayBuffer");
    const sab = new SharedArrayBuffer(16), view = new Uint8Array(sab);
    let w;
    const timer = setTimeout(() => { try { w.terminate(); } catch (e) {} resolve("no reply from the worker in 15 s"); }, 15000);
    try {
      w = new Worker("/__sab-worker.js");
      w.onmessage = (e) => {
        clearTimeout(timer);
        try { w.terminate(); } catch (x) {}
        if (e.data !== "shared") return resolve("the worker received a copy (" + e.data + "), not shared memory");
        resolve(view[0] === 42 ? "" : "the worker wrote 42 but this side reads " + view[0]);
      };
      w.onerror = (e) => { clearTimeout(timer); resolve("worker error: " + (e && e.message)); };
      w.postMessage(what === "sab" ? sab : view);
    } catch (e) { clearTimeout(timer); resolve("postMessage failed: " + e); }
  });
  for (const [name, what] of [["posting a SharedArrayBuffer shares it", "sab"], ["posting a Uint8Array view of a SAB shares it", "view"]]) {
    const why = await post(what);
    add(name, why === "", why);
  }
  // A Worker streams 100 unsolicited messages (setTimeout-spaced) while the
  // main thread's setInterval counter runs: pass only if all arrive in order
  // through onmessage and the counter advanced meanwhile. Bounded by a 15 s
  // timer, so a failure names its reason instead of hanging.
  const stream = (nWorkers) => new Promise((resolve) => {
    if (typeof Worker !== "function") return resolve("no Worker");
    const url = "/__stream-worker.js"; // served by the test server
    let counter = 0, first = -1, last = -1, finished = false;
    const iv = setInterval(() => { counter++; }, 10);
    const got = [], workers = [];
    const finish = (why) => {
      if (finished) return;
      finished = true;
      clearInterval(iv);
      workers.forEach((w) => { try { w.terminate(); } catch (e) {} });
      resolve(why);
    };
    const timer = setTimeout(() => finish("only " + got.reduce((a, g) => a + g.length, 0) + "/" + (100 * nWorkers) + " messages in 15 s"), 15000);
    try {
      for (let k = 0; k < nWorkers; k++) {
        const w = new Worker(url), mine = [];
        got.push(mine);
        w.onmessage = (e) => {
          if (e.data !== mine.length) return finish("worker " + k + ": message " + mine.length + " arrived as " + e.data + " (out of order)");
          mine.push(e.data);
          if (first < 0) first = counter;
          if (got.every((g) => g.length === 100)) {
            last = counter;
            clearTimeout(timer);
            finish(last > first ? "" : "the main-thread counter did not advance while the workers streamed (blocked main thread)");
          }
        };
        w.onerror = (e) => finish("worker error: " + (e && e.message));
        workers.push(w);
      }
    } catch (e) { finish("new Worker failed: " + e); }
  });
  for (const [name, n] of [["worker streams unsolicited messages", 1], ["2 concurrent workers stream unsolicited messages", 2]]) {
    const why = await stream(n);
    add(name, why === "", why);
  }
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
	Partial string   `json:"partial"`
	Budget  []string `json:"budget"`
	Log     []string `json:"log"`
	Title   string   `json:"title"`
	Error   string   `json:"error"`
	Version string   `json:"version"`
}

// renderInEngine runs e2e/webengine (its own module, pinned to
// go-webengine/engine v0.28.1) on url.
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
	if pr.Report == "" && pr.Error == "" && pr.Partial != "" {
		// The engine ended the page before the probe's last items completed
		// (it does not deliver a worker's unsolicited messages, or keep the
		// page alive for them): report the items done, and name the rest.
		t.Logf("%s: no final report or __webengineDone from the probe (the engine stopped at its JSTimeout or earlier); using its partial results", pr.Version)
		var c capReport
		if err := json.Unmarshal([]byte(pr.Partial), &c); err != nil {
			t.Fatal(err)
		}
		have := map[string]bool{}
		for _, x := range c.Caps {
			have[x.Name] = true
		}
		for _, n := range []string{"worker streams unsolicited messages", "2 concurrent workers stream unsolicited messages"} {
			if !have[n] {
				c.Caps = append(c.Caps, struct {
					Name   string `json:"name"`
					OK     bool   `json:"ok"`
					Detail string `json:"detail"`
				}{n, false, "the engine ended the page before the worker's 100 messages arrived (no report, no error)"})
			}
		}
		b, _ := json.Marshal(c)
		pr.Report = string(b)
	}
	if len(pr.Budget) > 0 && pr.Report != "" {
		// The engine stopped work on a cap (go-webengine v0.28.1 logs it): a
		// named failure of its own.
		var c capReport
		if err := json.Unmarshal([]byte(pr.Report), &c); err != nil {
			t.Fatal(err)
		}
		c.Caps = append(c.Caps, struct {
			Name   string `json:"name"`
			OK     bool   `json:"ok"`
			Detail string `json:"detail"`
		}{"timer budget (engine caps not exhausted)", false, strings.Join(pr.Budget, "; ")})
		b, _ := json.Marshal(c)
		pr.Report = string(b)
	}
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
	if len(r.Budget) > 0 {
		t.Fatalf("%s: timer budget exhausted: %s", r.Version, strings.Join(r.Budget, "; "))
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
