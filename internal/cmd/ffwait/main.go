// Command ffwait is the control for go-wasmcontainers/gpu#25: a standalone
// page (no engine, no container) that times each await of a WebGPU
// submit + read-back, in the page and in a Worker, and prints the page's JSON
// report. It starts the real Firefox binary headed on a fresh profile (as the
// e2e Firefox lane does) or, with --no-launch, only serves the page so any
// other browser can be pointed at the printed URL.
//
//	go run ./internal/cmd/ffwait --out /path/report.json
package main

import (
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/spf13/cobra"
)

var osExit = os.Exit

func main() {
	if err := RootCmd().Execute(); err != nil {
		osExit(1)
	}
}

// pageHTML is served with COOP/COEP so SharedArrayBuffer and Atomics.wait work.
// The query string selects the variant list; the report is POSTed to /report.
const pageHTML = `<!doctype html><meta charset=utf-8><title>ffwait</title><pre id=o>running</pre><script>
const SRC = ` + "`" + workJS + "`" + `;
(async () => {
  const q = new URLSearchParams(location.search);
  const out = { ua: navigator.userAgent, page: null, worker: null, workerBlocked: null };
  const log = (m) => { document.getElementById("o").textContent = m; };
  try {
    log("page"); out.page = await (0, eval)("(" + SRC + ")")({n: +(q.get("n")||12), ctx: "page"});
    log("worker");
    out.worker = await new Promise((res) => {
      const w = new Worker(URL.createObjectURL(new Blob(["onmessage=async(e)=>{postMessage(await (" + SRC + ")(e.data))}"], {type: "text/javascript"})));
      w.onmessage = (e) => res(e.data); w.onerror = (e) => res({error: String(e.message)});
      w.postMessage({n: +(q.get("n")||12), ctx: "worker"});
    });
  } catch (e) { out.error = String(e); }
  await fetch("/report", {method: "POST", body: JSON.stringify(out)});
  log("done");
})();
</script>`

// workJS runs in the page and in a Worker: every variant times its awaits
// with performance.now(). Each row is the median of n runs, in ms, plus the
// min and max. wallPhase is the completion time modulo 200 ms (a polling
// timer would show a constant phase).
const workJS = `async function work(cfg) {
  const out = { ctx: cfg.ctx, rows: {} };
  const a = await navigator.gpu.requestAdapter();
  if (!a) return { error: "no adapter" };
  const d = await a.requestDevice();
  const now = () => performance.now();
  const med = (xs) => { const s = xs.slice().sort((p, q) => p - q); return s[s.length >> 1]; };
  const rec = (name, rows) => {
    const keys = Object.keys(rows[0]); const r = {};
    for (const k of keys) { const v = rows.map((x) => x[k]); r[k] = {med: +med(v).toFixed(3), min: +Math.min(...v).toFixed(3), max: +Math.max(...v).toFixed(3)}; }
    out.rows[name] = r;
  };
  const size = 4096;
  const src = d.createBuffer({size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST});
  const stage = d.createBuffer({size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  const copyCmd = () => { const e = d.createCommandEncoder(); e.copyBufferToBuffer(src, 0, stage, 0, size); return e.finish(); };
  const mc = new MessageChannel(); const pend = []; mc.port1.onmessage = () => pend.shift()();
  const yieldNow = () => new Promise((r) => { pend.push(r); mc.port2.postMessage(0); });
  const kicked = async (q, p, k) => {
    let done = false; const fin = () => { done = true; }; p.then(fin, fin);
    let gap = 0.25, last = now();
    while (!done) { await yieldNow(); if (done) break; if (now() - last >= gap) { q.submit([]); k.n++; last = now(); gap = Math.min(gap * 2, 8); } }
    return p;
  };
  const variants = {
    // submit, then map the staging buffer (what the host does today minus the OSWD).
    map_only: async () => { const t0 = now(); d.queue.submit([copyCmd()]); const t1 = now(); await stage.mapAsync(GPUMapMode.READ); const t2 = now(); new Uint8Array(stage.getMappedRange()).slice(); const t3 = now(); stage.unmap(); return {submit: t1 - t0, map: t2 - t1, total: t3 - t0}; },
    // onSubmittedWorkDone first, then map.
    oswd_then_map: async () => { const t0 = now(); d.queue.submit([copyCmd()]); await d.queue.onSubmittedWorkDone(); const t1 = now(); await stage.mapAsync(GPUMapMode.READ); const t2 = now(); stage.unmap(); return {oswd: t1 - t0, map: t2 - t1, total: t2 - t0}; },
    // onSubmittedWorkDone alone.
    oswd_only: async () => { const t0 = now(); d.queue.submit([copyCmd()]); await d.queue.onSubmittedWorkDone(); return {total: now() - t0}; },
    // empty submit, onSubmittedWorkDone.
    empty_oswd: async () => { const t0 = now(); d.queue.submit([]); await d.queue.onSubmittedWorkDone(); return {total: now() - t0}; },
    // mapAsync issued BEFORE the submit that writes the buffer.
    map_before_submit: async () => { const t0 = now(); const p = stage.mapAsync(GPUMapMode.READ); d.queue.submit([copyCmd()]); await p; const t1 = now(); stage.unmap(); return {total: t1 - t0}; },
    // map a buffer nothing wrote (no submit at all).
    map_idle: async () => { const t0 = now(); await stage.mapAsync(GPUMapMode.READ); const t1 = now(); stage.unmap(); return {total: t1 - t0}; },
    // writeBuffer + submit + map, size 4 KiB.
    write_map: async () => { const t0 = now(); d.queue.writeBuffer(src, 0, new Uint8Array(size)); d.queue.submit([copyCmd()]); await stage.mapAsync(GPUMapMode.READ); const t1 = now(); stage.unmap(); return {total: t1 - t0}; },
    // the same map_only while a 1 ms timer ticks (is the event loop kept awake?).
    map_with_ticker: async () => { let on = true; (async () => { while (on) await wait(1); })(); const t0 = now(); d.queue.submit([copyCmd()]); await stage.mapAsync(GPUMapMode.READ); const t1 = now(); stage.unmap(); on = false; return {total: t1 - t0}; },
    // wgpu-core's queue.submit ends with a non-blocking maintain(Poll): an empty
    // submit a moment after the work may deliver the map callback early.
    kick_empty_submit: async () => { const t0 = now(); d.queue.submit([copyCmd()]); let done = false, k = 0; const p = stage.mapAsync(GPUMapMode.READ).then(() => { done = true; }); while (!done) { await wait(0); if (!done) { d.queue.submit([]); k++; } } await p; const t1 = now(); stage.unmap(); return {total: t1 - t0, kicks: k}; },
    kick_write_buffer: async () => { const t0 = now(); d.queue.submit([copyCmd()]); let done = false, k = 0; const p = stage.mapAsync(GPUMapMode.READ).then(() => { done = true; }); while (!done) { await wait(0); if (!done) { d.queue.writeBuffer(src, 0, new Uint8Array(4)); k++; } } await p; const t1 = now(); stage.unmap(); return {total: t1 - t0, kicks: k}; },
    kick_oswd: async () => { const t0 = now(); d.queue.submit([copyCmd()]); let done = false, k = 0; const p = d.queue.onSubmittedWorkDone().then(() => { done = true; }); while (!done) { await wait(0); if (!done) { d.queue.submit([]); k++; } } await p; const t1 = now(); return {total: t1 - t0, kicks: k}; },
    // the candidate fix: kick with an empty submit after 0.25 ms, then at doubling intervals up to 8 ms.
    kicked_map: async () => { const t0 = now(); d.queue.submit([copyCmd()]); const k = {n: 0}; await kicked(d.queue, stage.mapAsync(GPUMapMode.READ), k); const t1 = now(); stage.unmap(); return {total: t1 - t0, kicks: k.n}; },
    kicked_oswd_then_map: async () => { const t0 = now(); d.queue.submit([copyCmd()]); const k = {n: 0}; await kicked(d.queue, d.queue.onSubmittedWorkDone(), k); await kicked(d.queue, stage.mapAsync(GPUMapMode.READ), k); const t1 = now(); stage.unmap(); return {total: t1 - t0, kicks: k.n}; },
    // two submits in flight, one map.
    two_submits: async () => { const t0 = now(); d.queue.submit([copyCmd()]); d.queue.submit([]); await stage.mapAsync(GPUMapMode.READ); const t1 = now(); stage.unmap(); return {total: t1 - t0}; },
  };
  const only = cfg.only;
  for (const [name, fn] of Object.entries(variants)) {
    if (only && !only.includes(name)) continue;
    const rows = []; const phases = [];
    for (let i = 0; i < cfg.n + 2; i++) { const r = await fn(); phases.push(+((performance.timeOrigin + now()) % 200).toFixed(1)); if (i >= 2) rows.push(r); }
    rec(name, rows); out.rows[name].phase200 = phases.slice(2);
  }
  return out;
}`

// RootCmd is the command, for tests.
func RootCmd() *cobra.Command {
	var outPath, query string
	var noLaunch bool
	var timeout time.Duration
	cmd := &cobra.Command{
		Use:          "ffwait [--out file] [--no-launch]",
		Short:        "Time each await of a WebGPU submit and read-back in a bare page and a Worker",
		SilenceUsage: true,
		Args:         cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			ln, err := net.Listen("tcp", "127.0.0.1:0")
			if err != nil {
				return err
			}
			reports := make(chan []byte, 1)
			mux := http.NewServeMux()
			mux.HandleFunc("/", func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "text/html; charset=utf-8")
				w.Header().Set("Cross-Origin-Opener-Policy", "same-origin")
				w.Header().Set("Cross-Origin-Embedder-Policy", "require-corp")
				fmt.Fprint(w, pageHTML)
			})
			mux.HandleFunc("/report", func(w http.ResponseWriter, r *http.Request) {
				b, _ := io.ReadAll(io.LimitReader(r.Body, 1<<20))
				select {
				case reports <- b:
				default:
				}
			})
			srv := &http.Server{Handler: mux}
			go srv.Serve(ln)
			defer srv.Close()
			url := fmt.Sprintf("http://localhost:%d/?%s", ln.Addr().(*net.TCPAddr).Port, query)
			if noLaunch {
				fmt.Fprintln(cmd.ErrOrStderr(), "open", url)
			} else if stop, err := launchFirefox(url); err != nil {
				return err
			} else {
				defer stop()
			}
			select {
			case b := <-reports:
				if outPath != "" {
					return os.WriteFile(outPath, b, 0o644)
				}
				_, err := cmd.OutOrStdout().Write(append(b, '\n'))
				return err
			case <-time.After(timeout):
				return fmt.Errorf("no report in %v", timeout)
			}
		},
	}
	cmd.Flags().StringVar(&outPath, "out", "", "write the report here instead of stdout")
	cmd.Flags().StringVar(&query, "query", "n=12", "page query string")
	cmd.Flags().BoolVar(&noLaunch, "no-launch", false, "serve only; open the printed URL in any browser")
	cmd.Flags().DurationVar(&timeout, "timeout", 3*time.Minute, "wait this long for the report")
	return cmd
}

// launchFirefox starts the macOS Firefox (FFWAIT_EXECUTABLE overrides) headed
// on a fresh profile with WebGPU on; stop kills it by PID.
func launchFirefox(url string) (func(), error) {
	exe := os.Getenv("FFWAIT_EXECUTABLE")
	if exe == "" {
		exe = "/Applications/Firefox.app/Contents/MacOS/firefox"
	}
	base, err := os.UserConfigDir()
	if err != nil {
		return nil, err
	}
	base = filepath.Join(base, "ffwait")
	if err := os.MkdirAll(base, 0o755); err != nil {
		return nil, err
	}
	profile, err := os.MkdirTemp(base, "profile-")
	if err != nil {
		return nil, err
	}
	prefs := "user_pref(\"dom.webgpu.enabled\", true);\n" + os.Getenv("FFWAIT_PREFS") +
		"\nuser_pref(\"browser.shell.checkDefaultBrowser\", false);\nuser_pref(\"app.update.auto\", false);\n"
	if err := os.WriteFile(filepath.Join(profile, "user.js"), []byte(prefs), 0o644); err != nil {
		return nil, err
	}
	cmd := exec.Command(exe, "--no-remote", "--profile", profile, url)
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return func() { cmd.Process.Kill(); cmd.Wait(); os.RemoveAll(profile) }, nil
}
