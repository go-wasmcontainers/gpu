// Command gpuprobe asks a real browser, through Playwright, what WebGPU it
// offers a page: whether navigator.gpu exists (in a window and in a
// worker), and the adapter's vendor, architecture and whether it is a
// fallback (software) adapter. CI uses it to say what its browser lanes
// can and cannot prove; a fallback adapter is never reported as a GPU.
//
//	go run ./internal/cmd/gpuprobe --browser chromium
package main

import (
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"os"
	"strings"

	"github.com/playwright-community/playwright-go"
	"github.com/spf13/cobra"
)

var osExit = os.Exit

func main() {
	if err := RootCmd().Execute(); err != nil {
		osExit(1)
	}
}

// Report is what the browser said.
type Report struct {
	Browser  string `json:"browser"`
	Window   Probe  `json:"window"`
	Worker   Probe  `json:"worker"`
	UserArgs string `json:"args,omitempty"`
	// Canvas (--canvas): one row per alpha mode; pixels are RGBA, except
	// GPU, which is the canvas texture's bytes in Format order.
	Canvas []CanvasProbe `json:"canvas,omitempty"`
}

// CanvasProbe is a WebGPU OffscreenCanvas's clear colour (0.2, 0.4, 0.6, 1)
// seen at each step to the page.
type CanvasProbe struct {
	AlphaMode string `json:"alphaMode,omitempty"`
	Format    string `json:"format,omitempty"`
	GPU       []int  `json:"gpu,omitempty"`      // copyTextureToBuffer of the canvas texture
	Worker2D  []int  `json:"worker2d,omitempty"` // the transferred ImageBitmap drawn in the worker
	Page2D    []int  `json:"page2d,omitempty"`   // the same bitmap drawn in the page
	Error     string `json:"error,omitempty"`
}

// Probe is one context's answer.
type Probe struct {
	HasGPU       bool   `json:"hasGPU"`
	Adapter      bool   `json:"adapter"`
	Vendor       string `json:"vendor,omitempty"`
	Architecture string `json:"architecture,omitempty"`
	Fallback     bool   `json:"fallback"`
	Error        string `json:"error,omitempty"`
}

// probeJS runs in the page and in a worker it starts.
const probeJS = `async function probe() {
  const r = { hasGPU: !!(self.navigator && navigator.gpu), adapter: false, fallback: false };
  if (!r.hasGPU) return r;
  try {
    const a = await navigator.gpu.requestAdapter();
    if (!a) return r;
    r.adapter = true;
    const info = a.info || {};
    r.vendor = info.vendor || ""; r.architecture = info.architecture || "";
    r.fallback = !!(info.isFallbackAdapter || a.isFallbackAdapter);
  } catch (e) { r.error = String(e); }
  return r;
}`

// canvasJS runs in a worker: for each alpha mode it renders a known clear
// colour into a WebGPU OffscreenCanvas, reads the canvas texture back on
// the GPU (copyTextureToBuffer: did the GPU write the pixels?), transfers
// the canvas to an ImageBitmap and draws it into a 2D OffscreenCanvas (is
// the bitmap readable in the worker?), then posts the bitmap to the page,
// which draws it again (is it readable where the dashboard draws it?).
const canvasJS = `async function canvasProbe() {
  const out = [];
  const a = await navigator.gpu.requestAdapter();
  if (!a) return [{error: "no adapter"}];
  const d = await a.requestDevice();
  const format = navigator.gpu.getPreferredCanvasFormat();
  const clear = {r: 0.2, g: 0.4, b: 0.6, a: 1};
  for (const alphaMode of ["premultiplied", "opaque"]) {
    const r = {alphaMode, format};
    try {
      d.pushErrorScope("validation");
      const c = new OffscreenCanvas(8, 8);
      const ctx = c.getContext("webgpu");
      ctx.configure({device: d, format, alphaMode,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC});
      const tex = ctx.getCurrentTexture();
      const buf = d.createBuffer({size: 256 * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
      const enc = d.createCommandEncoder();
      const pass = enc.beginRenderPass({colorAttachments: [{view: tex.createView(), clearValue: clear, loadOp: "clear", storeOp: "store"}]});
      pass.end();
      enc.copyTextureToBuffer({texture: tex}, {buffer: buf, bytesPerRow: 256}, [8, 8]);
      d.queue.submit([enc.finish()]);
      await buf.mapAsync(GPUMapMode.READ);
      r.gpu = Array.from(new Uint8Array(buf.getMappedRange(), 4 * (4 + 4 * 64), 4));
      buf.unmap();
      const bm = c.transferToImageBitmap();
      const w2 = new OffscreenCanvas(8, 8).getContext("2d");
      w2.drawImage(bm, 0, 0);
      r.worker2d = Array.from(w2.getImageData(4, 4, 1, 1).data);
      r.bitmap = bm;
      const err = await d.popErrorScope();
      if (err) r.error = err.message;
    } catch (e) { r.error = String(e); }
    out.push(r);
  }
  return out;
}`

const page = `<!doctype html><meta charset=utf-8><script>
` + probeJS + `
` + canvasJS + `
window.probeWorker = () => new Promise((res) => {
  const src = probe.toString() + "; probe().then(r => postMessage(r));";
  const w = new Worker(URL.createObjectURL(new Blob([src], {type: "text/javascript"})));
  w.onmessage = (e) => res(e.data);
  w.onerror = (e) => res({ hasGPU: false, error: String(e.message) });
});
window.probeCanvas = () => new Promise((res) => {
  const src = canvasProbe.toString() + "; canvasProbe().then(rs => postMessage(rs, rs.filter(r => r.bitmap).map(r => r.bitmap)));";
  const w = new Worker(URL.createObjectURL(new Blob([src], {type: "text/javascript"})));
  w.onmessage = (e) => res(e.data.map((r) => {
    if (r.bitmap) {
      const g = document.createElement("canvas"); g.width = g.height = 8;
      const x = g.getContext("2d", {willReadFrequently: true});
      x.drawImage(r.bitmap, 0, 0);
      r.page2d = Array.from(x.getImageData(4, 4, 1, 1).data);
      r.bitmap.close(); delete r.bitmap;
    }
    return r;
  }));
  w.onerror = (e) => res([{ error: String(e.message) }]);
});
</script>`

// RootCmd is the command, for tests.
func RootCmd() *cobra.Command {
	var browser, args, channel string
	var headed, canvas bool
	cmd := &cobra.Command{
		Use:          "gpuprobe [--browser chromium|firefox|webkit]",
		Short:        "Report the WebGPU adapter a headless browser offers",
		SilenceUsage: true,
		Args:         cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			rep, err := run(browser, args, channel, headed, canvas)
			if err != nil {
				return err
			}
			enc := json.NewEncoder(cmd.OutOrStdout())
			enc.SetIndent("", "  ")
			return enc.Encode(rep)
		},
	}
	cmd.Flags().StringVar(&browser, "browser", "chromium", "chromium, firefox or webkit")
	cmd.Flags().StringVar(&args, "args", "", "extra browser arguments, space-separated (e.g. --enable-unsafe-webgpu)")
	cmd.Flags().StringVar(&channel, "channel", "", `browser channel ("chromium": the full browser in new headless mode, not the headless shell)`)
	cmd.Flags().BoolVar(&headed, "headed", false, "show the browser window")
	cmd.Flags().BoolVar(&canvas, "canvas", false, "also follow a WebGPU OffscreenCanvas frame from the GPU to the page")
	return cmd
}

func run(browser, args, channel string, headed, canvas bool) (*Report, error) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, err
	}
	srv := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		fmt.Fprint(w, page)
	})}
	go srv.Serve(ln)
	defer srv.Close()

	pw, err := playwright.Run()
	if err != nil {
		return nil, err
	}
	defer pw.Stop()
	var bt playwright.BrowserType
	switch browser {
	case "chromium":
		bt = pw.Chromium
	case "firefox":
		bt = pw.Firefox
	case "webkit":
		bt = pw.WebKit
	default:
		return nil, fmt.Errorf("unknown browser %q", browser)
	}
	opts := playwright.BrowserTypeLaunchOptions{Headless: playwright.Bool(!headed)}
	if channel != "" {
		opts.Channel = playwright.String(channel)
	}
	if args != "" {
		opts.Args = strings.Fields(args)
	}
	b, err := bt.Launch(opts)
	if err != nil {
		return nil, err
	}
	defer b.Close()
	p, err := b.NewPage()
	if err != nil {
		return nil, err
	}
	// localhost is a secure context, which WebGPU requires.
	if _, err := p.Goto("http://localhost:" + fmt.Sprint(ln.Addr().(*net.TCPAddr).Port) + "/"); err != nil {
		return nil, err
	}
	rep := &Report{Browser: browser + " " + b.Version(), UserArgs: args}
	for _, c := range []struct {
		expr string
		dst  *Probe
	}{{"probe()", &rep.Window}, {"probeWorker()", &rep.Worker}} {
		v, err := p.Evaluate(c.expr)
		if err != nil {
			c.dst.Error = err.Error()
			continue
		}
		b, _ := json.Marshal(v)
		if err := json.Unmarshal(b, c.dst); err != nil {
			c.dst.Error = err.Error()
		}
	}
	if canvas && rep.Worker.Adapter {
		v, err := p.Evaluate("probeCanvas()")
		if err != nil {
			rep.Canvas = []CanvasProbe{{Error: err.Error()}}
			return rep, nil
		}
		b, _ := json.Marshal(v)
		if err := json.Unmarshal(b, &rep.Canvas); err != nil {
			rep.Canvas = []CanvasProbe{{Error: err.Error()}}
		}
	}
	return rep, nil
}
