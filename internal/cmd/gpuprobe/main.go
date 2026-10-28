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

const page = `<!doctype html><meta charset=utf-8><script>
` + probeJS + `
window.probeWorker = () => new Promise((res) => {
  const src = probe.toString() + "; probe().then(r => postMessage(r));";
  const w = new Worker(URL.createObjectURL(new Blob([src], {type: "text/javascript"})));
  w.onmessage = (e) => res(e.data);
  w.onerror = (e) => res({ hasGPU: false, error: String(e.message) });
});
</script>`

// RootCmd is the command, for tests.
func RootCmd() *cobra.Command {
	var browser, args, channel string
	var headed bool
	cmd := &cobra.Command{
		Use:          "gpuprobe [--browser chromium|firefox|webkit]",
		Short:        "Report the WebGPU adapter a headless browser offers",
		SilenceUsage: true,
		Args:         cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			rep, err := run(browser, args, channel, headed)
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
	return cmd
}

func run(browser, args, channel string, headed bool) (*Report, error) {
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
	return rep, nil
}
