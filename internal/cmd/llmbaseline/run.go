package main

import (
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/mxschmitt/playwright-go"
)

// Config is one row of the table.
type Config struct {
	Name string
	Kind string // "llamacpp-cpu", "llamacpp-metal" or "webllm"
	GGUF string // path of the GGUF file (llama.cpp kinds)
	// WebLLM: model id (a directory under <corpus>/mlc-ai/), model library
	// file under <corpus>/libs/, and whether the library needs shader-f16.
	MLCID, Lib string
	F16        bool
}

// Env is everything a round needs.
type Env struct {
	Corpus      string
	BenchCPU    string // llama-bench built without Metal and BLAS
	BenchMetal  string
	PP, TG      int
	MaxLoad     float64
	MaxRedo     int
	Wait        time.Duration
	Channel     string // Playwright channel for the browser ("chromium")
	Out         io.Writer
	loadFn      func(context.Context) (float64, error)
	sleep       func(time.Duration)
	runLlama    func(ctx context.Context, e *Env, c Config) (Sample, error)
	runWebLLM   func(ctx context.Context, e *Env, c Config) (Sample, error)
	execOutput  func(ctx context.Context, name string, args ...string) ([]byte, error)
	prompt      string
	srvAddr     string
	pw          *playwright.Playwright
	browserInfo string
}

func defaultExec(ctx context.Context, name string, args ...string) ([]byte, error) {
	var out strings.Builder
	c := exec.CommandContext(ctx, name, args...)
	c.Stdout = &out
	c.Stderr = os.Stderr
	err := c.Run()
	return []byte(out.String()), err
}

// Interleave runs rounds: in each round every configuration once, the
// starting configuration rotating so that none is always first. A run whose
// 1-minute load is above MaxLoad is not taken: the round waits and checks again.
func Interleave(ctx context.Context, e *Env, configs []Config, rounds int) ([]Sample, error) {
	var all []Sample
	for r := 0; r < rounds; r++ {
		for i := range configs {
			c := configs[(i+r)%len(configs)]
			var load float64
			for redo := 0; ; redo++ {
				l, err := e.loadFn(ctx)
				if err != nil {
					return all, err
				}
				if l <= e.MaxLoad {
					load = l
					break
				}
				if redo >= e.MaxRedo {
					return all, fmt.Errorf("round %d, %s: 1-minute load %.2f stayed above %.1f for %d checks: deferred", r+1, c.Name, l, e.MaxLoad, redo)
				}
				fmt.Fprintf(e.Out, "load %.2f > %.1f, waiting before %s\n", l, e.MaxLoad, c.Name)
				e.sleep(e.Wait)
			}
			var s Sample
			var err error
			if c.Kind == "webllm" {
				s, err = e.runWebLLM(ctx, e, c)
			} else {
				s, err = e.runLlama(ctx, e, c)
			}
			if err != nil {
				return all, fmt.Errorf("round %d, %s: %w", r+1, c.Name, err)
			}
			s.Config, s.Round, s.Load1 = c.Name, r+1, load
			all = append(all, s)
			fmt.Fprintf(e.Out, "round %d %-34s load %.2f  pp %8.1f  tg %7.1f tok/s %s\n", r+1, c.Name, load, s.PP, s.TG, s.Adapter)
		}
	}
	return all, nil
}

func llamaRun(ctx context.Context, e *Env, c Config) (Sample, error) {
	bin, ngl, threads := e.BenchCPU, "0", "1"
	if c.Kind == "llamacpp-metal" {
		bin, ngl = e.BenchMetal, "99"
	}
	out, err := e.execOutput(ctx, bin, "-m", c.GGUF, "-t", threads, "-ngl", ngl,
		"-p", fmt.Sprint(e.PP), "-n", fmt.Sprint(e.TG), "-r", "1", "-o", "json")
	if err != nil {
		return Sample{}, err
	}
	pp, ppN, tg, tgN, err := ParseBench(out)
	return Sample{PP: pp, TG: tg, PPToks: ppN, TGToks: tgN}, err
}

// serve exposes the corpus to the page: the WebLLM bundle, the model libraries
// and the MLC repositories laid out as WebLLM fetches them
// (<model>/resolve/main/<file>). Nothing else is reachable.
func serve(corpus string) (addr string, stop func(), err error) {
	mux := http.NewServeMux()
	file := func(prefix, root string) {
		mux.HandleFunc(prefix, func(w http.ResponseWriter, r *http.Request) {
			rel := strings.TrimPrefix(r.URL.Path, prefix)
			rel = strings.Replace(rel, "/resolve/main/", "/", 1)
			if !filepath.IsLocal(rel) {
				http.NotFound(w, r)
				return
			}
			http.ServeFile(w, r, filepath.Join(root, filepath.FromSlash(rel)))
		})
	}
	file("/mlc/", filepath.Join(corpus, "mlc-ai"))
	file("/lib/", filepath.Join(corpus, "libs"))
	mux.HandleFunc("/webllm.js", func(w http.ResponseWriter, r *http.Request) {
		http.ServeFile(w, r, filepath.Join(corpus, "webllm", "pkg", "package", "lib", "index.js"))
	})
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		io.WriteString(w, pageHTML)
	})
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return "", nil, err
	}
	srv := &http.Server{Handler: mux}
	go srv.Serve(ln)
	return fmt.Sprintf("http://localhost:%d", ln.Addr().(*net.TCPAddr).Port), func() { srv.Close() }, nil
}

// webllmResult is what the page returns.
type webllmResult struct {
	Adapter struct {
		Vendor, Architecture, Description string
		Fallback                          bool
	}
	F16              bool
	PromptTokens     int     `json:"promptTokens"`
	CompletionTokens int     `json:"completionTokens"`
	PrefillTPS       float64 `json:"prefillTps"`
	DecodeTPS        float64 `json:"decodeTps"`
	Error            string
}

// webllmRun launches a fresh browser for the round (a cold start of the
// engine, from a warmed disk cache), loads the model, warms up with one short
// generation, then measures one pp/tg generation. It refuses a fallback
// adapter and any adapter whose vendor is not "apple".
func webllmRun(ctx context.Context, e *Env, c Config) (Sample, error) {
	opts := playwright.BrowserTypeLaunchOptions{Headless: playwright.Bool(true), Channel: playwright.String(e.Channel)}
	b, err := e.pw.Chromium.Launch(opts)
	if err != nil {
		return Sample{}, err
	}
	defer b.Close()
	e.browserInfo = b.Version()
	p, err := b.NewPage()
	if err != nil {
		return Sample{}, err
	}
	if _, err := p.Goto(e.srvAddr + "/"); err != nil {
		return Sample{}, err
	}
	arg := map[string]any{"id": c.MLCID, "lib": c.Lib, "f16": c.F16, "pp": e.PP, "tg": e.TG, "prompt": e.prompt}
	v, err := p.Evaluate(`(a) => window.runCase(a)`, arg)
	if err != nil {
		return Sample{}, err
	}
	m, ok := v.(map[string]any)
	if !ok {
		return Sample{}, fmt.Errorf("page returned %T", v)
	}
	var r webllmResult
	if err := decodeMap(m, &r); err != nil {
		return Sample{}, err
	}
	if r.Error != "" {
		return Sample{}, fmt.Errorf("page: %s", r.Error)
	}
	a := r.Adapter
	desc := fmt.Sprintf("%s %s", a.Vendor, a.Architecture)
	if err := CheckAdapter(a.Vendor, a.Architecture, a.Fallback); err != nil {
		return Sample{}, err
	}
	return Sample{PP: r.PrefillTPS, TG: r.DecodeTPS, PPToks: r.PromptTokens, TGToks: r.CompletionTokens, Adapter: desc}, nil
}

// CheckAdapter accepts the Apple GPU and nothing else: a fallback adapter or a
// software one (SwiftShader) never counts as a GPU.
func CheckAdapter(vendor, arch string, fallback bool) error {
	if fallback {
		return fmt.Errorf("adapter %q %q is a fallback adapter", vendor, arch)
	}
	if strings.Contains(strings.ToLower(vendor+arch), "swiftshader") || strings.ToLower(vendor) != "apple" {
		return fmt.Errorf("adapter vendor %q architecture %q is not the Apple GPU", vendor, arch)
	}
	return nil
}
