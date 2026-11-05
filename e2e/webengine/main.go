// Command webengine renders one page in go-webengine's pure-Go engine and
// prints a JSON result: the page's report (the line "__REPORT__<json>" its
// script logged just before calling __webengineDone), the captured console
// log and any error. It is the Firefox/Safari/Chromium lanes' counterpart
// for the engine of the go-webengine org (GPU_E2E_BROWSER=webengine).
//
// It lives in its own module (e2e/webengine/go.mod) so that the gpu module's
// dependency graph does not grow.
//
//	webengine -url URL [-timeout 60s]
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/go-webengine/engine"
)

type result struct {
	Report  string `json:"report"`
	Partial string `json:"partial"`
	// Budget holds the engine's "script cap reached" / "timer-job cap
	// reached" log lines.
	Budget  []string `json:"budget"`
	Log     []string `json:"log"`
	Title   string   `json:"title"`
	Error   string   `json:"error"`
	Version string   `json:"version"`
}

const (
	reportPrefix  = "__REPORT__"
	partialPrefix = "__PARTIAL__" // the probe's results so far, logged after each item
)

func main() {
	url := flag.String("url", "", "page to render")
	timeout := flag.Duration("timeout", 60*time.Second, "script budget (JSTimeout)")
	flag.Parse()
	var res result
	res.Version = "go-webengine/engine v0.30.0"
	var mu sync.Mutex
	e := engine.New()
	e.EnableDoneSignal = true
	e.EnableWorkerStreaming = true // a live loop for a worker's unsolicited messages (v0.30.0)
	e.JSTimeout = *timeout
	e.MaxWasmModuleBytes = 512 << 20
	e.MaxWasmMemoryBytes = 2 << 30
	e.JSLog = func(l string) {
		mu.Lock()
		defer mu.Unlock()
		if strings.Contains(l, "cap reached") { // v0.30.0: maxScripts / maxTimerJobs stopped work
			res.Budget = append(res.Budget, l)
		}
		if i := strings.Index(l, partialPrefix); i >= 0 {
			res.Partial = l[i+len(partialPrefix):] // the last one wins
			return
		}
		if i := strings.Index(l, reportPrefix); i >= 0 && res.Report == "" {
			res.Report = l[i+len(reportPrefix):]
			return
		}
		res.Log = append(res.Log, l)
	}
	ctx, cancel := context.WithTimeout(context.Background(), *timeout+30*time.Second)
	defer cancel()
	doc, err := e.Fetch(ctx, *url)
	if err == nil {
		res.Title = doc.Title
		_, _, err = e.RenderDocument(ctx, doc, image.Rect(0, 0, 800, 600))
	}
	if err != nil {
		res.Error = err.Error()
	}
	mu.Lock()
	defer mu.Unlock()
	if err := json.NewEncoder(os.Stdout).Encode(res); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
