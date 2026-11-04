//go:build e2e

package e2e

import (
	"os"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"
)

// TestLLMAB: interleaved A/B of clones of go-wasmcontainers/llm (a clean clone
// of the previous tag against the change), the way the decode and prefill
// figures of docs/perf in that repository were measured. Needs LLM_MODELS and:
//
//	LLM_REPOS   comma-separated clones of go-wasmcontainers/llm to compare
//	LLM_ARGS    ";"-separated sets of `generate bench` flags (default "-p 128 -n 128"),
//	            e.g. "-p 128 -n 128 -batch 8;-p 128 -n 128 -tpr 4"
//	LLM_SKIPS   comma-separated kernel classes left out (profiling by bisection,
//	            `generate bench -skip`), "none" for the whole pass (default "none")
//	LLM_ROUNDS  interleaved rounds (default 5)
//
// Every container run is one round of each repository and flag set; it skips
// above a 1-minute load of 8. The result lines start with RESULT.
func TestLLMAB(t *testing.T) {
	env := needLLM(t)
	if os.Getenv("LLM_REPOS") == "" {
		t.Skip("LLM_REPOS is not set")
	}
	if load := loadAverage(); load > 8 {
		t.Skipf("1-minute load average %.1f above 8: not measuring", load)
	}
	repos := strings.Split(os.Getenv("LLM_REPOS"), ",")
	skips := strings.Split(os.Getenv("LLM_SKIPS"), ",")
	if os.Getenv("LLM_SKIPS") == "" {
		skips = []string{"none"}
	}
	argSets := strings.Split(os.Getenv("LLM_ARGS"), ";")
	if os.Getenv("LLM_ARGS") == "" {
		argSets = []string{"-p 128 -n 128"}
	}
	rounds, _ := strconv.Atoi(os.Getenv("LLM_ROUNDS"))
	if rounds == 0 {
		rounds = 5
	}
	pre := map[string][]float64{}
	dec := map[string][]float64{}
	re := regexp.MustCompile(`round \d+: prefill ([0-9.]+) tok/s, decode ([0-9.]+) tok/s`)
	var keys []string
	for i := 0; i < rounds; i++ {
		for _, r := range repos {
			for _, sk := range skips {
				for _, as := range argSets {
					e := env
					e.repo = r
					a := append([]string{"bench", "-rounds", "2", "-ctx", "512"}, strings.Fields(as)...)
					if sk != "" && sk != "none" {
						a = append(a, "-skip", sk)
					}
					rep := runLLMCases(t, e, []engineCase{{Name: "bench", Module: "generate.wasm", GPU: true, GPUMemory: 512 << 20, Args: a}}, launchFallbackArgs())
					res := rep.Results[0]
					if res.Exit != 0 {
						t.Fatalf("exit %d %s\n%s%s", res.Exit, res.Error, res.Stdout, res.Stderr)
					}
					k := r[strings.LastIndex(r, "/")+1:] + " skip=" + sk + " [" + as + "]"
					if i == 0 {
						keys = append(keys, k)
					}
					for _, m := range re.FindAllStringSubmatch(res.Stdout, -1) {
						x, _ := strconv.ParseFloat(m[1], 64)
						y, _ := strconv.ParseFloat(m[2], 64)
						pre[k], dec[k] = append(pre[k], x), append(dec[k], y)
					}
				}
			}
		}
		t.Logf("round %d load %.2f", i+1, loadAverage())
	}
	med := func(v []float64) float64 { s := slices.Clone(v); slices.Sort(s); return s[len(s)/2] }
	for _, k := range keys {
		t.Logf("RESULT %s: prefill p50 %.1f (%.1f-%.1f) decode p50 %.1f (%.1f-%.1f) ms/tok %.2f n=%d", k, med(pre[k]), slices.Min(pre[k]), slices.Max(pre[k]), med(dec[k]), slices.Min(dec[k]), slices.Max(dec[k]), 1000/med(dec[k]), len(dec[k]))
	}
}

// TestLLMCheckLong: GPU against the Go CPU path on a prompt of about 190
// tokens (several prompt chunks and several attention chunks, which the short
// oracle prompts do not reach).
func TestLLMCheckLong(t *testing.T) {
	env := needLLM(t)
	prompt := strings.Repeat("The quick brown fox jumps over the lazy dog while the old clock keeps ticking. ", 11)
	rep := runLLMCases(t, env, []engineCase{{Name: "check", Module: "generate.wasm", GPU: true, GPUMemory: 512 << 20,
		Args: []string{"check", "-n", "16", "-ctx", "512", prompt}}}, launchFallbackArgs())
	res := rep.Results[0]
	t.Logf("exit %d\n%s%s", res.Exit, res.Stdout, res.Stderr)
	if res.Exit != 0 || !strings.Contains(res.Stdout, "equal prefix: 16 of 16") {
		t.Fatal("GPU and CPU paths differ on a long prompt")
	}
}
