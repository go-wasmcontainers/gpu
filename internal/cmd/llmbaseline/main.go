// Command llmbaseline measures the reference speeds that in-container WebGPU
// inference is compared against (docs/llm-design.md, step 0): llama.cpp on one
// CPU thread (and Metal, as context), and WebLLM in Chrome for Testing on the
// Apple GPU, interleaved round by round with the load checked before each run.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/mxschmitt/playwright-go"
	"github.com/spf13/cobra"
)

var osExit = os.Exit

func main() {
	if err := RootCmd().Execute(); err != nil {
		osExit(1)
	}
}

// RootCmd is the command tree, driven by the tests through SetArgs.
func RootCmd() *cobra.Command {
	root := &cobra.Command{Use: "llmbaseline", Short: "step-0 baselines for in-container LLM inference", SilenceUsage: true}
	root.AddCommand(hfCmd(), urlCmd(), runCmd(), tableCmd())
	return root
}

func hfCmd() *cobra.Command {
	var dir string
	c := &cobra.Command{
		Use:   "hf REPO [FILE...]",
		Short: "download files of a Hugging Face repo at its current commit, verify them, print the pins as JSON",
		Args:  cobra.MinimumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			pins, err := FetchHF(cmd.Context(), dir, args[0], args[1:])
			if err != nil {
				return err
			}
			enc := json.NewEncoder(cmd.OutOrStdout())
			enc.SetIndent("", " ")
			return enc.Encode(pins)
		},
	}
	c.Flags().StringVar(&dir, "dir", "", "corpus directory (files go to DIR/REPO/PATH)")
	_ = c.MarkFlagRequired("dir")
	return c
}

func urlCmd() *cobra.Command {
	var out, sha string
	c := &cobra.Command{
		Use:   "url URL",
		Short: "download one file, print its size and sha256 (checked against --sha256 when given)",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			n, s, err := FetchURL(cmd.Context(), args[0], out, sha)
			if err != nil {
				return err
			}
			fmt.Fprintf(cmd.OutOrStdout(), "%d %s %s\n", n, s, out)
			return nil
		},
	}
	c.Flags().StringVar(&out, "out", "", "destination file")
	c.Flags().StringVar(&sha, "sha256", "", "expected sha256")
	_ = c.MarkFlagRequired("out")
	return c
}

// configs is the table's rows. The llama.cpp rows take the GGUF files, the
// WebLLM rows the MLC builds; SmolLM2-135M has MLC q4f16_1 and q4f32_1 builds
// outside WebLLM's prebuilt list, run here with their published libraries.
//
// The llama.cpp rows run the go-pkgx bottle (bottle) by default. A hand build
// is only ever an explicit comparison input (handCPU, handMetal): its rows are
// added, labelled "[hand]", and never replace the bottle's.
func configs(corpus, smolGGUF, qwenGGUF, bottle, handCPU, handMetal string, webllm bool) []Config {
	sm := filepath.Join(corpus, smolGGUF)
	qw := filepath.Join(corpus, qwenGGUF)
	models := []struct{ name, gguf string }{{"SmolLM2-135M", sm}, {"Qwen2.5-0.5B", qw}}
	var out []Config
	add := func(label, cpu, metal string) {
		if cpu == "" {
			return
		}
		for _, m := range models {
			out = append(out, Config{Name: m.name + " llama.cpp Q4_0 CPU -t 1 " + label, Kind: "llamacpp-cpu", GGUF: m.gguf, Bin: cpu})
		}
		for _, m := range models {
			out = append(out, Config{Name: m.name + " llama.cpp Q4_0 Metal (context) " + label, Kind: "llamacpp-metal", GGUF: m.gguf, Bin: metal})
		}
	}
	add("[bottle]", bottle, bottle)
	if handCPU != "" && handMetal != "" {
		add("[hand]", handCPU, handMetal)
	}
	if webllm {
		out = append(out,
			Config{Name: "SmolLM2-135M WebLLM q4f16_1", Kind: "webllm", MLCID: "SmolLM2-135M-Instruct-q4f16_1-MLC", Lib: "SmolLM2-135M-Instruct-q4f16_1_cs1k-webgpu.wasm", F16: true},
			Config{Name: "SmolLM2-135M WebLLM q4f32_1", Kind: "webllm", MLCID: "SmolLM2-135M-Instruct-q4f32_1-MLC", Lib: "SmolLM2-135M-Instruct-q4f32_1_cs1k-webgpu.wasm"},
			Config{Name: "Qwen2.5-0.5B WebLLM q4f16_1", Kind: "webllm", MLCID: "Qwen2.5-0.5B-Instruct-q4f16_1-MLC", Lib: "Qwen2-0.5B-Instruct-q4f16_1_cs1k-webgpu.wasm", F16: true},
			Config{Name: "Qwen2.5-0.5B WebLLM q4f32_1", Kind: "webllm", MLCID: "Qwen2.5-0.5B-Instruct-q4f32_1-MLC", Lib: "Qwen2-0.5B-Instruct-q4f32_1_cs1k-webgpu.wasm"})
	}
	return out
}

func uptimeLoad(ctx context.Context, exec func(context.Context, string, ...string) ([]byte, error)) (float64, error) {
	out, err := exec(ctx, "uptime")
	if err != nil {
		return 0, err
	}
	return ParseLoad(string(out))
}

func runCmd() *cobra.Command {
	var (
		corpus, bench, handCPU, handMetal, smol, qwen, out, only string
		rounds, pp, tg                                           int
		maxLoad                                                  float64
		wait                                                     time.Duration
		noWebLLM                                                 bool
	)
	c := &cobra.Command{
		Use:   "run",
		Short: "interleaved rounds over every configuration; writes the samples as JSON",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			if rounds < 5 {
				return fmt.Errorf("--rounds %d: a publishable table needs at least 5", rounds)
			}
			if (handCPU == "") != (handMetal == "") {
				return fmt.Errorf("--hand-cpu and --hand-metal go together")
			}
			path, err := exec.LookPath(bench)
			if err != nil {
				return fmt.Errorf("the go-pkgx bottle is not installed (pkgm install github.com/ggml-org/llama.cpp): %w", err)
			}
			e := &Env{Corpus: corpus, PP: pp, TG: tg,
				MaxLoad: maxLoad, MaxRedo: 40, Wait: wait, Channel: "chromium", Out: cmd.OutOrStdout(),
				execOutput: defaultExec, execOutputBoth: defaultExecBoth, sleep: time.Sleep, runLlama: llamaRun, runWebLLM: webllmRun,
				prompt: "the" + strings.Repeat(" the", pp-1)}
			e.loadFn = func(ctx context.Context) (float64, error) { return uptimeLoad(ctx, e.execOutput) }
			cfgs := configs(corpus, smol, qwen, path, handCPU, handMetal, !noWebLLM)
			if only != "" {
				var keep []Config
				for _, c := range cfgs {
					if strings.Contains(c.Name, only) {
						keep = append(keep, c)
					}
				}
				cfgs = keep
			}
			for _, c := range cfgs {
				if c.Kind == "llamacpp-cpu" {
					if err := VerifyCPU(cmd.Context(), e, c); err != nil {
						return fmt.Errorf("%s: %w", c.Name, err)
					}
					fmt.Fprintf(cmd.OutOrStdout(), "verified: %s took no layer on a GPU\n", c.Name)
				}
			}
			if !noWebLLM {
				addr, stop, err := serve(corpus)
				if err != nil {
					return err
				}
				defer stop()
				e.srvAddr = addr
				pw, err := playwright.Run()
				if err != nil {
					return err
				}
				defer pw.Stop()
				e.pw = pw
			}
			samples, err := Interleave(cmd.Context(), e, cfgs, rounds)
			if len(samples) > 0 {
				if werr := writeSamples(out, samples); werr != nil && err == nil {
					err = werr
				}
			}
			fmt.Fprintf(cmd.OutOrStdout(), "browser: %s\n", e.browserInfo)
			return err
		},
	}
	f := c.Flags()
	f.StringVar(&corpus, "corpus", "", "corpus directory")
	f.StringVar(&bench, "bench", "llama-bench", "llama-bench of the go-pkgx bottle (default: from PATH)")
	f.StringVar(&handCPU, "hand-cpu", "", "comparison only: a hand-built llama-bench without Metal and BLAS")
	f.StringVar(&handMetal, "hand-metal", "", "comparison only: a hand-built llama-bench with Metal")
	f.BoolVar(&noWebLLM, "no-webllm", false, "skip the WebLLM rows (and the browser)")
	f.StringVar(&smol, "smollm2", "", "SmolLM2 GGUF, relative to the corpus")
	f.StringVar(&qwen, "qwen", "", "Qwen2.5 GGUF, relative to the corpus")
	f.StringVar(&out, "out", "", "results JSON")
	f.StringVar(&only, "only", "", "run only the configurations whose name contains this")
	f.IntVar(&rounds, "rounds", 7, "rounds (at least 5)")
	f.IntVar(&pp, "pp", 128, "prompt tokens")
	f.IntVar(&tg, "tg", 128, "generated tokens")
	f.Float64Var(&maxLoad, "max-load", 8, "do not start a run while the 1-minute load is above this")
	f.DurationVar(&wait, "wait", 30*time.Second, "pause between load checks")
	for _, n := range []string{"corpus", "smollm2", "qwen", "out"} {
		_ = c.MarkFlagRequired(n)
	}
	return c
}

func writeSamples(path string, s []Sample) error {
	b, err := json.MarshalIndent(s, "", " ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0o644)
}

func tableCmd() *cobra.Command {
	var skip int
	c := &cobra.Command{
		Use:   "table RESULTS.json",
		Short: "print the p50 and spread table of a results file",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			b, err := os.ReadFile(args[0])
			if err != nil {
				return err
			}
			var all, s []Sample
			if err := json.Unmarshal(b, &all); err != nil {
				return err
			}
			for _, x := range all {
				if x.Round > skip {
					s = append(s, x)
				}
			}
			var order []string
			seen := map[string]bool{}
			for _, x := range s {
				if !seen[x.Config] {
					seen[x.Config] = true
					order = append(order, x.Config)
				}
			}
			_, err = io.WriteString(cmd.OutOrStdout(), Table(s, order))
			return err
		},
	}
	c.Flags().IntVar(&skip, "skip-rounds", 1, "leading rounds dropped as warm-up (cold browser, cold page cache)")
	return c
}
