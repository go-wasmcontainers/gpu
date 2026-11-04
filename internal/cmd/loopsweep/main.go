// Command loopsweep finds the smallest loop-budget work (gpubatch layer 4)
// with which a set of e2e tests still passes with zero truncations: it runs
// them at LoopWork = 2^k for k from --from to --to, with the per-loop floor
// at 1 so that the floor masks nothing, and prints one line per k. The
// smallest passing k, against the shipped default (gpubatch.DefaultLoopWork),
// is the margin of the default.
//
//	loopsweep --from 24 --to 38 --run 'TestLLM(Oracle|Trace|Shaders)' \
//	    --env LLM_MODELS=... --env LLM_REPO=... --out DIR
package main

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"

	"github.com/spf13/cobra"
)

var osExit = os.Exit

// Result is one point of the sweep.
type Result struct {
	K    int
	Work uint64
	Pass bool
}

// Sweep runs one test run per k in [from, to] and returns the results in
// order; run gets the work for that k.
func Sweep(from, to int, run func(k int, work uint64) (bool, error)) ([]Result, error) {
	if from < 0 || to < from || to > 62 {
		return nil, fmt.Errorf("loopsweep: bad range %d..%d", from, to)
	}
	var out []Result
	for k := from; k <= to; k++ {
		w := uint64(1) << k
		ok, err := run(k, w)
		if err != nil {
			return out, err
		}
		out = append(out, Result{k, w, ok})
	}
	return out, nil
}

// Smallest returns the smallest k that passes, or -1.
func Smallest(rs []Result) int {
	for _, r := range rs {
		if r.Pass {
			return r.K
		}
	}
	return -1
}

// RootCmd is the command.
func RootCmd() *cobra.Command {
	var from, to int
	var re, dir, repo string
	var envs []string
	cmd := &cobra.Command{
		Use:          "loopsweep",
		Short:        "smallest loop-budget work with which the e2e tests pass",
		SilenceUsage: true,
		Args:         cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			if re == "" || dir == "" {
				return fmt.Errorf("--run and --out are required")
			}
			if err := os.MkdirAll(dir, 0o755); err != nil {
				return err
			}
			rs, err := Sweep(from, to, func(k int, work uint64) (bool, error) {
				f, err := os.Create(filepath.Join(dir, "k"+strconv.Itoa(k)+".txt"))
				if err != nil {
					return false, err
				}
				defer f.Close()
				c := exec.Command("go", "test", "-tags", "e2e", "-count=1", "-run", re, "./e2e/")
				c.Dir = repo
				c.Stdin = nil
				c.Stdout, c.Stderr = io.Writer(f), io.Writer(f)
				c.Env = append(append(os.Environ(), envs...),
					"GPU_E2E_LOOP_WORK="+strconv.FormatUint(work, 10), "GPU_E2E_LOOP_MIN=1")
				err = c.Run()
				_, isExit := err.(*exec.ExitError)
				if err != nil && !isExit {
					return false, err
				}
				fmt.Fprintf(cmd.OutOrStdout(), "k=%d work=%d pass=%v\n", k, work, err == nil)
				return err == nil, nil
			})
			if err != nil {
				return err
			}
			fmt.Fprintf(cmd.OutOrStdout(), "smallest passing k: %d\n", Smallest(rs))
			return nil
		},
	}
	f := cmd.Flags()
	f.IntVar(&from, "from", 24, "smallest exponent")
	f.IntVar(&to, "to", 36, "largest exponent")
	f.StringVar(&re, "run", "", "go test -run regexp")
	f.StringVar(&dir, "out", "", "directory for each run's output")
	f.StringVar(&repo, "repo", ".", "the gpu repository")
	f.StringArrayVar(&envs, "env", nil, "KEY=VALUE for the test run (repeatable)")
	return cmd
}

func main() {
	if err := RootCmd().Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		osExit(1)
	}
}
