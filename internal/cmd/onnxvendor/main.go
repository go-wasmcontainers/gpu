// Command onnxvendor copies the ONNX node tests this repository's onnx
// package can run into onnx/testdata/node: from a checkout of
// github.com/onnx/onnx (v1.22.0 is the last tag that ships
// onnx/backend/test/data/node), every test whose model decodes, uses only
// supported operators and passes onnx.NewSession. The list is derived from
// the operator set, not written by hand, so a new operator brings its
// tests along on the next run.
//
//	go run ./internal/cmd/onnxvendor --from ~/src/onnx/onnx/backend/test/data/node --to onnx/testdata/node
package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/spf13/cobra"

	"github.com/go-wasmcontainers/gpu/onnx"
)

var osExit = os.Exit

func main() {
	if err := RootCmd().Execute(); err != nil {
		osExit(1)
	}
}

// RootCmd is the command, for tests.
func RootCmd() *cobra.Command {
	var from, to string
	var maxBytes int64
	cmd := &cobra.Command{
		Use:          "onnxvendor --from DIR --to DIR",
		Short:        "Copy the ONNX node tests the onnx package supports",
		SilenceUsage: true,
		Args:         cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			kept, skipped, err := vendor(from, to, maxBytes)
			if err != nil {
				return err
			}
			fmt.Fprintf(cmd.OutOrStdout(), "kept %d tests, skipped %d\n", len(kept), skipped)
			for _, k := range kept {
				fmt.Fprintln(cmd.OutOrStdout(), " ", k)
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&from, "from", "", "onnx/backend/test/data/node of an onnx checkout")
	cmd.Flags().StringVar(&to, "to", "onnx/testdata/node", "destination")
	cmd.Flags().Int64Var(&maxBytes, "max-bytes", 256<<10, "skip a test whose files exceed this size")
	_ = cmd.MarkFlagRequired("from")
	return cmd
}

// vendor copies the supported tests and returns their names.
func vendor(from, to string, maxBytes int64) ([]string, int, error) {
	entries, err := os.ReadDir(from)
	if err != nil {
		return nil, 0, err
	}
	var kept []string
	skipped := 0
	for _, e := range entries {
		if !e.IsDir() || !strings.HasPrefix(e.Name(), "test_") {
			continue
		}
		dir := filepath.Join(from, e.Name())
		ok, err := supported(dir, maxBytes)
		if err != nil {
			return nil, 0, err
		}
		if !ok {
			skipped++
			continue
		}
		if err := copyTree(dir, filepath.Join(to, e.Name())); err != nil {
			return nil, 0, err
		}
		kept = append(kept, e.Name())
	}
	slices.Sort(kept)
	return kept, skipped, nil
}

// supported reports whether the test's model loads in package onnx and
// the test is small enough to commit.
func supported(dir string, maxBytes int64) (bool, error) {
	var size int64
	err := filepath.WalkDir(dir, func(p string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		info, err := d.Info()
		if err != nil {
			return err
		}
		size += info.Size()
		return nil
	})
	if err != nil {
		return false, err
	}
	if size > maxBytes {
		return false, nil
	}
	b, err := os.ReadFile(filepath.Join(dir, "model.onnx"))
	if err != nil {
		return false, nil
	}
	m, err := onnx.Decode(b, onnx.DefaultLimits)
	if err != nil {
		return false, nil
	}
	if _, err := onnx.NewSession(m, onnx.DefaultLimits); err != nil {
		return false, nil
	}
	return true, nil
}

func copyTree(src, dst string) error {
	return filepath.WalkDir(src, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(src, p)
		if err != nil {
			return err
		}
		out := filepath.Join(dst, rel)
		if d.IsDir() {
			return os.MkdirAll(out, 0o755)
		}
		if !d.Type().IsRegular() {
			return fmt.Errorf("%s: not a regular file", p)
		}
		in, err := os.Open(p)
		if err != nil {
			return err
		}
		defer in.Close()
		f, err := os.Create(out)
		if err != nil {
			return err
		}
		if _, err := io.Copy(f, in); err != nil {
			f.Close()
			return err
		}
		return f.Close()
	})
}
