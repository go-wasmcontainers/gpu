package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestVendor: from a tree holding one supported test (copied from the
// repository's vendored set) and one with an unparsable model, only the
// first is kept, files and all.
func TestVendor(t *testing.T) {
	from := t.TempDir()
	src := "../../../onnx/testdata/node/test_relu"
	if err := copyTree(src, filepath.Join(from, "test_relu")); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(from, "test_broken"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(from, "test_broken", "model.onnx"), []byte{0xff, 0xff}, 0o644); err != nil {
		t.Fatal(err)
	}
	to := filepath.Join(t.TempDir(), "out")
	var out bytes.Buffer
	cmd := RootCmd()
	cmd.SetArgs([]string{"--from", from, "--to", to})
	cmd.SetOut(&out)
	cmd.SetErr(&out)
	if err := cmd.Execute(); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "kept 1 tests, skipped 1") {
		t.Errorf("output %q", out.String())
	}
	want, _ := os.ReadFile(filepath.Join(src, "test_data_set_0", "input_0.pb"))
	got, err := os.ReadFile(filepath.Join(to, "test_relu", "test_data_set_0", "input_0.pb"))
	if err != nil || !bytes.Equal(got, want) {
		t.Errorf("copied input differs: %v", err)
	}
	if _, err := os.Stat(filepath.Join(to, "test_broken")); !os.IsNotExist(err) {
		t.Errorf("the broken test was copied: %v", err)
	}
	// Too big for the size limit: skipped.
	cmd = RootCmd()
	cmd.SetArgs([]string{"--from", from, "--to", filepath.Join(t.TempDir(), "o2"), "--max-bytes", "10"})
	out.Reset()
	cmd.SetOut(&out)
	if err := cmd.Execute(); err != nil || !strings.Contains(out.String(), "kept 0 tests") {
		t.Errorf("max-bytes: %v %q", err, out.String())
	}
	cmd = RootCmd()
	cmd.SetArgs([]string{"--from", filepath.Join(from, "missing")})
	cmd.SetOut(&out)
	cmd.SetErr(&out)
	if err := cmd.Execute(); err == nil {
		t.Error("a missing source directory was accepted")
	}
}
