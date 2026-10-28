package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	witdata "github.com/go-wasmcontainers/gpu/webgpu/wit"
	"github.com/go-wasmcontainers/gpu/wit"
)

// TestGeneratedIsCurrent: wasigpu/imports_wasip1.go is what witgen writes
// from the vendored WIT today (regenerate after changing either).
func TestGeneratedIsCurrent(t *testing.T) {
	ifs, err := wit.Parse(witdata.WebGPU)
	if err != nil {
		t.Fatal(err)
	}
	want, err := Generate(ifs["webgpu"], "wasigpu")
	if err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile("../../../wasigpu/imports_wasip1.go")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Fatal("wasigpu/imports_wasip1.go is stale: go run ./internal/cmd/witgen --wit webgpu/wit/webgpu.wit --interface webgpu --package wasigpu --out wasigpu/imports_wasip1.go")
	}
}

func TestCommand(t *testing.T) {
	out := filepath.Join(t.TempDir(), "x.go")
	var so bytes.Buffer
	cmd := RootCmd()
	cmd.SetArgs([]string{"--wit", "../../../webgpu/wit/webgpu.wit", "--interface", "webgpu", "--package", "x", "--out", out})
	cmd.SetOut(&so)
	if err := cmd.Execute(); err != nil {
		t.Fatal(err)
	}
	b, _ := os.ReadFile(out)
	if !strings.Contains(string(b), "//go:wasmimport wasi:webgpu/webgpu@0.3.0-rc.2 [method]gpu-device.create-buffer") {
		t.Error("no create-buffer import")
	}
	cmd = RootCmd()
	cmd.SetArgs([]string{"--wit", "../../../webgpu/wit/webgpu.wit", "--interface", "nope", "--package", "x", "--out", out})
	cmd.SetOut(&so)
	cmd.SetErr(&so)
	if err := cmd.Execute(); err == nil {
		t.Error("unknown interface accepted")
	}
}
