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

// Interfaces are those wasigpu imports: wasi:webgpu, then wasi-gfx's
// surface and surface-webgpu.
func interfaces(t *testing.T) []*wit.Interface {
	t.Helper()
	w, err := wit.Parse(witdata.WebGPU)
	if err != nil {
		t.Fatal(err)
	}
	s, err := wit.ParseFiles([]*wit.Interface{w["webgpu"]}, witdata.Surface)
	if err != nil {
		t.Fatal(err)
	}
	return []*wit.Interface{w["webgpu"], s["surface"], s["surface-webgpu"]}
}

// TestGeneratedIsCurrent: wasigpu/imports_wasip1.go is what witgen writes
// from the WIT today (regenerate after changing either; the command is in
// wasigpu/doc.go).
func TestGeneratedIsCurrent(t *testing.T) {
	want, err := Generate(interfaces(t), "wasigpu")
	if err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile("../../../wasigpu/imports_wasip1.go")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Fatal("wasigpu/imports_wasip1.go is stale: regenerate it (see wasigpu/wasigpu.go)")
	}
}

func TestCommand(t *testing.T) {
	out := filepath.Join(t.TempDir(), "x.go")
	var so bytes.Buffer
	cmd := RootCmd()
	cmd.SetArgs([]string{"--wit", "../../../webgpu/wit/webgpu.wit", "--wit", "../../../webgpu/wit/surface.wit",
		"--interface", "webgpu", "--interface", "surface-webgpu", "--package", "x", "--out", out})
	cmd.SetOut(&so)
	if err := cmd.Execute(); err != nil {
		t.Fatal(err)
	}
	b, _ := os.ReadFile(out)
	for _, want := range []string{
		"//go:wasmimport wasi:webgpu/webgpu@0.3.0-rc.2 [method]gpu-device.create-buffer",
		"//go:wasmimport wasi-gfx:surface/surface-webgpu@0.2.0 [method]context.present",
	} {
		if !strings.Contains(string(b), want) {
			t.Errorf("no %q", want)
		}
	}
	cmd = RootCmd()
	cmd.SetArgs([]string{"--wit", "../../../webgpu/wit/webgpu.wit", "--interface", "nope", "--package", "x", "--out", out})
	cmd.SetOut(&so)
	cmd.SetErr(&so)
	if err := cmd.Execute(); err == nil {
		t.Error("unknown interface accepted")
	}
}
