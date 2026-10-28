package main

import (
	"bytes"
	"compress/gzip"
	"encoding/binary"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeIDX(t *testing.T, path string, hdr []uint32, data []byte) {
	t.Helper()
	var buf bytes.Buffer
	z := gzip.NewWriter(&buf)
	binary.Write(z, binary.BigEndian, hdr)
	z.Write(data)
	z.Close()
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestExtract(t *testing.T) {
	dir := t.TempDir()
	pix := make([]byte, 2*28*28)
	pix[28*28+5] = 200 // picture 1, pixel (5, 0)
	writeIDX(t, filepath.Join(dir, "img.gz"), []uint32{2051, 2, 28, 28}, pix)
	writeIDX(t, filepath.Join(dir, "lab.gz"), []uint32{2049, 2}, []byte{3, 9})
	out := filepath.Join(dir, "d.png")
	var so bytes.Buffer
	cmd := RootCmd()
	cmd.SetArgs([]string{"--images", filepath.Join(dir, "img.gz"), "--labels", filepath.Join(dir, "lab.gz"), "--index", "1", "--out", out})
	cmd.SetOut(&so)
	if err := cmd.Execute(); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(so.String(), "label 9") {
		t.Errorf("output %q", so.String())
	}
	f, _ := os.Open(out)
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	if r, _, _, _ := img.At(5, 0).RGBA(); r>>8 != 200 {
		t.Errorf("pixel %d", r>>8)
	}
	for _, args := range [][]string{
		{"--images", filepath.Join(dir, "lab.gz"), "--labels", filepath.Join(dir, "lab.gz")}, // wrong magic
		{"--images", filepath.Join(dir, "img.gz"), "--labels", filepath.Join(dir, "lab.gz"), "--index", "2"},
		{"--images", filepath.Join(dir, "nope"), "--labels", filepath.Join(dir, "lab.gz")},
	} {
		cmd := RootCmd()
		cmd.SetArgs(append(args, "--out", out))
		cmd.SetOut(&so)
		cmd.SetErr(&so)
		if err := cmd.Execute(); err == nil {
			t.Errorf("%v accepted", args)
		}
	}
}
