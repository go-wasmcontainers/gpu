package adapter

import (
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"

	wg "github.com/go-gfx/webgpu"
)

func TestOptionsAskForHighPerformance(t *testing.T) {
	for _, fb := range []bool{false, true} {
		o := Options(fb)
		if o.PowerPreference != wg.PowerPreferenceHighPerformance {
			t.Errorf("fallback %v: power preference %v, want high-performance", fb, o.PowerPreference)
		}
		if o.AllowFallback != fb {
			t.Errorf("AllowFallback %v, want %v", o.AllowFallback, fb)
		}
	}
}

// TestNoOtherRequestAdapterOptions: no source file of the module builds its
// own RequestAdapterOptions (it would omit the power preference again).
func TestNoOtherRequestAdapterOptions(t *testing.T) {
	err := filepath.WalkDir("../..", func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || !strings.HasSuffix(p, ".go") || strings.HasSuffix(p, "_test.go") || filepath.Base(p) == "adapter.go" {
			return err
		}
		b, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		if strings.Contains(string(b), "RequestAdapterOptions{") {
			t.Errorf("%s builds its own RequestAdapterOptions: use adapter.Options", p)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}
