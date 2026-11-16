package webgpu

import (
	"errors"
	"fmt"
	"testing"

	wg "github.com/go-gfx/webgpu"
)

func TestUnconfigured(t *testing.T) {
	// The error Firefox 160 gives for the second getCurrentTexture after a
	// present (go-wasmcontainers/playground#56).
	firefox := &wg.JSError{Op: "getCurrentTexture", Name: "InvalidStateError", Message: "GPUCanvasContext.getCurrentTexture: Canvas not configured"}
	for _, c := range []struct {
		name string
		err  error
		want bool
	}{
		{"firefox", firefox, true},
		{"wrapped", fmt.Errorf("surface: %w", firefox), true},
		{"nil", nil, false},
		{"another op", &wg.JSError{Op: "configure", Name: "InvalidStateError"}, false},
		{"another error", &wg.JSError{Op: "getCurrentTexture", Name: "TypeError"}, false},
		{"not a browser error", errors.New("boom"), false},
	} {
		if got := unconfigured(c.err); got != c.want {
			t.Errorf("%s: unconfigured = %v, want %v", c.name, got, c.want)
		}
	}
}
