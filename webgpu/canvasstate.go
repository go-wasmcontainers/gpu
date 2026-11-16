package webgpu

import (
	"errors"

	wg "github.com/go-gfx/webgpu"
)

// unconfigured reports whether err is the browser saying a canvas's WebGPU
// context has no configuration: getCurrentTexture threw an InvalidStateError.
//
// Firefox drops a context's configuration when transferToImageBitmap takes
// the frame (measured on Firefox 160, macOS: the second getCurrentTexture
// after a present throws "InvalidStateError: GPUCanvasContext.getCurrentTexture:
// Canvas not configured"; Chromium keeps the configuration). Resizing the
// canvas restores it. The guest sees the context it configured, so the host
// configures it again and retries once.
func unconfigured(err error) bool {
	var je *wg.JSError
	return errors.As(err, &je) && je.Op == "getCurrentTexture" && je.Name == "InvalidStateError"
}
