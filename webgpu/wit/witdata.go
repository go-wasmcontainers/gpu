// Package witdata holds the wasi:webgpu WIT source (README.md here for its
// origin), shared by the host (package webgpu) and the guest bindings
// (package wasigpu), so both read the same interface.
package witdata

import _ "embed"

// WebGPU is wit/webgpu.wit of WebAssembly/wasi-gfx.
//
//go:embed webgpu.wit
var WebGPU string
