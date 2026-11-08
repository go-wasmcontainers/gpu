// Package adapter holds the one set of options every production
// RequestAdapter in this module uses.
package adapter

import wg "github.com/go-gfx/webgpu"

// Options asks for the high-performance adapter. Without a power preference a
// browser on a machine with an integrated and a discrete GPU returns the
// integrated one (gpu#23: Chromium on Linux picked the Ryzen iGPU, rdna-2,
// instead of the Radeon PRO W7700).
func Options(allowFallback bool) *wg.RequestAdapterOptions {
	return &wg.RequestAdapterOptions{AllowFallback: allowFallback, PowerPreference: wg.PowerPreferenceHighPerformance}
}
