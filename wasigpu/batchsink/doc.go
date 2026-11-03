// The package is built for wasip1 and for native tests, and is empty under
// GOOS=js, where go-gfx/webgpu has no remote device (the browser runs the
// host side, package gpubatch). This file keeps it a package there.

package batchsink
