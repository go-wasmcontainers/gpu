//go:build wasip1

package batchsink

import "unsafe"

//go:wasmimport wasi_webgpu_batch caps
func hostCaps(buf unsafe.Pointer, n uint32) uint32

//go:wasmimport wasi_webgpu_batch context_create
func hostContextCreate(out unsafe.Pointer) uint32

//go:wasmimport wasi_webgpu_batch submit
func hostSubmit(ctx uint32, frame unsafe.Pointer, frameLen uint32, read unsafe.Pointer, readLen uint32) uint32

//go:wasmimport wasi_webgpu_batch context_destroy
func hostContextDestroy(ctx uint32) uint32

// ptr is the address of a slice's first byte, or nil for an empty one (a
// host never reads a zero-length range, and &b[0] would panic).
func ptr(b []byte) unsafe.Pointer {
	if len(b) == 0 {
		return nil
	}
	return unsafe.Pointer(&b[0])
}

// wasi is Host over the imports.
type wasi struct{}

func (wasi) Caps(buf []byte) Errno { return Errno(hostCaps(ptr(buf), uint32(len(buf)))) }
func (wasi) ContextCreate() (uint32, Errno) {
	var h uint32
	e := Errno(hostContextCreate(unsafe.Pointer(&h)))
	return h, e
}
func (wasi) Submit(ctx uint32, frame, read []byte) Errno {
	return Errno(hostSubmit(ctx, ptr(frame), uint32(len(frame)), ptr(read), uint32(len(read))))
}
func (wasi) ContextDestroy(ctx uint32) Errno { return Errno(hostContextDestroy(ctx)) }

// Open creates a context on the container's GPU through the host module
// wasi_webgpu_batch. It fails with an *Error: Denied without run --gpu,
// Unsupported where there is no GPU or only a software adapter.
func Open() (*Sink, error) { return NewSink(wasi{}) }
