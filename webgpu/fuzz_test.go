package webgpu

import (
	"encoding/binary"
	"testing"

	"github.com/go-wasmcontainers/engine/rootfs"
	"github.com/go-wasmcontainers/engine/wasi"
)

// FuzzWebGPUCalls: any import of the module with any core arguments over
// any memory answers or traps, never panics (the guest controls all of
// it), and handles stay bounded.
func FuzzWebGPUCalls(f *testing.F) {
	f.Add([]byte{0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8}, uint16(0), []byte{0})
	m := NewModule(nil)
	m.Limits.Handles = 16
	funcs := m.Funcs()
	f.Fuzz(func(t *testing.T, memory []byte, fn uint16, raw []byte) {
		in := m.Open(wasi.New(wasi.Config{FS: rootfs.New(), Caps: []string{"gpu"}}))
		defer in.Close()
		mm := make(mem, 4096)
		copy(mm, memory)
		for len(raw) > 0 {
			i := int(fn) % len(funcs)
			args := make([]uint64, len(funcs[i].Params))
			for j := range args {
				if len(raw) >= 4 {
					args[j] = uint64(binary.LittleEndian.Uint32(raw))
					raw = raw[4:]
				} else {
					args[j] = uint64(len(raw))
					raw = nil
				}
			}
			r := in.Call(i, args, mm)
			for r.Realloc != nil {
				r = r.Realloc.Then(2048, mm)
			}
			fn = uint16(r.Ret) + fn*31 + 7
		}
		if n := in.(*instance).h.Len(); n > 16 {
			t.Fatalf("%d handles past the quota of 16", n)
		}
	})
}
