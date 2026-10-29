package webgpu

import (
	"encoding/binary"
	"testing"

	"github.com/go-wasmcontainers/engine/rootfs"
	"github.com/go-wasmcontainers/engine/wasi"
	"github.com/go-wasmcontainers/gpu/wit"
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

// FuzzStreamCalls: the stream built-ins and the $root waitable-set
// functions, called in any order with any arguments over any memory, with
// live pointer and key streams being written, answer or trap and never
// panic; a call that would block is not resumed but must be woken when the
// process ends.
func FuzzStreamCalls(f *testing.F) {
	f.Add([]byte{0, 1, 2, 3}, uint16(0), []byte{1, 0, 0, 0, 64, 0, 0, 0, 4, 0, 0, 0})
	f.Add([]byte{}, uint16(7), []byte{0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0})
	mods := NewModules(nil)
	type target struct {
		mod int
		fn  int
		n   int // core parameters
	}
	var targets []target
	for _, mi := range []int{1, 3} {
		for i, hf := range mods[mi].Funcs() {
			if mi == 3 || mods[mi].exports[i].builtin != nil {
				targets = append(targets, target{mi, i, len(hf.Params)})
			}
		}
	}
	f.Fuzz(func(t *testing.T, memory []byte, fn uint16, raw []byte) {
		p := wasi.New(wasi.Config{FS: rootfs.New()})
		ins := make([]wasi.Instance, len(mods))
		for i, m := range mods {
			ins[i] = m.Open(p)
		}
		h := ins[0].(*instance).h
		var streams []*Stream
		for _, kind := range []string{"pointer-down", "key-down"} {
			s := NewStream(inputElem(kind))
			h.New("stream", s)
			streams = append(streams, s)
		}
		mm := make(mem, 4096)
		copy(mm, memory)
		var waits []<-chan struct{}
		for step := 0; len(raw) > 0 && step < 64; step++ {
			tg := targets[int(fn)%len(targets)]
			args := make([]uint64, tg.n)
			for j := range args {
				if len(raw) >= 4 {
					args[j] = uint64(binary.LittleEndian.Uint32(raw))
					raw = raw[4:]
				} else {
					args[j] = uint64(len(raw))
					raw = nil
				}
			}
			streams[0].Push([]any{float64(step), 1.0})
			streams[1].Push([]any{nil, wit.Some{Value: "k"}, true, false, false, false})
			r := ins[tg.mod].Call(tg.fn, args, mm)
			for r.Realloc != nil {
				r = r.Realloc.Then(2048, mm)
			}
			if r.Wait != nil {
				waits = append(waits, r.Wait)
			}
			fn = uint16(r.Ret) + fn*31 + 7
		}
		for _, in := range ins {
			in.Close()
		}
		for _, w := range waits {
			<-w // the process ended: every waiter is woken
		}
	})
}
