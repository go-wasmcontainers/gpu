package nn

import (
	"encoding/binary"
	"os"
	"testing"

	"github.com/go-wasmcontainers/engine/rootfs"
	"github.com/go-wasmcontainers/engine/wasi"
)

// FuzzHostCalls: any sequence of calls with any arguments over any guest
// memory answers with an nn_errno and never panics or traps (a guest
// controls all of it). The memory starts with the MNIST model in place so
// sequences can get past load.
func FuzzHostCalls(f *testing.F) {
	model, _ := os.ReadFile("../onnx/testdata/models/mnist-12/mnist-12.onnx")
	f.Add([]byte{0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 16, 2, 1, 0, 0, 0, 20, 5, 1})
	f.Fuzz(func(t *testing.T, prog []byte) {
		mod := NewModule()
		mod.Limits.Graphs, mod.Limits.Contexts = 2, 4
		in := mod.Open(wasi.New(wasi.Config{FS: rootfs.New()}))
		m := make(mem, 1<<17)
		copy(m[1<<16:], model)
		binary.LittleEndian.PutUint32(m[0:], 1<<16)
		binary.LittleEndian.PutUint32(m[4:], uint32(len(model)))
		funcs := mod.Funcs()
		for len(prog) >= 2 {
			fn := int(prog[0]) % len(funcs)
			n := len(funcs[fn].Params)
			prog = prog[1:]
			args := make([]uint64, n)
			for i := range args {
				if len(prog) == 0 {
					break
				}
				// Small values reach handles and offsets; a high bit
				// reaches far pointers.
				v := uint64(prog[0])
				if v&0x80 != 0 {
					v = (v & 0x7f) << 12
				}
				args[i] = v
				prog = prog[1:]
			}
			res := in.Call(fn, args, m)
			if res.Trap != "" || res.Ret > uint64(NotFound) {
				t.Fatalf("call %d %v: %+v", fn, args, res)
			}
		}
		in.Close()
	})
}
