package gpubatch

import (
	"bytes"
	"encoding/binary"
	"testing"

	"github.com/go-gfx/webgpu"
	"github.com/go-gfx/webgpu/wire"
	"github.com/go-virtio/gpu/wgpubatch"
	"github.com/go-wasmcontainers/engine/rootfs"
	"github.com/go-wasmcontainers/engine/wasi"
)

func seedFrame(stream []uint32, data []byte) []byte {
	k, d, err := webgpu.EncodeDescriptor(webgpu.BufferDescriptor{Size: 64, Usage: webgpu.BufferUsageMapRead | webgpu.BufferUsageCopyDst})
	if err != nil {
		panic(err)
	}
	e := wgpubatch.NewEncoder(4096)
	e.Create(1, uint32(k), d)
	if stream != nil {
		e.Stream(stream, data)
	}
	f, err := e.Frame()
	if err != nil {
		panic(err)
	}
	return append([]byte(nil), f...)
}

// FuzzBatchCalls drives arbitrary call sequences, with arbitrary arguments,
// over arbitrary guest memory: the module must never panic or hang, must
// bound every guest range before it reads or writes, and must write guest
// memory only where the call says (caps' buffer, the handle's slot, the
// submit's read window).
//
// ops is a list of 11-byte records: a function (0..5, 4 and 5 are not
// functions) and five 16-bit arguments, each widened by a flag byte to the
// extremes a guest can pass (0xffffffff, 2^31, the end of memory).
func FuzzBatchCalls(f *testing.F) {
	rec := func(fn byte, a ...uint16) []byte {
		b := []byte{fn}
		for i := 0; i < 5; i++ {
			v := uint16(0)
			if i < len(a) {
				v = a[i]
			}
			b = binary.LittleEndian.AppendUint16(b, v)
		}
		return append(b, 0) // widening flags
	}
	good := seedFrame([]uint32{uint32(wire.ReadBuffer), 1, 8, 0, 16, 4}, nil)
	seedMem := append(make([]byte, 256), good...) // the frame at 256
	n := uint16(len(good))
	ops := bytes.Join([][]byte{
		rec(0, 16, 256),             // caps
		rec(1, 32),                  // context_create
		rec(2, 1, 256, n, 1024, 32), // submit with a window
		rec(2, 1, 256, n, 1024, 0),
		rec(2, 1, 256, n, 0xfff0, 40000), // window past memory
		rec(3, 1),
	}, nil)
	f.Add(ops, seedMem)
	f.Add(ops, make([]byte, 64))
	f.Add([]byte{}, []byte{})

	f.Fuzz(func(t *testing.T, ops, seed []byte) {
		const memSize = 1 << 14
		m := make(mem, memSize)
		for i := range m {
			if len(seed) > 0 {
				m[i] = seed[i%len(seed)]
			}
		}
		dev := &fakeDevice{}
		mod := NewModule(&fakeProvider{dev: dev})
		mod.Limits = Limits{MaxFrame: 4096, MaxObjects: 64, MaxRead: 4096, Memory: 1 << 20, MaxMemory: 1 << 20}
		in := mod.Open(wasi.New(wasi.Config{FS: rootfs.New(), Caps: []string{"gpu"}}))
		defer in.Close()
		widen := func(v uint16, flag byte, i int) uint64 {
			switch flag >> (2 * i) & 3 {
			case 1:
				return 0xffffffff
			case 2:
				return 1<<31 + uint64(v)
			case 3:
				return memSize - uint64(v&63)
			}
			return uint64(v)
		}
		for len(ops) >= 11 {
			fn := int(ops[0] % 6)
			var args []uint64
			for i := 0; i < 5; i++ {
				args = append(args, widen(binary.LittleEndian.Uint16(ops[1+2*i:]), ops[11-1], i))
			}
			ops = ops[11:]
			if fn < len(funcs) {
				args = args[:len(funcs[fn].Params)]
			}
			before := bytes.Clone(m)
			r := in.Call(fn, args, m)
			for r.Wait != nil {
				<-r.Wait
				r = r.Resume(m)
			}
			if fn >= len(funcs) {
				if r.Trap == "" {
					t.Fatalf("function %d did not trap", fn)
				}
				continue
			}
			// Where may the call have written?
			lo, hi := uint64(0), uint64(0)
			lo2, hi2 := uint64(0), uint64(0) // a second window (last_error writes its length and its text)
			switch fn {
			case fnCaps:
				lo, hi = args[0], args[0]+wgpubatch.CapsSizeV2
			case fnContextCreate:
				lo, hi = args[0], args[0]+4
			case fnSubmit:
				lo, hi = args[3], args[3]+args[4]
			case fnLastError: // last_error(ctx, buf, len, out)
				lo, hi = args[1], args[1]+args[2]
				lo2, hi2 = args[3], args[3]+4
			}
			for i := range m {
				if (uint64(i) < lo || uint64(i) >= hi) && (uint64(i) < lo2 || uint64(i) >= hi2) && m[i] != before[i] {
					t.Fatalf("function %d wrote guest byte %d outside [%d, %d)", fn, i, lo, hi)
				}
			}
			if r.Ret > uint64(Unbounded) { // the last errno
				t.Fatalf("function %d returned %d", fn, r.Ret)
			}
		}
	})
}
