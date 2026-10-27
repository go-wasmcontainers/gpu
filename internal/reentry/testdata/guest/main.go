//go:build wasip1

// Command guest is the R1 probe: a Go wasip1 command that imports a host
// function which, while the guest is inside that call, calls back into the
// guest's cabi_realloc export (the canonical ABI's way for a host to return
// a list or a string). It prints one line per round and "ok" at the end.
package main

import (
	"fmt"
	"os"
	"runtime"
	"unsafe"
)

// pinned keeps every block cabi_realloc handed out reachable, so the
// collector never frees memory the host has written into.
var pinned [][]byte

//go:wasmexport cabi_realloc
func cabiRealloc(old, oldSize, align, size uint32) uint32 {
	if size == 0 {
		return align
	}
	b := make([]byte, size+align)
	pinned = append(pinned, b)
	p := uintptr(unsafe.Pointer(&b[0]))
	p = (p + uintptr(align) - 1) &^ (uintptr(align) - 1)
	if old != 0 && oldSize != 0 {
		copy(unsafe.Slice((*byte)(unsafe.Pointer(p)), size), unsafe.Slice((*byte)(unsafe.Pointer(uintptr(old))), oldSize))
	}
	return uint32(p)
}

// give asks the host for n bytes of pattern; the host allocates them with
// cabi_realloc and writes {ptr, len} at ret.
//
//go:wasmimport probe give
func give(n uint32, ret unsafe.Pointer)

type list struct {
	ptr uint32
	len uint32
}

func main() {
	const rounds = 2000
	for i := 0; i < rounds; i++ {
		n := uint32(1 + i*37%65536)
		var l list
		give(n, unsafe.Pointer(&l))
		if l.len != n {
			fmt.Printf("round %d: len %d, want %d\n", i, l.len, n)
			os.Exit(1)
		}
		b := unsafe.Slice((*byte)(unsafe.Pointer(uintptr(l.ptr))), l.len)
		for j, c := range b {
			if c != byte(j*7+i) {
				fmt.Printf("round %d: byte %d is %d, want %d\n", i, j, c, byte(j*7+i))
				os.Exit(1)
			}
		}
		if i%100 == 0 {
			pinned = pinned[:0] // let the collector run over released blocks
			runtime.GC()
		}
	}
	fmt.Println("ok", rounds)
}
