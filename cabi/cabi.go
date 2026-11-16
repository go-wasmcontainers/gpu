// Package cabi is the guest side of the Canonical ABI's memory for Go
// programs built with GOOS=wasip1: the cabi_realloc export a host calls to
// return lists and strings, guest-side allocations for arguments, and the
// guest's own linear memory as a wit.Memory. Blocks stay pinned (reachable
// from Go) until Release, which a caller runs once a host call's results
// have been copied out.
//
// A program links one cabi_realloc: every guest package of this module
// that needs it imports this one.
package cabi

import "unsafe"

var pinned [][]byte

// free holds blocks Release unpinned, for Alloc to reuse: a guest that writes
// a 250 KiB frame to a GPU buffer every frame allocated (and the collector
// freed) one block per frame, 250 KiB of the 331 KiB per frame the quake
// image allocated (go-wasmcontainers/playground#56). At most maxFree blocks and
// maxFreeBytes bytes are kept.
var free [][]byte

const (
	maxFree      = 8
	maxFreeBytes = 8 << 20
)

// Alloc returns a pinned block of size bytes aligned to align, zeroed.
func Alloc(align, size uint32) uint32 {
	if align == 0 {
		align = 1
	}
	n := int(size) + int(align)
	var b []byte
	best := -1
	for i, f := range free {
		if cap(f) >= n && (best < 0 || cap(f) < cap(free[best])) {
			best = i
		}
	}
	if best >= 0 {
		b = free[best][:n]
		free = append(free[:best], free[best+1:]...)
		clear(b)
	} else {
		b = make([]byte, n)
	}
	pinned = append(pinned, b)
	p := uintptr(unsafe.Pointer(unsafe.SliceData(b)))
	p = (p + uintptr(align) - 1) &^ (uintptr(align) - 1)
	return uint32(p)
}

// Release unpins every block (their contents must have been read). The blocks
// are kept for reuse, within maxFree blocks and maxFreeBytes bytes; the rest
// are left to the collector.
func Release() {
	total := 0
	for _, f := range free {
		total += cap(f)
	}
	for i, b := range pinned {
		if len(free) < maxFree && total+cap(b) <= maxFreeBytes {
			free = append(free, b)
			total += cap(b)
		}
		pinned[i] = nil
	}
	pinned = pinned[:0]
}

// Buffer returns a block of size bytes aligned to align that Release does
// not unpin: it stays valid while the caller keeps keep reachable (a
// buffer the host fills later, such as a pending stream read's).
func Buffer(align, size uint32) (addr uint32, keep []byte) {
	if align == 0 {
		align = 1
	}
	keep = make([]byte, size+align)
	p := uintptr(unsafe.Pointer(unsafe.SliceData(keep)))
	p = (p + uintptr(align) - 1) &^ (uintptr(align) - 1)
	return uint32(p), keep
}

// Memory is the guest's own linear memory.
type Memory struct{}

func at(off uint32) unsafe.Pointer { return unsafe.Add(unsafe.Pointer(nil), uintptr(off)) }

// Read returns n bytes at off.
func (Memory) Read(off, n uint32) ([]byte, bool) {
	if n == 0 {
		return []byte{}, true
	}
	return unsafe.Slice((*byte)(at(off)), n), true
}

// Write copies b to off.
func (Memory) Write(off uint32, b []byte) bool {
	if len(b) > 0 {
		copy(unsafe.Slice((*byte)(at(off)), len(b)), b)
	}
	return true
}

// Allocator is Alloc as a wit.Allocator.
type Allocator struct{}

// Alloc implements wit.Allocator.
func (Allocator) Alloc(align, size uint32) (uint32, error) { return Alloc(align, size), nil }
