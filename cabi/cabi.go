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

// Alloc returns a pinned block of size bytes aligned to align.
func Alloc(align, size uint32) uint32 {
	if align == 0 {
		align = 1
	}
	b := make([]byte, size+align)
	pinned = append(pinned, b)
	p := uintptr(unsafe.Pointer(unsafe.SliceData(b)))
	p = (p + uintptr(align) - 1) &^ (uintptr(align) - 1)
	return uint32(p)
}

// Release unpins every block (their contents must have been read).
func Release() { pinned = pinned[:0] }

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
