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

// pinnedArg holds the blocks of arguments (AllocArg): the host only reads
// them during the call, so Release can keep them for reuse. Blocks the host
// fills with results (Alloc, the cabi_realloc export) are not reused: the
// values lifted from them alias the block (Memory.Read does not copy), and
// the caller keeps those after Release.
var pinnedArg [][]byte

// free holds argument blocks Release unpinned, for AllocArg to reuse: a guest
// that writes a 250 KiB frame to a GPU buffer every frame allocated (and the
// collector freed) one block per frame, 250 KiB of the 331 KiB per frame the
// quake image allocated (go-wasmcontainers/playground#56). At most maxFree
// blocks and maxFreeBytes bytes are kept.
var free [][]byte

const (
	maxFree      = 8
	maxFreeBytes = 8 << 20
)

func align0(align uint32) uint32 {
	if align == 0 {
		return 1
	}
	return align
}

func addr(b []byte, align uint32) uint32 {
	p := uintptr(unsafe.Pointer(unsafe.SliceData(b)))
	p = (p + uintptr(align) - 1) &^ (uintptr(align) - 1)
	return uint32(p)
}

// Alloc returns a pinned block of size bytes aligned to align, for the host
// to fill (a result); Release unpins it.
func Alloc(align, size uint32) uint32 {
	align = align0(align)
	b := make([]byte, size+align)
	pinned = append(pinned, b)
	return addr(b, align)
}

// AllocArg returns a pinned, zeroed block of size bytes aligned to align for an
// argument the host reads during the call. Release keeps it for reuse.
func AllocArg(align, size uint32) uint32 {
	align = align0(align)
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
	pinnedArg = append(pinnedArg, b)
	return addr(b, align)
}

// Release unpins every block (their contents must have been read). Argument
// blocks are kept for reuse, within maxFree blocks and maxFreeBytes bytes; the
// rest are left to the collector.
func Release() {
	total := 0
	for _, f := range free {
		total += cap(f)
	}
	for i, b := range pinnedArg {
		if len(free) < maxFree && total+cap(b) <= maxFreeBytes {
			free = append(free, b)
			total += cap(b)
		}
		pinnedArg[i] = nil
	}
	pinnedArg = pinnedArg[:0]
	clear(pinned)
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

// Allocator is AllocArg as a wit.Allocator: the blocks it gives hold a call's
// arguments.
type Allocator struct{}

// Alloc implements wit.Allocator.
func (Allocator) Alloc(align, size uint32) (uint32, error) { return AllocArg(align, size), nil }
