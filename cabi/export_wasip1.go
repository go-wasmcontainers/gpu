//go:build wasip1

package cabi

// realloc is the Canonical ABI's allocator, called by the host during an
// import to return a list or a string (measured safe for Go wasip1 guests
// by internal/reentry). Growing an old block is not needed by the hosts
// of this module and copies.
//
//go:wasmexport cabi_realloc
func realloc(old, oldSize, align, size uint32) uint32 {
	p := Alloc(align, size)
	if old != 0 && oldSize != 0 {
		var m Memory
		b, _ := m.Read(old, min(oldSize, size))
		m.Write(p, append([]byte(nil), b...))
	}
	return p
}
