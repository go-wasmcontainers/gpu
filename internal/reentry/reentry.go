// Package reentry measures risk R1 of docs/DESIGN.md: can a host, while a
// Go wasip1 guest is inside a //go:wasmimport call, call the guest's
// //go:wasmexport cabi_realloc and write the result into the block it
// returns? The canonical ABI needs exactly that to return a list or a
// string from a host function.
//
// Realloc is the host side, written against wazero's api.Module so the test
// and the future canonical-ABI host share it.
package reentry

import (
	"context"
	"errors"
	"fmt"

	"github.com/tetratelabs/wazero/api"
)

// ErrNoRealloc is returned when the guest exports no cabi_realloc.
var ErrNoRealloc = errors.New("guest exports no cabi_realloc")

// Realloc calls the guest's cabi_realloc(0, 0, align, size) and returns the
// pointer, checking that the block lies inside the guest's memory.
func Realloc(ctx context.Context, mod api.Module, align, size uint32) (uint32, error) {
	f := mod.ExportedFunction("cabi_realloc")
	if f == nil {
		return 0, ErrNoRealloc
	}
	res, err := f.Call(ctx, 0, 0, uint64(align), uint64(size))
	if err != nil {
		return 0, fmt.Errorf("cabi_realloc: %w", err)
	}
	if len(res) != 1 {
		return 0, fmt.Errorf("cabi_realloc returned %d values", len(res))
	}
	p := uint32(res[0])
	if err := CheckBlock(p, align, size, mod.Memory().Size()); err != nil {
		return 0, err
	}
	return p, nil
}

// CheckBlock refuses a block a guest's allocator returned unless it is
// aligned and lies inside a memory of memSize bytes. A hostile guest
// chooses p, so the sum is computed without overflow.
func CheckBlock(p, align, size, memSize uint32) error {
	if align > 1 && p%align != 0 {
		return fmt.Errorf("cabi_realloc returned %#x, not aligned to %d", p, align)
	}
	if uint64(p)+uint64(size) > uint64(memSize) {
		return fmt.Errorf("cabi_realloc returned %#x+%d, outside memory", p, size)
	}
	return nil
}
