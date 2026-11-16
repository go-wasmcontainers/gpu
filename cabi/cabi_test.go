package cabi

import "testing"

// A frame-sized argument allocated on every host call used to cost one
// 250 KiB block per call: Release now keeps argument blocks and AllocArg reuses them.
func TestReleasedArgBlocksAreReused(t *testing.T) {
	const frame = 256000
	Release()
	free = nil
	AllocArg(4, frame) // the first call allocates
	Release()
	if n := testing.AllocsPerRun(100, func() {
		AllocArg(4, frame)
		AllocArg(8, 64)
		Release()
	}); n > 1 { // the small block is allowed to miss once, never the frame
		t.Errorf("AllocArg and Release of a frame allocate %.0f times per call, want 0 (1 at most)", n)
	}
}

func TestReusedBlockIsZeroed(t *testing.T) {
	free = nil
	AllocArg(4, 1024)
	for i := range pinnedArg[0] {
		pinnedArg[0][i] = 0xFF
	}
	Release()
	AllocArg(4, 1000)
	for i, v := range pinnedArg[0] {
		if v != 0 {
			t.Fatalf("byte %d of a reused block is %#x, want 0", i, v)
		}
	}
	Release()
}

// Values lifted from a result block alias it (Memory.Read does not copy) and
// outlive Release, so result blocks must never be handed out again.
func TestResultBlocksAreNotReused(t *testing.T) {
	free = nil
	Alloc(4, 64)
	result := pinned[0]
	for i := range result {
		result[i] = 0xAB
	}
	Release()
	AllocArg(4, 64)
	Alloc(4, 64)
	Release()
	for i, v := range result {
		if v != 0xAB {
			t.Fatalf("byte %d of a released result block was overwritten (%#x): a caller still holds it", i, v)
		}
	}
}

func TestFreeListIsBounded(t *testing.T) {
	free = nil
	for range 20 {
		AllocArg(4, 1<<20)
	}
	Release()
	total := 0
	for _, f := range free {
		total += cap(f)
	}
	if len(free) > maxFree || total > maxFreeBytes {
		t.Errorf("%d blocks and %d bytes kept, want at most %d and %d", len(free), total, maxFree, maxFreeBytes)
	}
}
