package cabi

import "testing"

// A frame-sized argument allocated on every host call used to cost one
// 250 KiB block per call: Release now keeps the blocks and Alloc reuses them.
func TestReleasedBlocksAreReused(t *testing.T) {
	const frame = 256000
	Release()
	free = nil
	Alloc(4, frame) // the first call allocates
	Release()
	if n := testing.AllocsPerRun(100, func() {
		Alloc(4, frame)
		Alloc(8, 64)
		Release()
	}); n > 1 { // the small block is allowed to miss once, never the frame
		t.Errorf("Alloc and Release of a frame allocate %.0f times per call, want 0 (1 at most)", n)
	}
}

func TestReusedBlockIsZeroed(t *testing.T) {
	free = nil
	Alloc(4, 1024)
	for i := range pinned[0] {
		pinned[0][i] = 0xFF
	}
	Release()
	Alloc(4, 1000)
	for i, v := range pinned[0] {
		if v != 0 {
			t.Fatalf("byte %d of a reused block is %#x, want 0", i, v)
		}
	}
	Release()
}

func TestFreeListIsBounded(t *testing.T) {
	free = nil
	for range 20 {
		Alloc(4, 1<<20)
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
