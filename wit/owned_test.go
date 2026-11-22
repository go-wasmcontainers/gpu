package wit

import (
	"bytes"
	"runtime"
	"testing"
)

// freshMem stands for the browser's memory: Read copies out of the guest's
// linear memory into a new array every time (one allocation of n bytes), and
// says so by ReadOwned.
type freshMem struct{ b []byte }

func (m freshMem) Read(off, n uint32) ([]byte, bool) {
	if uint64(off)+uint64(n) > uint64(len(m.b)) {
		return nil, false
	}
	return append([]byte(nil), m.b[off:off+n]...), true
}
func (m freshMem) Write(off uint32, b []byte) bool { return false }
func (m freshMem) ReadOwned(off, n uint32) ([]byte, bool) {
	return m.Read(off, n)
}

// aliasMem hands out a view of its own array, as a Memory that does not copy.
type aliasMem struct{ b []byte }

func (m aliasMem) Read(off, n uint32) ([]byte, bool) {
	if uint64(off)+uint64(n) > uint64(len(m.b)) {
		return nil, false
	}
	return m.b[off : off+n], true
}
func (m aliasMem) Write(off uint32, b []byte) bool { return false }

// TestListU8CopiesOnce: a list<u8> lifted from a memory that copies on Read is
// that copy: one allocation per list, not two. Every host call that carries
// bulk data (queue.write-buffer-with-copy, write-texture) allocated the data
// twice, which fed the collector on the engine's one thread.
func TestListU8CopiesOnce(t *testing.T) {
	data := bytes.Repeat([]byte{1, 2, 3, 4, 5, 6, 7, 8}, 8192) // 64 KiB
	elem := &Type{Kind: U8}
	var sink any
	m := &freshMem{data}
	lift := func() {
		v, err := loadList(m, 0, uint32(len(data)), elem)
		if err != nil {
			t.Fatal(err)
		}
		sink = v
	}
	// The array, and the slice header boxed in the any: 2. The copy made it 3.
	if n := testing.AllocsPerRun(50, lift); n > 2 {
		t.Errorf("a 64 KiB list<u8> from a copying memory allocates %.0f times, want 2 (the array and its box)", n)
	}
	// And the bytes: one array of the list's size, not two.
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	for i := 0; i < 100; i++ {
		lift()
	}
	runtime.ReadMemStats(&after)
	if per := (after.TotalAlloc - before.TotalAlloc) / 100; per > uint64(len(data))+uint64(len(data))/4 {
		t.Errorf("a 64 KiB list<u8> allocates %d bytes per lift, want about 65536 (one copy)", per)
	}
	got, _ := sink.([]byte)
	if !bytes.Equal(got, data) {
		t.Error("the lifted list differs from the guest's bytes")
	}
	if len(got) > 0 && &got[0] == &m.b[0] {
		t.Error("the lifted list is the guest's own array")
	}
	if _, err := loadList(m, 1, uint32(len(data)), elem); err == nil {
		t.Error("a list past the end of the memory was lifted")
	}
}

// TestListU8FromAViewIsCopied: a memory that returns a view of its own array
// (it does not say its Read is a copy) still gets a copy, so a later write by
// the guest does not change what the host holds.
func TestListU8FromAViewIsCopied(t *testing.T) {
	m := aliasMem{[]byte{1, 2, 3, 4}}
	v, err := loadList(m, 0, 4, &Type{Kind: U8})
	if err != nil {
		t.Fatal(err)
	}
	m.b[0] = 99
	if got := v.([]byte); got[0] != 1 {
		t.Errorf("the lifted list changed with the guest's memory: %v", got)
	}
}
