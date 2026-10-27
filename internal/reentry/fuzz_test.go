package reentry

import (
	"math/big"
	"testing"
)

// FuzzRealloc checks CheckBlock against arbitrary-precision arithmetic: a
// block is accepted exactly when it is aligned and p+size <= memSize.
func FuzzRealloc(f *testing.F) {
	f.Add(uint32(0), uint32(1), uint32(0), uint32(65536))
	f.Add(uint32(65535), uint32(1), uint32(2), uint32(65536))
	f.Add(uint32(0xfffffff0), uint32(16), uint32(0x20), uint32(0xffffffff))
	f.Add(uint32(8), uint32(4), uint32(65528), uint32(65536))
	f.Fuzz(func(t *testing.T, p, align, size, mem uint32) {
		end := new(big.Int).Add(big.NewInt(int64(p)), big.NewInt(int64(size)))
		want := end.Cmp(big.NewInt(int64(mem))) <= 0 && (align <= 1 || p%align == 0)
		got := CheckBlock(p, align, size, mem) == nil
		if got != want {
			t.Fatalf("CheckBlock(%#x, %d, %d, %d) accepted=%v, want %v", p, align, size, mem, got, want)
		}
	})
}

func TestCheckBlock(t *testing.T) {
	for _, c := range []struct {
		p, align, size, mem uint32
		ok                  bool
	}{
		{0, 1, 0, 0, true},
		{0, 1, 1, 0, false},
		{4, 8, 1, 64, false},
		{8, 8, 56, 64, true},
		{8, 8, 57, 64, false},
		{0xffffffff, 1, 1, 0xffffffff, false},
	} {
		if got := CheckBlock(c.p, c.align, c.size, c.mem) == nil; got != c.ok {
			t.Errorf("CheckBlock(%#x, %d, %d, %d) = %v, want %v", c.p, c.align, c.size, c.mem, got, c.ok)
		}
	}
}
