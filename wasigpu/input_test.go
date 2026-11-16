package wasigpu

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/go-wasmcontainers/gpu/wit"
)

// testMem is guest memory for wit.Load.
type testMem []byte

func (m testMem) Read(off, n uint32) ([]byte, bool) {
	if uint64(off)+uint64(n) > uint64(len(m)) {
		return nil, false
	}
	return m[off : off+n], true
}

func (m testMem) Write(off uint32, b []byte) bool {
	if uint64(off)+uint64(len(b)) > uint64(len(m)) {
		return false
	}
	copy(m[off:], b)
	return true
}

// A button-event as the Canonical ABI loads it from guest memory: the u8
// button is a uint64 (an assertion on uint32 read every button as 0, which
// the browser e2e caught: a right press arrived as button 0).
func TestDecodeButtonFromLoadedRecord(t *testing.T) {
	typ := surfaceIface.Types["button-event"]
	if typ == nil {
		t.Fatal("no button-event type")
	}
	mem := make(testMem, 64)
	mem[0] = 2
	binary.LittleEndian.PutUint64(mem[8:], math.Float64bits(30))
	binary.LittleEndian.PutUint64(mem[16:], math.Float64bits(40))
	v, err := wit.Load(mem, 0, typ)
	if err != nil {
		t.Fatal(err)
	}
	ev, err := decodeButton(v)
	if err != nil || ev != (ButtonEvent{Button: 2, X: 30, Y: 40}) {
		t.Errorf("%+v, %v; want button 2 at 30,40", ev, err)
	}
	if _, err := decodeButton("x"); err == nil {
		t.Error("a non-record decoded")
	}
}
