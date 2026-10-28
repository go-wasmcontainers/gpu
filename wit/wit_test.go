package wit

import (
	"encoding/binary"
	"os"
	"reflect"
	"slices"
	"testing"
)

func webgpu(t testing.TB) *Interface {
	t.Helper()
	b, err := os.ReadFile("../webgpu/wit/webgpu.wit")
	if err != nil {
		t.Fatal(err)
	}
	ifs, err := Parse(string(b))
	if err != nil {
		t.Fatal(err)
	}
	i := ifs["webgpu"]
	if i == nil {
		t.Fatalf("no interface webgpu in %v", ifs)
	}
	return i
}

func method(t testing.TB, i *Interface, res, name string) *Func {
	t.Helper()
	r := i.Resources[res]
	if r == nil {
		t.Fatalf("no resource %s", res)
	}
	for _, f := range r.Funcs {
		if f.Name == name {
			return f
		}
	}
	t.Fatalf("no %s.%s", res, name)
	return nil
}

func TestParseWebGPU(t *testing.T) {
	i := webgpu(t)
	if i.Package != "wasi:webgpu@0.3.0-rc.2" || i.ImportModule() != "wasi:webgpu/webgpu@0.3.0-rc.2" {
		t.Errorf("package %q, module %q", i.Package, i.ImportModule())
	}
	if len(i.Resources) < 25 || len(i.Funcs) != 1 || i.Funcs[0].Name != "get-gpu" {
		t.Errorf("%d resources, funcs %v", len(i.Resources), i.Funcs)
	}
	f := method(t, i, "gpu-device", "create-buffer")
	if f.ImportName("gpu-device") != "[method]gpu-device.create-buffer" || f.Params[0].Type.Kind != Borrow {
		t.Errorf("create-buffer: %q %+v", f.ImportName("gpu-device"), f.Params[0])
	}
	if ra := method(t, i, "gpu", "request-adapter"); !ra.Async || ra.Result.Kind != Option || ra.Result.Elem.Kind != Own {
		t.Errorf("request-adapter: %+v", ra)
	}
	// gpu-buffer-descriptor { size: u64, usage: flags(10), mapped-at-creation:
	// option<bool>, label: option<string> }: u64@0, u16@8, option<bool>
	// (2 bytes, align 1)@10, option<string> (12 bytes, align 4)@12 -> 24.
	d := i.Types["gpu-buffer-descriptor"]
	if Size(d) != 24 || Align(d) != 8 {
		t.Errorf("descriptor size %d align %d, want 24 and 8", Size(d), Align(d))
	}
	if got := string(Flatten(d)); got != "Iiiiiii" {
		t.Errorf("descriptor flattens to %q", got)
	}
	s := Sig(f)
	if string(s.Params) != "iIiiiiii" || string(s.Results) != "i" || s.RetPtr || s.ParamsInMemory {
		t.Errorf("create-buffer core signature %+v", s)
	}
	// get-mapped-range-get-with-copy -> result<list<u8>, error-record>: more
	// than one flat result, so a return area.
	if s := Sig(method(t, i, "gpu-buffer", "get-mapped-range-get-with-copy")); !s.RetPtr || len(s.Results) != 0 {
		t.Errorf("get-mapped-range signature %+v", s)
	}
}

type mem []byte

func (m mem) Read(off, n uint32) ([]byte, bool) {
	if uint64(off)+uint64(n) > uint64(len(m)) {
		return nil, false
	}
	return m[off : off+n], true
}

func (m mem) Write(off uint32, b []byte) bool {
	if uint64(off)+uint64(len(b)) > uint64(len(m)) {
		return false
	}
	copy(m[off:], b)
	return true
}

// bump allocates from the top half of a test memory.
type bump struct{ next uint32 }

func (b *bump) Alloc(align, size uint32) (uint32, error) {
	p := alignTo(b.next, max(align, 1))
	b.next = p + size
	return p, nil
}

func TestStoreLoadRoundTrip(t *testing.T) {
	i := webgpu(t)
	d := i.Types["gpu-buffer-descriptor"]
	v := []any{uint64(1 << 20), uint64(0x88), Some{true}, Some{"weights"}}
	m := make(mem, 4096)
	a := &bump{next: 1024}
	if err := Store(m, a, 64, d, v); err != nil {
		t.Fatal(err)
	}
	got, err := Load(m, 64, d)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, v) {
		t.Errorf("round trip: %#v, want %#v", got, v)
	}
	// Flat: lower then lift through a function's parameters.
	f := method(t, i, "gpu-device", "create-buffer")
	flat, err := LowerFlat(m, a, d, v)
	if err != nil {
		t.Fatal(err)
	}
	args := append([]uint64{7}, flat...)
	ps, err := LiftParams(f, args, m)
	if err != nil {
		t.Fatal(err)
	}
	if ps[0] != uint32(7) || !reflect.DeepEqual(ps[1], v) {
		t.Errorf("lifted %#v", ps)
	}
	// none options lower to zeros and lift back.
	v2 := []any{uint64(16), uint64(0x80), nil, nil}
	flat, _ = LowerFlat(m, a, d, v2)
	ps, err = LiftParams(f, append([]uint64{1}, flat...), m)
	if err != nil || !reflect.DeepEqual(ps[1], v2) {
		t.Errorf("none options: %#v %v", ps, err)
	}
}

// TestPlanApply: a result with a list goes through the return area, its
// allocation is planned without writing, then applied.
func TestPlanApply(t *testing.T) {
	i := webgpu(t)
	f := method(t, i, "gpu-buffer", "get-mapped-range-get-with-copy")
	m := make(mem, 4096)
	v := ResultValue{Value: []byte{1, 2, 3, 4, 5}}
	l, err := PlanResult(f, v, []uint64{1, 0, 0, 0, 0, 512})
	if err != nil {
		t.Fatal(err)
	}
	if len(l.Reqs) != 1 || l.Reqs[0] != (AllocReq{1, 5}) {
		t.Fatalf("requests %v", l.Reqs)
	}
	if !slices.Equal(m[512:520], make([]byte, 8)) {
		t.Fatal("the plan wrote to memory")
	}
	if _, err := l.Apply(m, []uint32{2000}); err != nil {
		t.Fatal(err)
	}
	// result<list<u8>, E>: discriminant 0 at 512, then (ptr, len) at 516.
	if m[512] != 0 || binary.LittleEndian.Uint32(m[516:]) != 2000 || binary.LittleEndian.Uint32(m[520:]) != 5 || !slices.Equal(m[2000:2005], []byte{1, 2, 3, 4, 5}) {
		t.Errorf("memory %v / %v", m[512:524], m[2000:2005])
	}
	if _, err := l.Apply(m, []uint32{4095}); err == nil {
		t.Error("an allocation outside memory was accepted")
	}
	if _, err := PlanResult(f, v, []uint64{1, 0, 0, 0, 0, 513}); err == nil {
		t.Error("a misaligned return area was accepted")
	}
}

func TestTraps(t *testing.T) {
	i := webgpu(t)
	d := i.Types["gpu-buffer-descriptor"]
	m := make(mem, 64)
	if _, err := Load(m, 4, d); err == nil {
		t.Error("misaligned load accepted")
	}
	if _, err := Load(m, 56, d); err == nil {
		t.Error("out-of-bounds load accepted")
	}
	m[10] = 7 // option discriminant 7
	if _, err := Load(m, 0, d); err == nil {
		t.Error("bad discriminant accepted")
	}
	m[10] = 0
	m[12], m[16], m[20] = 1, 60, 10 // some(string at 60, 10 bytes): out of bounds
	if _, err := Load(m, 0, d); err == nil {
		t.Error("string out of bounds accepted")
	}
}

// FuzzLoad: any memory, any type of the interface, any address: a value
// or an error, never a panic.
func FuzzLoad(f *testing.F) {
	i := webgpu(f)
	var types []*Type
	for _, t := range i.Types {
		types = append(types, t)
	}
	slices.SortFunc(types, func(a, b *Type) int { return int(Size(a)) - int(Size(b)) })
	f.Add([]byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, uint8(0), uint16(0))
	f.Fuzz(func(t *testing.T, b []byte, ti uint8, p uint16) {
		m := mem(append(b, make([]byte, 256)...))
		ty := types[int(ti)%len(types)]
		_, _ = Load(m, uint32(p)&^7, ty)
	})
}
