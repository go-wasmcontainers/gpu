package wit

import "testing"

// Size and Align of a type are asked on every host call that lowers or lifts
// it; for an enum they used to build a slice of one Case per label (101 for
// wasi:webgpu's gpu-texture-format) each time, and a render pass descriptor
// asks several times. In the browser the quake image ran this per frame
// (go-wasmcontainers/playground#56: the stack of an out-of-memory death ended in
// wit.Align -> cases -> makeslice).
func TestEnumLayoutDoesNotAllocate(t *testing.T) {
	labels := make([]string, 101)
	for i := range labels {
		labels[i] = "label"
	}
	enum := &Type{Kind: Enum, Labels: labels}
	rec := &Type{Kind: Record, Fields: []Field{{Name: "a", Type: enum}, {Name: "b", Type: enum}, {Name: "c", Type: &Type{Kind: U64}}}}
	if got := Size(enum); got != 1 {
		t.Fatalf("Size(enum of 101) = %d, want 1", got)
	}
	if got := Align(enum); got != 1 {
		t.Fatalf("Align(enum of 101) = %d, want 1", got)
	}
	big := &Type{Kind: Enum, Labels: make([]string, 300)}
	if Size(big) != 2 || Align(big) != 2 {
		t.Fatalf("enum of 300: size %d align %d, want 2 and 2", Size(big), Align(big))
	}
	if got := Size(rec); got != 16 {
		t.Fatalf("Size(record{enum, enum, u64}) = %d, want 16", got)
	}
	var sink uint32
	if n := testing.AllocsPerRun(200, func() { sink += Size(rec) + Align(rec) + Size(enum) + Align(enum) }); n != 0 {
		t.Errorf("Size and Align of enums allocate %.0f times per run, want 0", n)
	}
	_ = sink
}
