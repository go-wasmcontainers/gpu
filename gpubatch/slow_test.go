package gpubatch

import (
	"sync"
	"testing"
	"time"

	"github.com/go-gfx/webgpu"
	"github.com/go-gfx/webgpu/wire"
)

// A submit that runs SlowSubmitMin or more is reported with what its frame held:
// the size of the batch behind a slow host call.
func TestSlowSubmitSaysHowLargeTheFrameWas(t *testing.T) {
	var mu sync.Mutex
	var got []SubmitInfo
	SlowSubmit = func(i SubmitInfo) { mu.Lock(); got = append(got, i); mu.Unlock() }
	t.Cleanup(func() { SlowSubmit = nil })
	g := newGuest(t, gpuCaps, func(m *Module, p *fakeProvider) { p.dev.delay = 2 * SlowSubmitMin })
	c := g.newContext()
	frame := bufferFrame(t, 1, 64, webgpu.BufferUsageCopyDst, []uint32{uint32(wire.WriteBuffer), 1, 0, 0, 0, 4}, make([]byte, 4))
	if e := g.submit(c, frame, 4096, 0, 0); e != Success {
		t.Fatalf("errno %d", e)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(got) != 1 {
		t.Fatalf("%d slow submits reported, want 1", len(got))
	}
	i := got[0]
	if i.Bytes != len(frame) || i.Commands < 1 || i.Creates != 1 || i.Ran < SlowSubmitMin || i.Ran > time.Second {
		t.Fatalf("%+v: want the frame's %d bytes, at least one command, 1 create, a run of %v or more", i, len(frame), SlowSubmitMin)
	}
}

// A slow frame that creates objects says what each creation was, so a stall can
// be put on a kind (shader, pipeline, texture...) instead of on "the submit".
func TestSlowSubmitNamesEachCreation(t *testing.T) {
	var mu sync.Mutex
	var got []SubmitInfo
	SlowSubmit = func(i SubmitInfo) { mu.Lock(); got = append(got, i); mu.Unlock() }
	t.Cleanup(func() { SlowSubmit = nil })
	g := newGuest(t, gpuCaps, func(m *Module, p *fakeProvider) { p.dev.delay = 2 * SlowSubmitMin })
	c := g.newContext()
	frame := bufferFrame(t, 1, 64, webgpu.BufferUsageCopyDst, []uint32{uint32(wire.WriteBuffer), 1, 0, 0, 0, 4}, make([]byte, 4))
	if e := g.submit(c, frame, 4096, 0, 0); e != Success {
		t.Fatalf("errno %d", e)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(got) != 1 || len(got[0].Creations) != 1 {
		t.Fatalf("%+v: want one slow submit naming its one creation", got)
	}
	if k := got[0].Creations[0]; k.Kind != "buffer" || k.Bytes < 1 {
		t.Fatalf("%+v: want a buffer with a descriptor", k)
	}
}

func TestKindNameCoversTheWireKinds(t *testing.T) {
	for k := wire.KindBuffer; k <= wire.KindComputePipeline; k++ {
		if n := kindName(uint32(k)); n == "" || n[:5] == "kind-" {
			t.Errorf("kind %d has no name (%q)", k, n)
		}
	}
	if kindName(99) != "kind-99" {
		t.Errorf("an unknown kind is named %q", kindName(99))
	}
}

// A fast submit is not reported: the list stays the slow calls only.
func TestFastSubmitIsNotReported(t *testing.T) {
	n := 0
	SlowSubmit = func(SubmitInfo) { n++ }
	t.Cleanup(func() { SlowSubmit = nil })
	g := newGuest(t, gpuCaps, nil)
	c := g.newContext()
	if e := g.submit(c, bufferFrame(t, 1, 64, webgpu.BufferUsageCopyDst, nil, nil), 4096, 0, 0); e != Success {
		t.Fatalf("errno %d", e)
	}
	if n != 0 {
		t.Fatalf("%d reports for a fast submit", n)
	}
}
