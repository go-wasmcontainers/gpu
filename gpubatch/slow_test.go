package gpubatch

import (
	"sync"
	"testing"
	"time"

	"github.com/go-gfx/webgpu"
	"github.com/go-gfx/webgpu/wire"
	"github.com/go-virtio/gpu/wgpubatch"
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

// Created counts the objects a submit creates, fast or slow, and a steady frame
// (a stream of commands on objects that exist) creates none: the count does not
// move across it. A page reads this to assert 0 creations per frame.
func TestCreatedCountsCreationsAndASteadyFrameAddsNone(t *testing.T) {
	total := 0
	Created = func(n int) { total += n }
	t.Cleanup(func() { Created = nil })
	g := newGuest(t, gpuCaps, nil)
	c := g.newContext()
	if e := g.submit(c, bufferFrame(t, 1, 64, webgpu.BufferUsageCopyDst, nil, nil), 4096, 0, 0); e != Success {
		t.Fatalf("errno %d", e)
	}
	if total != 1 {
		t.Fatalf("after the creating frame the count is %d, want 1", total)
	}
	e := wgpubatch.NewEncoder(1 << 20)
	e.Stream([]uint32{uint32(wire.WriteBuffer), 1, 0, 0, 0, 4}, make([]byte, 4))
	f, err := e.Frame()
	if err != nil {
		t.Fatal(err)
	}
	steady := append([]byte(nil), f...)
	for i := 0; i < 5; i++ {
		if e := g.submit(c, steady, 4096, 0, 0); e != Success {
			t.Fatalf("steady frame %d: errno %d", i, e)
		}
	}
	if total != 1 {
		t.Fatalf("five steady frames moved the count to %d, want it to stay 1", total)
	}
}

// A fast submit is not reported: the list stays the slow calls only.
func TestFastSubmitIsNotReported(t *testing.T) {
	n := 0
	// A fast submit is one under the threshold; a wall-clock 10 ms is not fast on
	// a loaded runner under -race (run 45083368019), so the threshold is raised.
	min := SlowSubmitMin
	SlowSubmitMin = time.Minute
	SlowSubmit = func(SubmitInfo) { n++ }
	t.Cleanup(func() { SlowSubmit = nil; SlowSubmitMin = min })
	g := newGuest(t, gpuCaps, nil)
	c := g.newContext()
	if e := g.submit(c, bufferFrame(t, 1, 64, webgpu.BufferUsageCopyDst, nil, nil), 4096, 0, 0); e != Success {
		t.Fatalf("errno %d", e)
	}
	if n != 0 {
		t.Fatalf("%d reports for a fast submit", n)
	}
}
