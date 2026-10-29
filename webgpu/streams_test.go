package webgpu

import (
	"encoding/binary"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/go-wasmcontainers/engine/rootfs"
	"github.com/go-wasmcontainers/engine/wasi"
	"github.com/go-wasmcontainers/gpu/wit"
)

// streamRig is one process with the four modules open and a guest memory
// whose cabi_realloc is a bump allocator from 4096.
type streamRig struct {
	t      *testing.T
	mods   []*Module
	ins    []*instance
	mem    mem
	bump   uint32
	allocs int
}

func newRig(t *testing.T) *streamRig {
	r := &streamRig{t: t, mods: NewModules(nil), mem: make(mem, 1<<16), bump: 4096}
	p := wasi.New(wasi.Config{FS: rootfs.New()})
	for _, m := range r.mods {
		in := m.Open(p).(*instance)
		r.ins = append(r.ins, in)
		t.Cleanup(in.Close)
	}
	return r
}

var (
	surfaceIface = SurfaceInterfaces["surface"]
	pointerT     = surfaceIface.Types["pointer-event"]
	keyT         = surfaceIface.Types["key-event"]
)

// stream registers a stream of elem in the process's table.
func (r *streamRig) stream(elem *wit.Type) (*Stream, uint64) {
	s := NewStream(elem)
	n, err := r.ins[0].h.New("stream", s)
	if err != nil {
		r.t.Fatal(err)
	}
	return s, uint64(n)
}

// call runs the import name of module mod (1: surface, 3: $root),
// servicing reallocs, and returns its final result or the Wait one.
func (r *streamRig) call(mod int, name string, args ...uint64) wasi.Result {
	r.t.Helper()
	return r.finish(r.ins[mod].Call(index(r.t, r.mods[mod], name), args, r.mem))
}

func (r *streamRig) finish(res wasi.Result) wasi.Result {
	for res.Realloc != nil {
		p := (r.bump + res.Realloc.Align - 1) / res.Realloc.Align * res.Realloc.Align
		r.bump = p + res.Realloc.Size
		r.allocs++
		res = res.Realloc.Then(p, r.mem)
	}
	return res
}

// resume waits for a Wait result (at most a second) and resumes it.
func (r *streamRig) resume(res wasi.Result) wasi.Result {
	r.t.Helper()
	if res.Wait == nil {
		r.t.Fatalf("not waiting: %+v", res)
	}
	select {
	case <-res.Wait:
	case <-time.After(time.Second):
		r.t.Fatal("still waiting after 1s")
	}
	return r.finish(res.Resume(r.mem))
}

func (r *streamRig) pointerAt(p uint32) [2]float64 {
	return [2]float64{
		math.Float64frombits(binary.LittleEndian.Uint64(r.mem[p:])),
		math.Float64frombits(binary.LittleEndian.Uint64(r.mem[p+8:])),
	}
}

const (
	readDown  = "[stream-read-0][method]surface.on-pointer-down"
	readDownA = "[async-lower][stream-read-0][method]surface.on-pointer-down"
	cancelDn  = "[stream-cancel-read-0][method]surface.on-pointer-down"
	dropDown  = "[stream-drop-readable-0][method]surface.on-pointer-down"
	readKeyA  = "[async-lower][stream-read-0][method]surface.on-key-down"
)

func TestStreamSyncRead(t *testing.T) {
	r := newRig(t)
	s, end := r.stream(pointerT)
	s.Push([]any{1.5, 2.5})
	s.Push([]any{3.0, 4.0})
	res := r.call(1, readDown, end, 64, 4)
	if res.Trap != "" || res.Ret != 2<<4|copyCompleted {
		t.Fatalf("read: %+v", res)
	}
	if a, b := r.pointerAt(64), r.pointerAt(80); a != [2]float64{1.5, 2.5} || b != [2]float64{3, 4} {
		t.Errorf("elements %v %v", a, b)
	}
	// Nothing queued: the synchronous read blocks until an event.
	res = r.call(1, readDown, end, 64, 4)
	if res.Wait == nil {
		t.Fatalf("an empty synchronous read did not block: %+v", res)
	}
	go s.Push([]any{5.0, 6.0})
	if res = r.resume(res); res.Ret != 1<<4 || r.pointerAt(64) != [2]float64{5, 6} {
		t.Errorf("after the event: %+v %v", res, r.pointerAt(64))
	}
	// The writer goes away: what is queued, then DROPPED, then the end is done.
	s.Push([]any{7.0, 8.0})
	s.CloseWriter()
	if res = r.call(1, readDown, end, 64, 4); res.Ret != 1<<4 {
		t.Errorf("queued before close: %+v", res)
	}
	if res = r.call(1, readDown, end, 64, 4); res.Ret != copyDropped {
		t.Errorf("after close: %+v", res)
	}
	if res = r.call(1, readDown, end, 64, 4); res.Trap == "" {
		t.Error("read of a finished stream accepted")
	}
}

func TestStreamAsyncPollWait(t *testing.T) {
	r := newRig(t)
	s, end := r.stream(pointerT)
	set := r.call(3, "[waitable-set-new]").Ret
	if set == 0 {
		t.Fatal("no set")
	}
	if res := r.call(3, "[waitable-join]", end, set); res.Trap != "" {
		t.Fatal(res.Trap)
	}
	if res := r.call(1, readDownA, end, 128, 8); res.Ret != blocked {
		t.Fatalf("async read of an empty stream: %+v", res)
	}
	if res := r.call(1, readDownA, end, 128, 8); res.Trap == "" {
		t.Error("a second read while one is pending accepted")
	}
	if res := r.call(3, "[waitable-set-poll]", set, 8); res.Ret != eventNone {
		t.Errorf("poll with nothing: %+v", res)
	}
	s.Push([]any{9.0, 10.0})
	res := r.call(3, "[waitable-set-poll]", set, 8)
	idx, payload := binary.LittleEndian.Uint32(r.mem[8:]), binary.LittleEndian.Uint32(r.mem[12:])
	if res.Ret != eventStreamRead || uint64(idx) != end || payload != 1<<4 || r.pointerAt(128) != [2]float64{9, 10} {
		t.Fatalf("poll: code %d, end %d, payload %#x, element %v", res.Ret, idx, payload, r.pointerAt(128))
	}
	// wait blocks until the next read completes.
	r.call(1, readDownA, end, 128, 8)
	res = r.call(3, "[waitable-set-wait]", set, 8)
	go s.Push([]any{11.0, 12.0})
	if res = r.resume(res); res.Ret != eventStreamRead || r.pointerAt(128) != [2]float64{11, 12} {
		t.Errorf("wait: %+v %v", res, r.pointerAt(128))
	}
	// A synchronous read of a stream in a set traps (it could lose its event).
	if res := r.call(1, readDown, end, 128, 8); res.Trap == "" {
		t.Error("synchronous read of a stream in a set accepted")
	}
	if res := r.call(3, "[waitable-set-drop]", set); res.Trap == "" {
		t.Error("drop of a non-empty set accepted")
	}
	r.call(3, "[waitable-join]", end, 0)
	if res := r.call(3, "[waitable-set-drop]", set); res.Trap != "" {
		t.Error(res.Trap)
	}
}

func TestStreamCancelAndDrop(t *testing.T) {
	r := newRig(t)
	s, end := r.stream(pointerT)
	if res := r.call(1, cancelDn, end); res.Trap == "" {
		t.Error("cancel without a pending read accepted")
	}
	r.call(1, readDownA, end, 64, 1)
	if res := r.call(1, dropDown, end); res.Trap == "" {
		t.Error("drop during a pending read accepted")
	}
	if res := r.call(1, cancelDn, end); res.Ret != copyCancelled {
		t.Errorf("cancel: %+v", res)
	}
	// A cancel that finds an event delivers it instead.
	r.call(1, readDownA, end, 64, 1)
	s.Push([]any{1.0, 1.0})
	if res := r.call(1, cancelDn, end); res.Ret != 1<<4 {
		t.Errorf("cancel with an event: %+v", res)
	}
	closed := false
	s.SetOnClose(func() { closed = true })
	if res := r.call(1, dropDown, end); res.Trap != "" || !closed {
		t.Errorf("drop: %+v, unsubscribed %v", res, closed)
	}
	if res := r.call(1, readDown, end, 64, 1); res.Trap == "" {
		t.Error("read of a dropped stream accepted")
	}
}

func TestStreamTraps(t *testing.T) {
	r := newRig(t)
	_, end := r.stream(pointerT)
	for _, c := range []struct {
		name string
		args []uint64
		want string
	}{
		{readDown, []uint64{end, 65, 1}, "misaligned"},
		{readDown, []uint64{end, 1 << 16, 1}, "out of bounds"},
		{readDown, []uint64{end, 0, 1 << 31}, "elements"},
		{readDown, []uint64{999, 64, 1}, "invalid stream"},
		{readKeyA, []uint64{end, 64, 1}, "another element type"},
	} {
		if res := r.call(1, c.name, c.args...); !strings.Contains(res.Trap, c.want) {
			t.Errorf("%s %v: %+v, want a trap with %q", c.name, c.args, res, c.want)
		}
	}
	if res := r.call(3, "[waitable-set-poll]", 999, 8); res.Trap == "" {
		t.Error("poll of an invalid set accepted")
	}
	set := r.call(3, "[waitable-set-new]").Ret
	if res := r.call(3, "[waitable-set-poll]", set, 6); !strings.Contains(res.Trap, "misaligned") {
		t.Errorf("misaligned event: %+v", res)
	}
	if res := r.call(3, "[waitable-set-poll]", set, 1<<16-4); !strings.Contains(res.Trap, "out of bounds") {
		t.Errorf("event out of bounds: %+v", res)
	}
}

func TestStreamQueueBound(t *testing.T) {
	r := newRig(t)
	s, end := r.stream(pointerT)
	for i := range MaxQueued + 44 {
		s.Push([]any{float64(i), 0.0})
	}
	if s.Dropped() != 44 {
		t.Errorf("%d dropped, want 44", s.Dropped())
	}
	res := r.call(1, readDown, end, 64, MaxQueued+44)
	if res.Ret != MaxQueued<<4 || r.pointerAt(64)[0] != 44 {
		t.Errorf("read %#x, first x %v: the oldest are the ones dropped", res.Ret, r.pointerAt(64)[0])
	}
}

// TestStreamKeyText: key events carry an optional string, which the host
// stores through the guest's cabi_realloc.
func TestStreamKeyText(t *testing.T) {
	r := newRig(t)
	s, end := r.stream(keyT)
	ka, _ := KeyFromCode("KeyA")
	s.Push([]any{wit.Some{Value: ka}, wit.Some{Value: "é"}, false, true, false, true})
	s.Push([]any{nil, nil, true, false, false, false})
	res := r.call(1, readKeyA, end, 256, 2)
	if res.Ret != 2<<4 || r.allocs != 1 {
		t.Fatalf("read: %+v, %d allocations", res, r.allocs)
	}
	v, err := wit.Load(r.mem, 256, keyT)
	if err != nil {
		t.Fatal(err)
	}
	f := v.([]any)
	if f[0] != (wit.Some{Value: ka}) || f[1] != (wit.Some{Value: "é"}) || f[3] != true || f[5] != true {
		t.Errorf("first key event %v", f)
	}
	v, _ = wit.Load(r.mem, 256+wit.Size(keyT), keyT)
	if f := v.([]any); f[0] != nil || f[1] != nil || f[2] != true {
		t.Errorf("second key event %v", f)
	}
}

func TestKeyFromCode(t *testing.T) {
	labels := surfaceIface.Types["key"].Labels
	for code, want := range map[string]string{
		"KeyA": "key-a", "Digit0": "digit0", "ArrowUp": "arrow-up", "NumpadClearEntry": "numpad-clear-entry",
		"F12": "f12", "IntlRo": "intl-ro", "LaunchApp1": "launch-app1", "AudioVolumeDown": "audio-volume-down",
		"Space": "space", "ContextMenu": "context-menu", "FnLock": "fn-lock",
	} {
		n, ok := KeyFromCode(code)
		if !ok || labels[n] != want {
			t.Errorf("%s: %d %v, want %s", code, n, ok, want)
		}
	}
	for _, code := range []string{"", "Unidentified", "keyA", "Foo", strings.Repeat("A", 40)} {
		if _, ok := KeyFromCode(code); ok {
			t.Errorf("%q mapped", code)
		}
	}
}

// TestStreamProcessEnd: a guest blocked on a read or a wait is woken when
// its process ends (the handles close), and does not leak.
func TestStreamProcessEnd(t *testing.T) {
	r := newRig(t)
	_, end := r.stream(pointerT)
	res := r.call(1, readDown, end, 64, 1)
	set := r.call(3, "[waitable-set-new]").Ret
	wres := r.call(3, "[waitable-set-wait]", set, 8)
	r.ins[0].h.closeAll()
	for _, w := range []wasi.Result{res, wres} {
		select {
		case <-w.Wait:
		case <-time.After(time.Second):
			t.Fatal("a waiter was not woken at process end")
		}
	}
}
