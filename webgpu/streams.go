package webgpu

import (
	"reflect"
	"strings"
	"sync"

	"github.com/go-wasmcontainers/engine/wasi"
	"github.com/go-wasmcontainers/gpu/wit"
)

// Streams the host writes and a core-module guest reads, with the
// Canonical ABI's built-ins as wit-component names their core imports
// (component-model design/mvp/CanonicalABI.md, "canon stream.read" and
// "canon waitable-set.*"; wasm-tools crates/wit-component/src/validation.rs):
//
//	module of the function F returning stream<T>:
//	  [stream-read-0]F               (end, ptr, n) -> payload   blocks until progress
//	  [async-lower][stream-read-0]F  (end, ptr, n) -> payload or BLOCKED
//	  [stream-cancel-read-0]F, [async-lower][stream-cancel-read-0]F  (end) -> payload
//	  [stream-drop-readable-0]F      (end)
//	module "$root":
//	  [waitable-set-new] () -> set     [waitable-set-drop] (set)
//	  [waitable-join] (end, set)       0 as set leaves any set
//	  [waitable-set-wait] (set, ptr) -> event code   blocks
//	  [waitable-set-poll] (set, ptr) -> event code   EVENT_NONE (0) if none
//
// A payload is (elements copied << 4) | COMPLETED (0), DROPPED (1, the
// writer is gone) or CANCELLED (2); an event stores (end, payload) at ptr.
// The host touches guest memory only inside a call, so an asynchronous
// read's elements are written when the guest receives its event (poll,
// wait, or cancel): the buffer is the guest's again only then, so it
// cannot tell the difference.

// Canonical ABI constants.
const (
	eventNone       = wit.EventNone
	eventStreamRead = wit.EventStreamRead
	copyCompleted   = wit.CopyCompleted
	copyDropped     = wit.CopyDropped
	copyCancelled   = wit.CopyCancelled
	blocked         = wit.Blocked
)

// MaxQueued bounds the events a stream keeps for a guest that does not
// read them: past it the oldest is dropped (and counted).
const MaxQueued = 256

// RootModule is the module of the waitable-set built-ins.
const RootModule = wit.RootModule

type endState int

const (
	endIdle endState = iota
	endCopying
	endDone
)

// Stream is the readable end of a stream the host writes (a surface's
// input events).
type Stream struct {
	Elem *wit.Type

	mu      sync.Mutex
	queue   []any
	dropped uint64
	closed  bool // the writer is gone
	state   endState
	ptr, n  uint32 // the buffer of a pending asynchronous read
	set     *WaitSet
	wake    chan struct{}
	onClose func() // unsubscribes, once the guest drops the end
	once    sync.Once
}

// NewStream makes a readable end of elements of type elem.
func NewStream(elem *wit.Type) *Stream {
	return &Stream{Elem: elem, wake: make(chan struct{})}
}

// signal wakes whoever waits on s or its set; s.mu is held.
func (s *Stream) signal() {
	close(s.wake)
	s.wake = make(chan struct{})
	if s.set != nil {
		s.set.signal()
	}
}

// Push writes one event (a value of s.Elem); it never blocks.
func (s *Stream) Push(v any) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed || s.state == endDone {
		return
	}
	s.queue = append(s.queue, v)
	if len(s.queue) > MaxQueued {
		s.queue = s.queue[1:]
		s.dropped++
	}
	s.signal()
}

// CloseWriter ends the stream: once the queue is read, reads report
// DROPPED.
func (s *Stream) CloseWriter() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.closed {
		s.closed = true
		s.signal()
	}
}

// Dropped is how many events were lost to MaxQueued.
func (s *Stream) Dropped() uint64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.dropped
}

// SetOnClose sets what runs when the guest drops the end (or its process
// ends).
func (s *Stream) SetOnClose(f func()) {
	s.mu.Lock()
	s.onClose = f
	s.mu.Unlock()
}

// Destroy implements Destroyer.
func (s *Stream) Destroy() {
	s.once.Do(func() {
		s.mu.Lock()
		f := s.onClose
		s.closed = true
		if s.set != nil {
			s.set.remove(s)
			s.set = nil
		}
		s.signal()
		s.mu.Unlock()
		if f != nil {
			f()
		}
	})
}

// ready reports whether a copy can finish now; s.mu is held.
func (s *Stream) ready() bool { return len(s.queue) > 0 || s.closed }

// take finishes a copy of up to n elements: what to store and the
// payload; s.mu is held.
func (s *Stream) take(n uint32) ([]any, uint32) {
	if len(s.queue) == 0 {
		// closed, nothing left
		s.state = endDone
		return nil, copyDropped
	}
	k := min(int(n), len(s.queue))
	out := append([]any(nil), s.queue[:k]...)
	s.queue = s.queue[k:]
	s.state = endIdle
	return out, uint32(k)<<4 | copyCompleted
}

// WaitSet is a waitable set: the streams a guest waits on together.
type WaitSet struct {
	mu      sync.Mutex
	members []*Stream
	next    int // round robin start for fairness
	wake    chan struct{}
	waiting bool
}

func newWaitSet() *WaitSet { return &WaitSet{wake: make(chan struct{})} }

func (w *WaitSet) signal() {
	w.mu.Lock()
	close(w.wake)
	w.wake = make(chan struct{})
	w.mu.Unlock()
}

func (w *WaitSet) add(s *Stream) {
	w.mu.Lock()
	w.members = append(w.members, s)
	w.mu.Unlock()
}

func (w *WaitSet) remove(s *Stream) {
	w.mu.Lock()
	for i, m := range w.members {
		if m == s {
			w.members = append(w.members[:i], w.members[i+1:]...)
			break
		}
	}
	w.mu.Unlock()
}

// Destroy implements Destroyer: the process ended; wake a waiter.
func (w *WaitSet) Destroy() { w.signal() }

// streamIO serves the built-ins for one process, over its handle table.
type streamIO struct{ h *Handles }

func (io streamIO) stream(n uint64, elem *wit.Type) (*Stream, error) {
	o, err := io.h.Get(uint32(n), "stream")
	if err != nil {
		return nil, err
	}
	s := o.(*Stream)
	if !sameType(s.Elem, elem) {
		return nil, Trapf("stream %d has another element type", n)
	}
	return s, nil
}

// sameType compares types structurally, as the Canonical ABI does: each
// use of a named type is its own *wit.Type, which carries the name while
// the definition does not, so the top-level name is ignored (except for a
// resource handle, where it is the type).
func sameType(a, b *wit.Type) bool {
	if a == b {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	x, y := *a, *b
	if x.Kind != wit.Own && x.Kind != wit.Borrow {
		x.Name, y.Name = "", ""
	}
	return reflect.DeepEqual(x, y)
}

// deliver stores elems at ptr (obtaining any allocations they need from
// the guest's cabi_realloc) and then returns what then gives.
func deliver(elem *wit.Type, ptr uint32, elems []any, m wasi.Memory, then func(m wasi.Memory) wasi.Result) wasi.Result {
	if len(elems) == 0 {
		return then(m)
	}
	l, err := wit.PlanElems(ptr, elem, elems)
	if err != nil {
		return trapResult(err)
	}
	return applyLowering(l, m, func(_ uint64, m wasi.Memory) wasi.Result { return then(m) })
}

// read is [stream-read-0]F (async: [async-lower]...).
func (io streamIO) read(elem *wit.Type, async bool, args []uint64, m wasi.Memory) wasi.Result {
	s, err := io.stream(args[0], elem)
	if err != nil {
		return trapResult(err)
	}
	ptr, n := uint32(args[1]), uint32(args[2])
	if ptr%wit.Align(elem) != 0 {
		return trapResult(Trapf("misaligned stream buffer at %#x", ptr))
	}
	// Bound the size in 64 bits first, so n*size cannot wrap.
	if size := uint64(n) * uint64(wit.Size(elem)); size > wit.MaxListBytes {
		return trapResult(Trapf("stream buffer of %d elements", n))
	} else if _, ok := m.Read(ptr, uint32(size)); !ok {
		return trapResult(Trapf("stream buffer of %d elements at %#x out of bounds", n, ptr))
	}
	s.mu.Lock()
	if s.state != endIdle {
		s.mu.Unlock()
		return trapResult(Trapf("stream %d is not idle", args[0]))
	}
	if !async && s.set != nil {
		s.mu.Unlock()
		return trapResult(Trapf("synchronous read of stream %d, which is in a waitable set", args[0]))
	}
	if s.ready() {
		elems, payload := s.take(n)
		s.mu.Unlock()
		return deliver(elem, ptr, elems, m, func(wasi.Memory) wasi.Result { return wasi.Result{Ret: uint64(payload)} })
	}
	if async {
		s.state, s.ptr, s.n = endCopying, ptr, n
		s.mu.Unlock()
		return wasi.Result{Ret: blocked}
	}
	wake := s.wake
	s.mu.Unlock()
	return wasi.Result{Wait: wake, Resume: func(m wasi.Memory) wasi.Result { return io.read(elem, false, args, m) }}
}

// cancel is [stream-cancel-read-0]F: a pending read finishes with what is
// there, or CANCELLED.
func (io streamIO) cancel(elem *wit.Type, args []uint64, m wasi.Memory) wasi.Result {
	s, err := io.stream(args[0], elem)
	if err != nil {
		return trapResult(err)
	}
	s.mu.Lock()
	if s.state != endCopying {
		s.mu.Unlock()
		return trapResult(Trapf("cancel of stream %d, which has no read pending", args[0]))
	}
	if s.ready() {
		ptr := s.ptr
		elems, payload := s.take(s.n)
		s.mu.Unlock()
		return deliver(elem, ptr, elems, m, func(wasi.Memory) wasi.Result { return wasi.Result{Ret: uint64(payload)} })
	}
	s.state = endIdle
	s.mu.Unlock()
	return wasi.Result{Ret: copyCancelled}
}

// dropReadable is [stream-drop-readable-0]F.
func (io streamIO) dropReadable(elem *wit.Type, args []uint64) wasi.Result {
	s, err := io.stream(args[0], elem)
	if err != nil {
		return trapResult(err)
	}
	s.mu.Lock()
	copying := s.state == endCopying
	s.mu.Unlock()
	if copying {
		return trapResult(Trapf("drop of stream %d while a read is pending", args[0]))
	}
	if err := io.h.Drop(uint32(args[0]), "stream"); err != nil {
		return trapResult(err)
	}
	return wasi.Result{}
}

func (io streamIO) set(n uint64) (*WaitSet, error) {
	o, err := io.h.Get(uint32(n), "waitable-set")
	if err != nil {
		return nil, err
	}
	return o.(*WaitSet), nil
}

// root serves the $root module's built-ins (wit.WaitableBuiltins order).
func (io streamIO) root(fn int, args []uint64, m wasi.Memory) wasi.Result {
	switch fn {
	case wit.WaitableSetNew:
		n, err := io.h.New("waitable-set", newWaitSet())
		if err != nil {
			return trapResult(err)
		}
		return wasi.Result{Ret: uint64(n)}
	case wit.WaitableSetDrop:
		w, err := io.set(args[0])
		if err != nil {
			return trapResult(err)
		}
		w.mu.Lock()
		busy := len(w.members) > 0 || w.waiting
		w.mu.Unlock()
		if busy {
			return trapResult(Trapf("drop of waitable set %d, which is not empty", args[0]))
		}
		if err := io.h.Drop(uint32(args[0]), "waitable-set"); err != nil {
			return trapResult(err)
		}
		return wasi.Result{}
	case wit.WaitableJoin:
		o, err := io.h.Get(uint32(args[0]), "stream")
		if err != nil {
			return trapResult(Trapf("waitable %d: %v", args[0], err))
		}
		s := o.(*Stream)
		var w *WaitSet
		if args[1] != 0 {
			if w, err = io.set(args[1]); err != nil {
				return trapResult(err)
			}
		}
		s.mu.Lock()
		if s.set != nil {
			s.set.remove(s)
		}
		s.set = w
		if w != nil {
			w.add(s)
		}
		s.mu.Unlock()
		return wasi.Result{}
	case wit.WaitableSetPoll, wit.WaitableSetWait:
		w, err := io.set(args[0])
		if err != nil {
			return trapResult(err)
		}
		ptr := uint32(args[1])
		if ptr%4 != 0 {
			return trapResult(Trapf("misaligned event at %#x", ptr))
		}
		if _, ok := m.Read(ptr, 8); !ok {
			return trapResult(Trapf("event at %#x out of bounds", ptr))
		}
		return io.event(w, ptr, fn == wit.WaitableSetWait, m)
	}
	return wasi.Result{Trap: "no such function"}
}

// event delivers one member's finished read, or none (poll), or waits.
func (io streamIO) event(w *WaitSet, ptr uint32, wait bool, m wasi.Memory) wasi.Result {
	w.mu.Lock()
	members := append([]*Stream(nil), w.members...)
	start := w.next
	wake := w.wake
	w.mu.Unlock()
	for i := range members {
		s := members[(start+i)%len(members)]
		s.mu.Lock()
		if s.state != endCopying || !s.ready() {
			s.mu.Unlock()
			continue
		}
		bufPtr := s.ptr
		elems, payload := s.take(s.n)
		s.mu.Unlock()
		w.mu.Lock()
		w.next = (start + i + 1) % max(len(members), 1)
		w.mu.Unlock()
		idx := io.indexOf(s)
		return deliver(s.Elem, bufPtr, elems, m, func(m wasi.Memory) wasi.Result {
			return storeEvent(m, ptr, eventStreamRead, idx, payload)
		})
	}
	if !wait {
		return storeEvent(m, ptr, eventNone, 0, 0)
	}
	w.mu.Lock()
	w.waiting = true
	w.mu.Unlock()
	return wasi.Result{Wait: wake, Resume: func(m wasi.Memory) wasi.Result {
		w.mu.Lock()
		w.waiting = false
		w.mu.Unlock()
		return io.event(w, ptr, true, m)
	}}
}

func (io streamIO) indexOf(s *Stream) uint32 {
	io.h.mu.Lock()
	defer io.h.mu.Unlock()
	for n, e := range io.h.tab {
		if e.obj == s {
			return n
		}
	}
	return 0
}

func storeEvent(m wasi.Memory, ptr uint32, code, idx, payload uint32) wasi.Result {
	var b [8]byte
	b[0], b[1], b[2], b[3] = byte(idx), byte(idx>>8), byte(idx>>16), byte(idx>>24)
	b[4], b[5], b[6], b[7] = byte(payload), byte(payload>>8), byte(payload>>16), byte(payload>>24)
	if !m.Write(ptr, b[:]) {
		return wasi.Result{Trap: "wasi:webgpu: event out of bounds"}
	}
	return wasi.Result{Ret: uint64(code)}
}

// rootFuncs is the $root module.
var rootFuncs = func() []wasi.HostFunc {
	var out []wasi.HostFunc
	for _, b := range wit.WaitableBuiltins {
		out = append(out, wasi.HostFunc{Name: b.Name, Params: b.Params, Result: b.Result})
	}
	return out
}()

// streamBuiltins are the built-ins for function F (core name name) whose
// result is a stream of elem.
func streamBuiltins(name string, elem *wit.Type) []export {
	var out []export
	for op, b := range wit.StreamBuiltins(name) {
		out = append(out, export{name: b.Name, builtin: &builtin{params: b.Params, result: b.Result, elem: elem, op: op}})
	}
	return out
}

// builtin is a Canonical ABI built-in a module serves next to its WIT
// functions.
type builtin struct {
	params, result string
	elem           *wit.Type
	op             int // wit.StreamRead etc.
}

func (b *builtin) call(io streamIO, args []uint64, m wasi.Memory) wasi.Result {
	switch b.op {
	case wit.StreamRead, wit.StreamReadAsync:
		return io.read(b.elem, b.op == wit.StreamReadAsync, args, m)
	case wit.StreamCancelRead, wit.StreamCancelReadAsync:
		return io.cancel(b.elem, args, m)
	}
	return io.dropReadable(b.elem, args)
}

// KeyFromCode maps a UI Events KeyboardEvent.code ("KeyA", "Digit0",
// "ArrowUp", "NumpadClearEntry") to the index of wasi-gfx:surface's key
// enum ("key-a", "digit0", "arrow-up", "numpad-clear-entry"); ok is false
// for a code it does not have (the event's key is then none).
func KeyFromCode(code string) (uint32, bool) {
	if code == "" || len(code) > 32 || code[0] < 'A' || code[0] > 'Z' {
		return 0, false // codes are PascalCase ASCII
	}
	var b strings.Builder
	for i, r := range code {
		if r >= 'A' && r <= 'Z' {
			if i > 0 {
				b.WriteByte('-')
			}
			b.WriteRune(r + 'a' - 'A')
			continue
		}
		b.WriteRune(r)
	}
	n, ok := keyIndex[b.String()]
	return n, ok
}

var keyIndex = func() map[string]uint32 {
	out := map[string]uint32{}
	for i, l := range SurfaceInterfaces["surface"].Types["key"].Labels {
		out[l] = uint32(i)
	}
	return out
}()
