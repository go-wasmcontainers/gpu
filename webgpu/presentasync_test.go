package webgpu

import (
	"encoding/binary"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/go-wasmcontainers/engine/rootfs"
	"github.com/go-wasmcontainers/engine/wasi"
)

// slowSession is a session whose queue.submit waits for a gate (the browser's
// synchronous submit) and whose present records that it ran.
type slowSession struct {
	gate      chan struct{}
	submitErr error
	mu        sync.Mutex
	calls     []string
}

type slowBackend struct{ s *slowSession }

func (b slowBackend) Open(Process, Limits, *Handles) Session { return b.s }

func (s *slowSession) Call(name string, args []any) (any, error) {
	s.mu.Lock()
	s.calls = append(s.calls, name)
	s.mu.Unlock()
	if name == "[method]gpu-queue.submit" {
		<-s.gate
		return nil, s.submitErr
	}
	return nil, nil
}
func (s *slowSession) Async(string) bool { return false }
func (s *slowSession) Close()            {}
func (s *slowSession) seen() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.calls...)
}

// lockedMem is guest memory the host goroutine and the test may touch at once.
type lockedMem struct {
	mu sync.Mutex
	m  mem
}

func (l *lockedMem) Read(off, n uint32) ([]byte, bool) {
	l.mu.Lock()
	defer l.mu.Unlock()
	b, ok := l.m.Read(off, n)
	return append([]byte(nil), b...), ok
}

func (l *lockedMem) Write(off uint32, b []byte) bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.m.Write(off, b)
}

// word is the little-endian u32 at off.
func (l *lockedMem) word(off uint32) uint32 {
	b, _ := l.Read(off, 4)
	return binary.LittleEndian.Uint32(b)
}

// presentHarness opens the surface-webgpu module on a slow session.
func presentHarness(t *testing.T) (*slowSession, *Module, wasi.Instance, *lockedMem) {
	t.Helper()
	s := &slowSession{gate: make(chan struct{})}
	m := NewModules(slowBackend{s})[2]
	in := m.Open(wasi.New(wasi.Config{FS: rootfs.New(), Caps: []string{"gpu"}}))
	t.Cleanup(in.Close)
	mm := &lockedMem{m: make(mem, 256)}
	binary.LittleEndian.PutUint32(mm.m[64:], 7) // the command buffer handle
	return s, m, in, mm
}

// beginArgs: self, queue, the list (pointer, length), done.
func beginArgs(done uint64) []uint64 { return []uint64{1, 2, 64, 1, done} }

// TestSubmitPresentBeginReturnsWhileTheFrameRuns: the browser holds queue.submit
// for as long as it likes; begin must hand the guest its thread back at once,
// the word done must turn 1 only when the frame is over, and end must collect it.
func TestSubmitPresentBeginReturnsWhileTheFrameRuns(t *testing.T) {
	s, m, in, mm := presentHarness(t)
	t0 := time.Now()
	r := in.Call(index(t, m, callSubmitPresentBegin), beginArgs(128), mm)
	if r.Trap != "" || r.Wait != nil {
		t.Fatalf("begin: %+v (it must return at once, with no wait)", r)
	}
	if d := time.Since(t0); d > 50*time.Millisecond {
		t.Errorf("begin took %v while the submit was held", d)
	}
	time.Sleep(20 * time.Millisecond)
	if got := mm.word(128); got != 0 {
		t.Fatalf("done = %d while the submit is held, want 0", got)
	}
	close(s.gate) // the browser lets the frame go
	deadline := time.Now().Add(2 * time.Second)
	for mm.word(128) != 1 {
		if time.Now().After(deadline) {
			t.Fatal("done never became 1")
		}
		time.Sleep(time.Millisecond)
	}
	if got := strings.Join(s.seen(), " "); got != "[method]gpu-queue.submit [method]context.present" {
		t.Errorf("calls %q, want the submit then the present", got)
	}
	r = in.Call(index(t, m, callSubmitPresentEnd), []uint64{1}, mm)
	if r.Trap != "" {
		t.Fatalf("end: %+v", r)
	}
	if r.Wait != nil {
		<-r.Wait
		if r = r.Resume(mm); r.Trap != "" {
			t.Fatalf("end resumed: %+v", r)
		}
	}
}

// TestSubmitPresentEndWaitsForTheFrame: end called while the frame runs waits
// for it (a Wait the engine serves off its thread), and a failed frame traps there.
func TestSubmitPresentEndWaitsForTheFrame(t *testing.T) {
	s, m, in, mm := presentHarness(t)
	s.submitErr = Trapf("submit: device lost")
	if r := in.Call(index(t, m, callSubmitPresentBegin), beginArgs(128), mm); r.Trap != "" {
		t.Fatalf("begin: %+v", r)
	}
	r := in.Call(index(t, m, callSubmitPresentEnd), []uint64{1}, mm)
	if r.Wait == nil {
		t.Fatalf("end with the frame running did not wait: %+v", r)
	}
	select {
	case <-r.Wait:
		t.Fatal("end's wait is over while the submit is held")
	case <-time.After(20 * time.Millisecond):
	}
	close(s.gate)
	<-r.Wait
	if r = r.Resume(mm); !strings.Contains(r.Trap, "device lost") {
		t.Errorf("a failed frame: %+v, want a trap naming the failure", r)
	}
	if got := mm.word(128); got != 1 {
		t.Errorf("done = %d after a failed frame, want 1 (the guest must not poll for ever)", got)
	}
}

// TestSubmitPresentRefusals: argument errors trap at begin, one frame at a time,
// and an end with no begin traps.
func TestSubmitPresentRefusals(t *testing.T) {
	s, m, in, mm := presentHarness(t)
	defer close(s.gate)
	begin := index(t, m, callSubmitPresentBegin)
	for _, c := range []struct {
		name string
		done uint64
	}{{"unaligned", 130}, {"outside memory", 1 << 20}, {"past the end by a word", 254}} {
		if r := in.Call(begin, beginArgs(c.done), mm); r.Trap == "" {
			t.Errorf("%s: begin accepted", c.name)
		}
	}
	if r := in.Call(index(t, m, callSubmitPresentEnd), []uint64{1}, mm); r.Trap == "" {
		t.Error("end with no begin accepted")
	}
	if r := in.Call(begin, beginArgs(128), mm); r.Trap != "" {
		t.Fatalf("begin: %+v", r)
	}
	if r := in.Call(begin, beginArgs(132), mm); r.Trap == "" {
		t.Error("a second begin before the end accepted")
	}
}
