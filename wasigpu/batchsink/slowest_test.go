//go:build !js

package batchsink

import (
	"testing"
	"time"
)

// timedHost is an asyncHost whose calls take the time cost names, on a clock
// the test owns: the sink's clock reads it.
type timedHost struct {
	*asyncHost
	now  time.Time
	cost map[string]time.Duration
}

func (h *timedHost) Submit(ctx uint32, frame, read []byte) Errno {
	h.now = h.now.Add(h.cost["submit"])
	return h.asyncHost.fakeHost.Submit(ctx, frame, read)
}

func (h *timedHost) SubmitBegin(ctx uint32, frame, read []byte) Errno {
	h.now = h.now.Add(h.cost["submit_begin"])
	return h.asyncHost.SubmitBegin(ctx, frame, read)
}

func (h *timedHost) SubmitReady(ctx uint32) (bool, Errno) {
	h.now = h.now.Add(h.cost["submit_ready"])
	return h.asyncHost.SubmitReady(ctx)
}

func (h *timedHost) SubmitEnd(ctx uint32) Errno {
	h.now = h.now.Add(h.cost["submit_end"])
	return h.asyncHost.SubmitEnd(ctx)
}

func newTimedSink(t *testing.T, polls int, cost map[string]time.Duration) (*Sink, *timedHost) {
	t.Helper()
	h := &timedHost{asyncHost: &asyncHost{fakeHost: fakeHost{caps: testCaps()}, polls: polls}, now: time.Unix(1000, 0), cost: cost}
	s, err := NewSink(h)
	if err != nil {
		t.Fatal(err)
	}
	s.sleep = func(time.Duration) {}
	s.clock = func() time.Time { return h.now }
	return s, h
}

// Slowest names the longest single host call of the submits since the reset,
// not the sum of a frame's calls and not the idle function's time.
func TestSlowestIsTheLongestHostCall(t *testing.T) {
	s, h := newTimedSink(t, 2, map[string]time.Duration{
		"submit_begin": 3 * time.Millisecond,
		"submit_ready": 1 * time.Millisecond,
		"submit_end":   40 * time.Millisecond,
	})
	if n, d := s.Slowest(); n != "" || d != 0 {
		t.Fatalf("before any submit: %q %v", n, d)
	}
	// the idle function takes 200 ms, which is no host call
	s.SetIdle(func() error { h.now = h.now.Add(200 * time.Millisecond); return nil })
	if err := submitOne(t, s, make([]byte, 4)); err != nil {
		t.Fatal(err)
	}
	if n, d := s.Slowest(); n != "submit_end" || d != 40*time.Millisecond {
		t.Errorf("Slowest = %q %v, want submit_end 40ms", n, d)
	}
	s.ResetSlowest()
	if n, d := s.Slowest(); n != "" || d != 0 {
		t.Errorf("after the reset: %q %v", n, d)
	}
	h.cost["submit_begin"] = 90 * time.Millisecond
	if err := submitOne(t, s, make([]byte, 4)); err != nil {
		t.Fatal(err)
	}
	if n, d := s.Slowest(); n != "submit_begin" || d != 90*time.Millisecond {
		t.Errorf("Slowest = %q %v, want submit_begin 90ms", n, d)
	}
}

// A poll that is the longest call is named too, and the blocking submit is
// "submit".
func TestSlowestNamesPollAndBlockingSubmit(t *testing.T) {
	s, _ := newTimedSink(t, 1, map[string]time.Duration{"submit_ready": 7 * time.Millisecond, "submit_end": time.Millisecond})
	s.SetIdle(func() error { return nil })
	if err := submitOne(t, s, nil); err != nil {
		t.Fatal(err)
	}
	if n, d := s.Slowest(); n != "submit_ready" || d != 7*time.Millisecond {
		t.Errorf("Slowest = %q %v, want submit_ready 7ms", n, d)
	}
	b, _ := newTimedSink(t, 1, map[string]time.Duration{"submit": 12 * time.Millisecond})
	if err := submitOne(t, b, nil); err != nil {
		t.Fatal(err)
	}
	if n, d := b.Slowest(); n != "submit" || d != 12*time.Millisecond {
		t.Errorf("blocking: Slowest = %q %v, want submit 12ms", n, d)
	}
}

// With no clock set the sink reads the wall clock: a real host call that takes
// a while is seen.
func TestSlowestWithTheWallClock(t *testing.T) {
	h := &wallHost{asyncHost: &asyncHost{fakeHost: fakeHost{caps: testCaps()}}}
	s, err := NewSink(h)
	if err != nil {
		t.Fatal(err)
	}
	s.SetIdle(func() error { return nil })
	if err := submitOne(t, s, nil); err != nil {
		t.Fatal(err)
	}
	if n, d := s.Slowest(); n != "submit_end" || d < 5*time.Millisecond {
		t.Errorf("Slowest = %q %v, want submit_end of at least 5ms", n, d)
	}
}

// wallHost's submit_end really takes 10 ms.
type wallHost struct{ *asyncHost }

func (h *wallHost) SubmitEnd(ctx uint32) Errno {
	time.Sleep(10 * time.Millisecond)
	return h.asyncHost.SubmitEnd(ctx)
}
