//go:build !js

package batchsink

import (
	"testing"
	"time"
)

// Waited splits a frame's time into the wait for the GPU (everything after the
// frame is sent: the polls, the idle function, the end) and the idle function's
// part of it, so the rest of the frame is the program's own work. Reset starts
// a new reading.
func TestWaitedIsTheTimeAfterTheFrameIsSent(t *testing.T) {
	s, h := newTimedSink(t, 2, map[string]time.Duration{
		"submit_begin": 3 * time.Millisecond, // not a wait: the frame is being sent
		"submit_ready": time.Millisecond,
		"submit_end":   40 * time.Millisecond,
	})
	s.SetIdle(func() error { h.now = h.now.Add(5 * time.Millisecond); return nil })
	if w, i := s.Waited(); w != 0 || i != 0 {
		t.Fatalf("before any submit: %v %v", w, i)
	}
	if err := submitOne(t, s, make([]byte, 4)); err != nil {
		t.Fatal(err)
	}
	// 3 polls of 1 ms, 2 idle slices of 5 ms, the end 40 ms.
	if w, i := s.Waited(); w != 53*time.Millisecond || i != 10*time.Millisecond {
		t.Errorf("Waited = %v %v, want 53ms 10ms", w, i)
	}
	s.ResetSlowest()
	if w, i := s.Waited(); w != 0 || i != 0 {
		t.Errorf("after the reset: %v %v", w, i)
	}
	b, _ := newTimedSink(t, 1, map[string]time.Duration{"submit": 12 * time.Millisecond})
	if err := submitOne(t, b, nil); err != nil {
		t.Fatal(err)
	}
	if w, i := b.Waited(); w != 12*time.Millisecond || i != 0 {
		t.Errorf("blocking submit: Waited = %v %v, want 12ms 0", w, i)
	}
}

// With no clock set the wait is read on the wall clock.
func TestWaitedWithTheWallClock(t *testing.T) {
	h := &wallHost{asyncHost: &asyncHost{fakeHost: fakeHost{caps: testCaps()}}}
	s, err := NewSink(h)
	if err != nil {
		t.Fatal(err)
	}
	s.SetIdle(func() error { return nil })
	if err := submitOne(t, s, nil); err != nil {
		t.Fatal(err)
	}
	if w, i := s.Waited(); w < 5*time.Millisecond || i > w {
		t.Errorf("Waited = %v %v, want a wait of at least 5ms holding the idle time", w, i)
	}
}
