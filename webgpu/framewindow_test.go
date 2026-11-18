package webgpu

import (
	"testing"
	"time"
)

// TestFrameWindowReady: ready-to-present is true while fewer than
// MaxFramesInFlight frames are unacknowledged, false at the limit, and true
// again after the page acknowledges one; sending (next) never waits while ready.
func TestFrameWindowReady(t *testing.T) {
	w := newFrameWindow()
	for i := 1; i <= MaxFramesInFlight; i++ {
		if !w.ready() {
			t.Fatalf("not ready with %d frames in flight", i-1)
		}
		if got := w.next(time.Second); got != uint64(i) {
			t.Fatalf("frame %d numbered %d", i, got)
		}
	}
	if w.ready() {
		t.Fatalf("ready with %d frames in flight", MaxFramesInFlight)
	}
	w.ack(1)
	if !w.ready() {
		t.Fatal("not ready after an acknowledgement")
	}
}

// TestFrameWindowNextWaitsForTheAck: next blocks while the window is full and
// returns, numbered, when the page acknowledges.
func TestFrameWindowNextWaitsForTheAck(t *testing.T) {
	w := newFrameWindow()
	for i := 0; i < MaxFramesInFlight; i++ {
		w.next(time.Second)
	}
	done := make(chan uint64, 1)
	go func() { done <- w.next(5 * time.Second) }()
	select {
	case n := <-done:
		t.Fatalf("next returned %d with a full window", n)
	case <-time.After(50 * time.Millisecond):
	}
	w.ack(1)
	select {
	case n := <-done:
		if n != MaxFramesInFlight+1 {
			t.Fatalf("numbered %d", n)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("next did not return after the acknowledgement")
	}
}

// TestFrameWindowGivesUp: a page that never acknowledges costs one wait per
// frame, never a hang (AckWait's contract).
func TestFrameWindowGivesUp(t *testing.T) {
	w := newFrameWindow()
	for i := 0; i < MaxFramesInFlight; i++ {
		w.next(time.Second)
	}
	t0 := time.Now()
	n := w.next(30 * time.Millisecond)
	if n != MaxFramesInFlight+1 || time.Since(t0) < 30*time.Millisecond {
		t.Fatalf("numbered %d after %v", n, time.Since(t0))
	}
}
