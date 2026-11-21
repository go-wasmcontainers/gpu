//go:build !js

package batchsink

import (
	"errors"
	"testing"
	"time"
)

// asyncHost is a fakeHost that also runs frames without blocking: a begun frame
// is over after polls polls of submit_ready, and fills its window at submit_end.
type asyncHost struct {
	fakeHost
	polls   int // polls that answer "not yet" for each frame
	left    int
	begun   int
	ended   int
	window  []byte
	order   []string
	readyE  Errno
	endE    Errno
	pending bool
}

func (h *asyncHost) SubmitBegin(ctx uint32, frame, read []byte) Errno {
	h.order = append(h.order, "begin")
	h.begun++
	h.left = h.polls
	h.window = read
	h.pending = true
	return Success
}

func (h *asyncHost) SubmitReady(ctx uint32) (bool, Errno) {
	h.order = append(h.order, "ready")
	if h.readyE != Success {
		return false, h.readyE
	}
	if h.left > 0 {
		h.left--
		return false, Success
	}
	return true, Success
}

func (h *asyncHost) SubmitEnd(ctx uint32) Errno {
	h.order = append(h.order, "end")
	h.ended++
	h.pending = false
	for i := range h.window {
		h.window[i] = byte(i)
	}
	return h.endE
}

func newAsyncSink(t *testing.T, polls int) (*Sink, *asyncHost) {
	t.Helper()
	h := &asyncHost{fakeHost: fakeHost{caps: testCaps()}, polls: polls}
	s, err := NewSink(h)
	if err != nil {
		t.Fatal(err)
	}
	s.sleep = func(time.Duration) {}
	return s, h
}

func submitOne(t *testing.T, s *Sink, window []byte) error {
	t.Helper()
	return s.SubmitRead([]uint32{1, 2}, nil, window)
}

// With an idle function and an AsyncHost, a submit begins the frame and runs the
// idle function between polls until the frame is over: once per poll that found
// it running, never after, and the read window is filled at the end.
func TestIdleRunsWhileTheFrameRuns(t *testing.T) {
	s, h := newAsyncSink(t, 3)
	idle := 0
	s.SetIdle(func() error { idle++; return nil })
	w := make([]byte, 8)
	if err := submitOne(t, s, w); err != nil {
		t.Fatal(err)
	}
	if idle != 3 || s.IdleSlices() != 3 {
		t.Errorf("idle ran %d times (%d slices), want 3: one per poll that found the frame running", idle, s.IdleSlices())
	}
	if h.begun != 1 || h.ended != 1 || h.fakeHost.frames != nil {
		t.Errorf("begun %d, ended %d, blocking submits %d: want the async path only", h.begun, h.ended, len(h.fakeHost.frames))
	}
	for i, b := range w {
		if int(b) != i {
			t.Fatalf("window byte %d = %d: not filled by submit_end", i, b)
		}
	}
	if s.Submits() != 1 {
		t.Errorf("%d submits counted, want 1", s.Submits())
	}
}

// A frame that is over at the first poll runs the idle function never.
func TestIdleNotRunWhenTheFrameIsAlreadyOver(t *testing.T) {
	s, _ := newAsyncSink(t, 0)
	s.SetIdle(func() error { t.Error("idle ran for a frame that was over"); return nil })
	if err := submitOne(t, s, nil); err != nil {
		t.Fatal(err)
	}
}

// Without an idle function the submit is the blocking one.
func TestNoIdleIsTheBlockingSubmit(t *testing.T) {
	s, h := newAsyncSink(t, 3)
	if err := submitOne(t, s, make([]byte, 4)); err != nil {
		t.Fatal(err)
	}
	if h.begun != 0 || len(h.fakeHost.frames) != 1 {
		t.Errorf("begun %d, blocking submits %d: want the blocking path", h.begun, len(h.fakeHost.frames))
	}
}

// A host without the async imports (a plain Host) is the blocking submit even
// with an idle function.
func TestIdleOnAPlainHost(t *testing.T) {
	s, h := newSink(t)
	s.SetIdle(func() error { t.Error("idle ran on a host that cannot begin a frame"); return nil })
	if err := submitOne(t, s, make([]byte, 4)); err != nil {
		t.Fatal(err)
	}
	if len(h.frames) != 1 {
		t.Errorf("%d blocking submits, want 1", len(h.frames))
	}
}

// An idle function that fails does not leave the frame begun: the frame is ended,
// the error comes back, and the context is still usable.
func TestIdleErrorEndsTheFrame(t *testing.T) {
	s, h := newAsyncSink(t, 4)
	boom := errors.New("boom")
	calls := 0
	s.SetIdle(func() error { calls++; return boom })
	if err := submitOne(t, s, make([]byte, 4)); !errors.Is(err, boom) {
		t.Fatalf("error %v, want the idle function's", err)
	}
	if calls != 1 || h.pending || h.ended != 1 {
		t.Errorf("idle ran %d times, pending %v, ended %d: want once, nothing left begun", calls, h.pending, h.ended)
	}
	if s.Err() != nil {
		t.Errorf("the idle error lost the context: %v", s.Err())
	}
	s.SetIdle(nil)
	if err := submitOne(t, s, nil); err != nil {
		t.Errorf("the next submit: %v", err)
	}
}

// A host error from submit_ready or submit_end loses the context, like a
// blocking submit's.
func TestAsyncHostErrorsLoseTheContext(t *testing.T) {
	for _, tc := range []struct {
		name string
		set  func(*asyncHost)
	}{
		{"ready", func(h *asyncHost) { h.readyE = ContextLost }},
		{"end", func(h *asyncHost) { h.endE = Timeout }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, h := newAsyncSink(t, 1)
			tc.set(h)
			s.SetIdle(func() error { return nil })
			err := submitOne(t, s, nil)
			var e *Error
			if !errors.As(err, &e) || e.Op != "submit" {
				t.Fatalf("error %v, want a submit *Error", err)
			}
			if h.pending {
				t.Error("a frame was left begun")
			}
			if s.Err() == nil || submitOne(t, s, nil) == nil {
				t.Error("the context was not lost")
			}
		})
	}
}
