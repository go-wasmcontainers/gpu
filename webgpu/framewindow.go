package webgpu

import (
	"sync"
	"time"
)

// MaxFramesInFlight is how many unacknowledged frames a surface may have
// before present waits (at most AckWait) for AckFrame.
const MaxFramesInFlight = 2

// AckWait bounds one wait for acknowledgements: a page that stops
// acknowledging slows a guest down to one frame per AckWait, never stops
// it.
const AckWait = time.Second

// frameWindow is the frames a surface has handed to the page and the page has
// not yet acknowledged: seq is the last frame sent, acked the last the page
// drew or dropped.
type frameWindow struct {
	mu    sync.Mutex
	seq   uint64
	acked uint64
	ackCh chan struct{} // closed, and replaced, at each acknowledgement
}

func newFrameWindow() *frameWindow { return &frameWindow{ackCh: make(chan struct{})} }

// ack records that frame seq was drawn or dropped.
func (w *frameWindow) ack(seq uint64) {
	w.mu.Lock()
	if seq > w.acked {
		w.acked = seq
		close(w.ackCh)
		w.ackCh = make(chan struct{})
	}
	w.mu.Unlock()
}

// ready reports whether the next frame can be sent without waiting: fewer
// than MaxFramesInFlight are unacknowledged. It is what the guest's
// ready-to-present asks, so that a guest with work to do while it waits can
// wait in slices of its own.
func (w *frameWindow) ready() bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.seq-w.acked < MaxFramesInFlight
}

// next waits (each wait at most wait) until fewer than MaxFramesInFlight frames
// are unacknowledged, gives up on the oldest one when a wait runs out, and
// returns the number of the frame it now sends.
func (w *frameWindow) next(wait time.Duration) uint64 {
	w.mu.Lock()
	for w.seq-w.acked >= MaxFramesInFlight {
		ch := w.ackCh
		w.mu.Unlock()
		select {
		case <-ch:
		case <-time.After(wait):
		}
		w.mu.Lock()
		if w.seq-w.acked >= MaxFramesInFlight {
			w.acked = w.seq - MaxFramesInFlight + 1 // give up waiting on this one
		}
	}
	w.seq++
	seq := w.seq
	w.mu.Unlock()
	return seq
}
