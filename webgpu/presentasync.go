package webgpu

import (
	"encoding/binary"
	"sync"

	"github.com/go-wasmcontainers/engine/wasi"
)

// The presentation without the wait (engine#160): submit-present-begin,
// submit-present-end. A guest that must keep a sound output fed cannot sit in
// queue.submit and context.present, which the browser may hold for hundreds of
// milliseconds (the host thread is in a synchronous call), so begin returns at
// once, the frame runs on the host's own goroutine, and the host stores 1 in
// the guest word done when it is over. The guest reads that word itself: a poll
// that went through the host would queue behind the thread the browser holds.

const (
	callSubmitPresentBegin = "[method]context.submit-present-begin"
	callSubmitPresentEnd   = "[method]context.submit-present-end"
)

// presentRun is the begun frame of a process.
type presentRun struct {
	done chan struct{}
	err  error
}

// presentSlot holds the one begun frame of a process, shared by its modules.
type presentSlot struct {
	mu  sync.Mutex
	run *presentRun
}

// begin starts a frame: the submit and the present on a goroutine, and the
// word done set to 1 at its end. It checks nothing a session's own calls do not.
func (ps *presentSlot) begin(s Session, vals []any, m wasi.Memory) wasi.Result {
	// vals: self (the context), queue, command buffers, done.
	if len(vals) != 4 {
		return trapResult(Trapf("submit-present-begin: %d arguments", len(vals)))
	}
	var doneAt uint32
	switch x := vals[3].(type) {
	case uint32:
		doneAt = x
	case uint64:
		doneAt = uint32(x)
	default:
		return trapResult(Trapf("submit-present-begin: done is a %T, not a u32", vals[3]))
	}
	if doneAt%4 != 0 {
		return trapResult(Trapf("submit-present-begin: done %d is not aligned", doneAt))
	}
	// The word must be inside the guest's memory now, so a bad pointer traps at
	// the call that gave it and not later, on a goroutine nobody waits for.
	if _, ok := m.Read(doneAt, 4); !ok {
		return trapResult(Trapf("submit-present-begin: done %d is outside memory", doneAt))
	}
	ps.mu.Lock()
	if ps.run != nil {
		ps.mu.Unlock()
		return trapResult(Trapf("submit-present-begin: a frame is already begun"))
	}
	run := &presentRun{done: make(chan struct{})}
	ps.run = run
	ps.mu.Unlock()
	go func() {
		defer close(run.done)
		_, run.err = s.Call("[method]gpu-queue.submit", []any{vals[1], vals[2]})
		if run.err == nil {
			_, run.err = s.Call("[method]context.present", []any{vals[0]})
		}
		var one [4]byte
		binary.LittleEndian.PutUint32(one[:], 1)
		m.Write(doneAt, one[:]) // the guest's poll sees the frame over, an error or not
	}()
	return wasi.Result{}
}

// end waits for the begun frame (at once when it is over) and traps if it failed.
func (ps *presentSlot) end() wasi.Result {
	ps.mu.Lock()
	run := ps.run
	ps.run = nil
	ps.mu.Unlock()
	if run == nil {
		return trapResult(Trapf("submit-present-end: no frame was begun"))
	}
	return wasi.Result{Wait: run.done, Resume: func(wasi.Memory) wasi.Result {
		if run.err != nil {
			return trapResult(run.err)
		}
		return wasi.Result{}
	}}
}
