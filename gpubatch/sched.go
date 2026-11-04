package gpubatch

import (
	"errors"
	"sync"
	"time"
)

// errAborted: the caller stopped waiting for its turn (its process ended).
var errAborted = errors.New("gpubatch: turn abandoned")

// sched serves the GPU's turns: one submit in flight at a time across every
// container of the module, and the waiting submits of different owners in
// round-robin order, so no owner can fill the queue ahead of another. An
// owner is whatever the caller passes (a module instance, one container's
// process). A turn's clock is the caller's: it starts when acquire returns,
// not when the submit was queued, so a container never pays for the turns of
// others. The zero value is ready to use.
//
// Why a global turn and not a per-device one: WebGPU's submit does not
// bound the work the GPU already holds, and destroy() does not cancel it
// (docs/gpu-scheduling.md). Work of two devices queues behind each other on
// the same GPU, so what the host must serialise is the GPU, not the device.
type sched struct {
	mu     sync.Mutex
	busy   bool
	ring   []any           // owners with an entry in queues, in arrival order
	queues map[any][]*turn // an owner's waiting submits, oldest first
	cursor int             // ring index to start the next search at
}

type turn struct {
	granted chan struct{}
	gone    bool // abandoned before it was granted
}

// acquire waits for the owner's turn. It returns a release function, to be
// called exactly once when the submit has finished; release(drain) with a
// non-nil drain keeps the GPU reserved until drain returns (a destroyed
// device's work is still running: the next turn must not start behind it).
// abort ends the wait with errAborted.
func (s *sched) acquire(owner any, abort <-chan struct{}) (release func(drain func()), err error) {
	s.mu.Lock()
	if !s.busy && len(s.ring) == 0 {
		s.busy = true
		s.mu.Unlock()
		return s.release, nil
	}
	t := &turn{granted: make(chan struct{})}
	if s.queues == nil {
		s.queues = map[any][]*turn{}
	}
	if _, ok := s.queues[owner]; !ok {
		s.ring = append(s.ring, owner)
	}
	s.queues[owner] = append(s.queues[owner], t)
	s.mu.Unlock()
	select {
	case <-t.granted:
		return s.release, nil
	case <-abort:
		s.mu.Lock()
		select {
		case <-t.granted: // granted at the same moment: hand the turn on
			s.mu.Unlock()
			s.release(nil)
		default:
			t.gone = true
			s.mu.Unlock()
		}
		return nil, errAborted
	}
}

// release ends the current turn.
func (s *sched) release(drain func()) {
	if drain == nil {
		s.next()
		return
	}
	go func() {
		drain()
		s.next()
	}()
}

// next gives the GPU to the next owner in round-robin order, or frees it.
func (s *sched) next() {
	s.mu.Lock()
	defer s.mu.Unlock()
	for len(s.ring) > 0 {
		if s.cursor >= len(s.ring) {
			s.cursor = 0
		}
		o := s.ring[s.cursor]
		q := s.queues[o]
		// Skip abandoned turns.
		for len(q) > 0 && q[0].gone {
			q = q[1:]
		}
		if len(q) == 0 {
			delete(s.queues, o)
			s.ring = append(s.ring[:s.cursor], s.ring[s.cursor+1:]...)
			continue
		}
		t := q[0]
		if len(q) == 1 {
			delete(s.queues, o)
			s.ring = append(s.ring[:s.cursor], s.ring[s.cursor+1:]...) // the next owner slides into cursor
		} else {
			s.queues[o] = q[1:]
			s.cursor++
		}
		close(t.granted)
		return // busy stays true: the turn passed on
	}
	s.busy = false
}

// Stats are what a module measured, for choosing its budget and for tests.
// Durations are the host's clock from the start of a turn to the end of the
// submit (decode and queueing excluded): host submit to the GPU's
// completion, read-back included.
type Stats struct {
	Submits   int64 `json:"submits"`   // submits that ran
	Sanctions int64 `json:"sanctions"` // submits over their budget
	// MaxSubmit is the longest submit that stayed within its budget, over
	// all frames; MaxStream those without a creation (the steady state, held
	// to Limits.Budget) and MaxCreate those with one (shader and pipeline
	// compilation, held to Limits.GPUTime).
	MaxSubmit time.Duration `json:"maxSubmit"`
	MaxStream time.Duration `json:"maxStream"`
	MaxCreate time.Duration `json:"maxCreate"`
	// MaxWait is the longest a submit waited for its turn.
	MaxWait time.Duration `json:"maxWait"`
	// MaxInvocations is the largest stream, in invocations as the compute
	// quota counts them.
	MaxInvocations uint64 `json:"maxInvocations"`
	// MaxDispatchInvocations is the largest single dispatch, counted alike.
	MaxDispatchInvocations uint64 `json:"maxDispatchInvocations"`
	// Truncations is how many submits hit the loop budget (layer 4); each is
	// also a sanction.
	Truncations int64 `json:"truncations"`
	// LoopModules is how many guest WGSL modules were parsed for layer 4,
	// LoopRewritten how many of them were rewritten, LoopFallbacks how many
	// the browser refused after the rewrite (kept as written, bounded by the
	// time budget only), and LoopSpent the time spent parsing, rewriting and
	// printing them (the cost of layer 4 at module creation).
	LoopModules   int64         `json:"loopModules"`
	LoopRewritten int64         `json:"loopRewritten"`
	LoopFallbacks int64         `json:"loopFallbacks"`
	LoopSpent     time.Duration `json:"loopSpent"`
	// Drains is how many times a destroyed device's work was waited out.
	Drains int64 `json:"drains"`
}
