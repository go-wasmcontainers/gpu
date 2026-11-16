package wasigpu

import (
	"encoding/binary"
	"fmt"

	"github.com/go-wasmcontainers/gpu/cabi"
	"github.com/go-wasmcontainers/gpu/wit"
)

// Input on a surface (wasi-gfx:surface on-pointer-* and on-key-*): each
// kind is a stream the host writes as the page reports events on the
// surface's pane. A stream is read without blocking, once per frame: Poll
// returns what arrived since the last call. Underneath, an asynchronous
// stream.read stays pending on a buffer and a waitable set reports when it
// completed (the Canonical ABI built-ins of package wit).

// Input kinds (the surface methods on-<kind>).
const (
	PointerDown = "pointer-down"
	PointerUp   = "pointer-up"
	PointerMove = "pointer-move"
	KeyDown     = "key-down"
	KeyUp       = "key-up"
)

// PointerEvent is a pointer position in surface pixels from the top-left.
type PointerEvent struct{ X, Y float64 }

// KeyEvent is a key: Key is its wasi-gfx:surface key label ("key-a",
// "arrow-up", "space"; "" when the host could not identify it), Text the
// text it produces ("" for none).
type KeyEvent struct {
	Key, Text              string
	Alt, Ctrl, Meta, Shift bool
}

// pollBatch is how many events one read takes at most.
const pollBatch = 32

// events is one input stream of a surface.
type events struct {
	fn      string // the method's core import name
	elem    *wit.Type
	end     uint32
	set     uint32
	buf     uint32
	keep    []byte // the pending read's buffer, kept reachable
	ev      uint32 // where waitable-set.poll stores (end, payload)
	evKeep  []byte
	pending bool
	done    bool
}

func (s *Surface) events(kind string) (*events, error) {
	fn := "[method]surface.on-" + kind
	f := funcs[surfaceModule+" "+fn]
	if f == nil || !wit.ReturnsStream(f) {
		return nil, fmt.Errorf("wasigpu: no input kind %q", kind)
	}
	end, err := handle(callIn(surfaceModule, fn, s.sf))
	if err != nil {
		return nil, err
	}
	set := uint32(builtin(wit.RootModule, wit.WaitableBuiltins[wit.WaitableSetNew].Name))
	builtin(wit.RootModule, wit.WaitableBuiltins[wit.WaitableJoin].Name, uint64(end), uint64(set))
	e := &events{fn: fn, elem: f.Result.Elem, end: end, set: set}
	e.buf, e.keep = cabi.Buffer(wit.Align(e.elem), pollBatch*wit.Size(e.elem))
	e.ev, e.evKeep = cabi.Buffer(4, 8)
	return e, nil
}

// builtin calls a Canonical ABI built-in import over core values.
func builtin(module, name string, args ...uint64) uint64 {
	c, ok := calls[module+" "+name]
	if !ok {
		return 0
	}
	return c(args)
}

func (e *events) streamBuiltin(op int, args ...uint64) uint64 {
	return builtin(surfaceModule, wit.StreamBuiltins(e.fn)[op].Name, args...)
}

// poll returns the events that arrived, without blocking.
func (e *events) poll() ([]any, error) {
	var out []any
	for !e.done {
		var payload uint32
		if e.pending {
			code := builtin(wit.RootModule, wit.WaitableBuiltins[wit.WaitableSetPoll].Name, uint64(e.set), uint64(e.ev))
			if code == wit.EventNone {
				break
			}
			b, _ := cabi.Memory{}.Read(e.ev, 8)
			payload = binary.LittleEndian.Uint32(b[4:])
			e.pending = false
		} else {
			payload = uint32(e.streamBuiltin(wit.StreamReadAsync, uint64(e.end), uint64(e.buf), pollBatch))
			if payload == wit.Blocked {
				e.pending = true
				break
			}
		}
		n := payload >> 4
		for i := range n {
			v, err := wit.Load(cabi.Memory{}, e.buf+i*wit.Size(e.elem), e.elem)
			if err != nil {
				cabi.Release()
				return out, err
			}
			out = append(out, v)
		}
		cabi.Release() // key text the host allocated has been copied out
		if payload&0xf == wit.CopyDropped {
			e.done = true
		}
	}
	return out, nil
}

// Closed reports whether the stream ended (the surface was dropped).
func (e *events) closed() bool { return e.done }

func (e *events) drop() {
	if e.pending {
		e.streamBuiltin(wit.StreamCancelRead, uint64(e.end))
		e.pending = false
	}
	builtin(wit.RootModule, wit.WaitableBuiltins[wit.WaitableJoin].Name, uint64(e.end), 0)
	e.streamBuiltin(wit.StreamDropReadable, uint64(e.end))
	builtin(wit.RootModule, wit.WaitableBuiltins[wit.WaitableSetDrop].Name, uint64(e.set))
}

// Extension kinds (not in upstream wasi-gfx:surface): relative pointer
// motion while the pointer is locked to the pane, and the mouse buttons.
const (
	PointerMotion = "pointer-motion"
	ButtonDown    = "button-down"
	ButtonUp      = "button-up"
)

// MotionEvent is pointer movement in surface pixels since the previous
// event, positive to the right and down.
type MotionEvent struct{ DX, DY float64 }

// ButtonEvent is a mouse button press or release at a surface pixel: Button
// is MouseEvent.button (0 left, 1 middle, 2 right, 3 back, 4 forward).
type ButtonEvent struct {
	Button int
	X, Y   float64
}

// MotionEvents is a surface's stream of relative pointer motion.
type MotionEvents struct{ e *events }

// Motion opens the surface's relative-motion stream. The host sends motion
// while the pointer is locked to the pane: ask for the lock with
// RequestPointerLock; the page grants it on the next click on the pane and
// releases it on Escape.
func (s *Surface) Motion() (*MotionEvents, error) {
	e, err := s.events(PointerMotion)
	if err != nil {
		return nil, err
	}
	return &MotionEvents{e}, nil
}

// Poll returns the events since the last call, oldest first; it never
// blocks.
func (m *MotionEvents) Poll() ([]MotionEvent, error) {
	vs, err := m.e.poll()
	out := make([]MotionEvent, 0, len(vs))
	for _, v := range vs {
		f, ok := v.([]any)
		if !ok || len(f) != 2 {
			return out, fmt.Errorf("wasigpu: motion event %T", v)
		}
		dx, _ := f[0].(float64)
		dy, _ := f[1].(float64)
		out = append(out, MotionEvent{dx, dy})
	}
	return out, err
}

// Closed reports whether the stream ended (its surface is gone).
func (m *MotionEvents) Closed() bool { return m.e.closed() }

// Drop releases the stream.
func (m *MotionEvents) Drop() { m.e.drop() }

// ButtonEvents is a surface's stream of one mouse-button kind.
type ButtonEvents struct{ e *events }

// Buttons opens the surface's mouse-button stream of kind (ButtonDown or
// ButtonUp).
func (s *Surface) Buttons(kind string) (*ButtonEvents, error) {
	if kind != ButtonDown && kind != ButtonUp {
		return nil, fmt.Errorf("wasigpu: %q is not a button kind", kind)
	}
	e, err := s.events(kind)
	if err != nil {
		return nil, err
	}
	return &ButtonEvents{e}, nil
}

// Poll returns the events since the last call, oldest first; it never
// blocks.
func (b *ButtonEvents) Poll() ([]ButtonEvent, error) {
	vs, err := b.e.poll()
	out := make([]ButtonEvent, 0, len(vs))
	for _, v := range vs {
		ev, derr := decodeButton(v)
		if derr != nil {
			return out, derr
		}
		out = append(out, ev)
	}
	return out, err
}

// decodeButton reads a button-event record as the Canonical ABI loads it:
// the u8 button comes back as a uint64.
func decodeButton(v any) (ButtonEvent, error) {
	f, ok := v.([]any)
	if !ok || len(f) != 3 {
		return ButtonEvent{}, fmt.Errorf("wasigpu: button event %T", v)
	}
	var n int
	switch b := f[0].(type) {
	case uint64:
		n = int(b)
	case uint32:
		n = int(b)
	default:
		return ButtonEvent{}, fmt.Errorf("wasigpu: button %T", f[0])
	}
	x, _ := f[1].(float64)
	y, _ := f[2].(float64)
	return ButtonEvent{n, x, y}, nil
}

// Closed reports whether the stream ended (its surface is gone).
func (b *ButtonEvents) Closed() bool { return b.e.closed() }

// Drop releases the stream.
func (b *ButtonEvents) Drop() { b.e.drop() }

// RequestPointerLock tells the host whether the guest wants the pointer
// locked to the surface's pane (true) or not (false). The page cannot lock
// the pointer without a user gesture, so it locks on the next click on the
// pane; Escape releases it (and the guest may ask again). While locked the
// pane sends relative motion (Motion) instead of absolute moves.
func (s *Surface) RequestPointerLock(wanted bool) error {
	_, err := callIn(surfaceModule, "[method]surface.request-pointer-lock", s.sf, wanted)
	return err
}

// PointerEvents is a surface's stream of one pointer kind.
type PointerEvents struct{ e *events }

// Pointer opens the surface's pointer stream of kind (PointerDown,
// PointerUp or PointerMove).
func (s *Surface) Pointer(kind string) (*PointerEvents, error) {
	if kind != PointerDown && kind != PointerUp && kind != PointerMove {
		return nil, fmt.Errorf("wasigpu: %q is not a pointer kind", kind)
	}
	e, err := s.events(kind)
	if err != nil {
		return nil, err
	}
	return &PointerEvents{e}, nil
}

// Poll returns the events since the last call, oldest first; it never
// blocks.
func (p *PointerEvents) Poll() ([]PointerEvent, error) {
	vs, err := p.e.poll()
	out := make([]PointerEvent, 0, len(vs))
	for _, v := range vs {
		f, ok := v.([]any)
		if !ok || len(f) != 2 {
			return out, fmt.Errorf("wasigpu: pointer event %T", v)
		}
		x, _ := f[0].(float64)
		y, _ := f[1].(float64)
		out = append(out, PointerEvent{x, y})
	}
	return out, err
}

// Closed reports whether the stream ended (its surface is gone).
func (p *PointerEvents) Closed() bool { return p.e.closed() }

// Drop releases the stream.
func (p *PointerEvents) Drop() { p.e.drop() }

// KeyEvents is a surface's stream of one key kind.
type KeyEvents struct{ e *events }

// Keys opens the surface's key stream of kind (KeyDown or KeyUp).
func (s *Surface) Keys(kind string) (*KeyEvents, error) {
	if kind != KeyDown && kind != KeyUp {
		return nil, fmt.Errorf("wasigpu: %q is not a key kind", kind)
	}
	e, err := s.events(kind)
	if err != nil {
		return nil, err
	}
	return &KeyEvents{e}, nil
}

// Poll returns the events since the last call, oldest first; it never
// blocks.
func (k *KeyEvents) Poll() ([]KeyEvent, error) {
	vs, err := k.e.poll()
	labels := surfaceIface.Types["key"].Labels
	out := make([]KeyEvent, 0, len(vs))
	for _, v := range vs {
		f, ok := v.([]any)
		if !ok || len(f) != 6 {
			return out, fmt.Errorf("wasigpu: key event %T", v)
		}
		var ev KeyEvent
		if s, ok := f[0].(wit.Some); ok {
			if n, ok := s.Value.(uint32); ok && int(n) < len(labels) {
				ev.Key = labels[n]
			}
		}
		if s, ok := f[1].(wit.Some); ok {
			ev.Text, _ = s.Value.(string)
		}
		ev.Alt, _ = f[2].(bool)
		ev.Ctrl, _ = f[3].(bool)
		ev.Meta, _ = f[4].(bool)
		ev.Shift, _ = f[5].(bool)
		out = append(out, ev)
	}
	return out, err
}

// Closed reports whether the stream ended (its surface is gone).
func (k *KeyEvents) Closed() bool { return k.e.closed() }

// Drop releases the stream.
func (k *KeyEvents) Drop() { k.e.drop() }
