package webgpu

import (
	"math"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/go-wasmcontainers/gpu/wit"
)

// InputEvent is a pointer or key event the page (the dashboard pane that
// shows a surface) sends for that surface; the host turns it into a
// wasi-gfx:surface pointer-event or key-event on the guest's streams of
// that kind.
type InputEvent struct {
	// Kind is "pointer-down", "pointer-up", "pointer-move", "key-down" or
	// "key-up" (the surface methods on-<kind>).
	Kind string
	// X, Y: pointers, in surface pixels from the top-left corner (the page
	// scales from the pane).
	X, Y float64
	// Code is KeyboardEvent.code ("KeyA"); an unknown code is a key of
	// none. Text is the text the key produces (KeyboardEvent.key when it is
	// one character), "" for none.
	Code, Text             string
	Alt, Ctrl, Meta, Shift bool
}

// InputKinds are the kinds a surface serves.
var InputKinds = []string{"pointer-up", "pointer-down", "pointer-move", "key-up", "key-down"}

// MaxKeyText bounds a key event's text, in bytes.
const MaxKeyText = 64

// MaxCoordinate bounds a pointer coordinate: far outside any surface
// (1920x1080 at most) but finite.
const MaxCoordinate = 1 << 16

// inputElem is the element type of a surface's on-<kind> stream.
func inputElem(kind string) *wit.Type {
	for _, f := range SurfaceInterfaces["surface"].Resources["surface"].Funcs {
		if f.Name == "on-"+kind && f.Result != nil && f.Result.Kind == wit.Stream {
			return f.Result.Elem
		}
	}
	return nil
}

// value checks e and returns it as a value of its kind's element type;
// ok is false for an event to drop (unknown kind, non-finite or far-off
// coordinates, text too long or not UTF-8).
func (e InputEvent) value() (any, bool) {
	switch e.Kind {
	case "pointer-down", "pointer-up", "pointer-move":
		for _, c := range []float64{e.X, e.Y} {
			if math.IsNaN(c) || math.Abs(c) > MaxCoordinate {
				return nil, false
			}
		}
		return []any{e.X, e.Y}, true
	case "key-down", "key-up":
		if len(e.Text) > MaxKeyText || !utf8.ValidString(e.Text) {
			return nil, false
		}
		var key, text any
		if n, ok := KeyFromCode(e.Code); ok {
			key = wit.Some{Value: n}
		}
		if e.Text != "" {
			text = wit.Some{Value: e.Text}
		}
		return []any{key, text, e.Alt, e.Ctrl, e.Meta, e.Shift}, true
	}
	return nil, false
}

// inputHub is a surface's subscribers: the streams its guest opened, by
// kind.
type inputHub struct {
	mu     sync.Mutex
	subs   map[string][]*Stream
	closed bool
}

// MaxInputStreams bounds the streams one surface serves (all kinds).
const MaxInputStreams = 64

// openStream makes a stream of kind (the method on-<kind>) subscribed to
// the hub; it unsubscribes when the guest drops it.
func (h *inputHub) openStream(kind string) (*Stream, error) {
	elem := inputElem(kind)
	if elem == nil {
		return nil, Trapf("no input stream %q", kind)
	}
	s := NewStream(elem)
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.subs == nil {
		h.subs = map[string][]*Stream{}
	}
	n := 0
	for _, l := range h.subs {
		n += len(l)
	}
	if n >= MaxInputStreams {
		return nil, Trapf("too many input streams on one surface (%d)", MaxInputStreams)
	}
	if h.closed {
		s.CloseWriter()
	} else {
		h.subs[kind] = append(h.subs[kind], s)
	}
	s.SetOnClose(func() { h.remove(kind, s) })
	return s, nil
}

func (h *inputHub) remove(kind string, s *Stream) {
	h.mu.Lock()
	defer h.mu.Unlock()
	l := h.subs[kind]
	for i, x := range l {
		if x == s {
			h.subs[kind] = append(l[:i:i], l[i+1:]...)
			return
		}
	}
}

// publish delivers e to every stream of its kind; it reports whether the
// event was valid.
func (h *inputHub) publish(e InputEvent) bool {
	v, ok := e.value()
	if !ok {
		return false
	}
	h.mu.Lock()
	subs := append([]*Stream(nil), h.subs[e.Kind]...)
	h.mu.Unlock()
	for _, s := range subs {
		s.Push(v)
	}
	return true
}

// close ends every stream (the surface is gone): readers see DROPPED once
// they have read what is queued.
func (h *inputHub) close() {
	h.mu.Lock()
	all := h.subs
	h.subs, h.closed = nil, true
	h.mu.Unlock()
	for _, l := range all {
		for _, s := range l {
			s.CloseWriter()
		}
	}
}

// inputKind is the kind of a surface method name, "" if it is not one.
func inputKind(method string) string {
	k, ok := strings.CutPrefix(method, "[method]surface.on-")
	if !ok {
		return ""
	}
	for _, x := range InputKinds {
		if x == k {
			return k
		}
	}
	return ""
}
