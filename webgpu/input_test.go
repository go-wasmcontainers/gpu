package webgpu

import (
	"math"
	"strings"
	"testing"

	"github.com/go-wasmcontainers/gpu/wit"
)

func TestInputValidation(t *testing.T) {
	ka, _ := KeyFromCode("KeyA")
	for _, c := range []struct {
		e    InputEvent
		want any // nil: dropped
	}{
		{InputEvent{Kind: "pointer-move", X: 3, Y: 4}, []any{3.0, 4.0}},
		{InputEvent{Kind: "pointer-down", X: -5, Y: 1e4}, []any{-5.0, 1e4}},
		{InputEvent{Kind: "pointer-up", X: math.NaN()}, nil},
		{InputEvent{Kind: "pointer-up", X: math.Inf(1)}, nil},
		{InputEvent{Kind: "pointer-up", Y: MaxCoordinate + 1}, nil},
		{InputEvent{Kind: "key-down", Code: "KeyA", Text: "a", Shift: true}, []any{wit.Some{Value: ka}, wit.Some{Value: "a"}, false, false, false, true}},
		{InputEvent{Kind: "key-up", Code: "Unidentified"}, []any{nil, nil, false, false, false, false}},
		{InputEvent{Kind: "key-down", Text: strings.Repeat("x", MaxKeyText+1)}, nil},
		{InputEvent{Kind: "key-down", Text: "\xff"}, nil},
		{InputEvent{Kind: "scroll"}, nil},
	} {
		v, ok := c.e.value()
		if c.want == nil {
			if ok {
				t.Errorf("%+v accepted as %v", c.e, v)
			}
			continue
		}
		if !ok || !equalValues(v, c.want) {
			t.Errorf("%+v: %v %v, want %v", c.e, v, ok, c.want)
		}
	}
}

func equalValues(a, b any) bool {
	x, ok1 := a.([]any)
	y, ok2 := b.([]any)
	if !ok1 || !ok2 || len(x) != len(y) {
		return false
	}
	for i := range x {
		if x[i] != y[i] {
			return false
		}
	}
	return true
}

func TestInputHub(t *testing.T) {
	var h inputHub
	down, err := h.openStream("pointer-down")
	if err != nil {
		t.Fatal(err)
	}
	move, _ := h.openStream("pointer-move")
	key, _ := h.openStream("key-down")
	if _, err := h.openStream("scroll"); err == nil {
		t.Error("an unknown kind opened")
	}
	if !h.publish(InputEvent{Kind: "pointer-down", X: 1, Y: 2}) || h.publish(InputEvent{Kind: "pointer-down", X: math.NaN()}) {
		t.Error("publish validity")
	}
	h.publish(InputEvent{Kind: "key-down", Code: "Enter"})
	if len(down.queue) != 1 || len(move.queue) != 0 || len(key.queue) != 1 {
		t.Errorf("queues %d %d %d: events went to the wrong kinds", len(down.queue), len(move.queue), len(key.queue))
	}
	// A dropped stream unsubscribes.
	down.Destroy()
	h.publish(InputEvent{Kind: "pointer-down"})
	if len(h.subs["pointer-down"]) != 0 {
		t.Error("a dropped stream stayed subscribed")
	}
	// Closing the hub (the surface is dropped) ends every stream, and one
	// opened afterwards starts ended.
	h.close()
	if !move.closed || !key.closed {
		t.Error("streams not ended with their surface")
	}
	late, _ := h.openStream("pointer-up")
	if !late.closed {
		t.Error("a stream opened on a closed surface is live")
	}
}

func TestInputStreamBound(t *testing.T) {
	var h inputHub
	for i := range MaxInputStreams {
		if _, err := h.openStream(InputKinds[i%len(InputKinds)]); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := h.openStream("pointer-down"); err == nil {
		t.Errorf("more than %d streams on one surface", MaxInputStreams)
	}
}

func TestInputKind(t *testing.T) {
	for m, want := range map[string]string{
		"[method]surface.on-pointer-down": "pointer-down",
		"[method]surface.on-key-up":       "key-up",
		"[method]surface.on-resize":       "",
		"[method]surface.width":           "",
		"[method]context.on-key-up":       "",
	} {
		if got := inputKind(m); got != want {
			t.Errorf("%s: %q, want %q", m, got, want)
		}
	}
}
