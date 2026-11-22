package webgpu

import (
	"testing"
	"time"
)

// A fake clock that each phase advances by its own cost.
type phaseClock struct{ t time.Time }

func (c *phaseClock) now() time.Time        { return c.t }
func (c *phaseClock) spend(d time.Duration) { c.t = c.t.Add(d) }

func TestTimePresentSplitsThePhases(t *testing.T) {
	c := &phaseClock{t: time.Unix(1000, 0)}
	var got []PresentInfo
	SlowPresent = func(i PresentInfo) { got = append(got, i) }
	defer func() { SlowPresent = nil }()

	i := timePresent(c.now,
		func() { c.spend(14 * time.Millisecond) },
		func() { c.spend(3 * time.Millisecond) },
		func() { c.spend(8 * time.Millisecond) })

	if i.Paced != 14*time.Millisecond || i.Acked != 3*time.Millisecond || i.Handed != 8*time.Millisecond {
		t.Fatalf("phases = %+v, want paced 14 ms, acked 3 ms, handed 8 ms", i)
	}
	if i.Total() != 25*time.Millisecond {
		t.Fatalf("total = %v, want 25 ms", i.Total())
	}
	if len(got) != 1 || got[0] != i {
		t.Fatalf("SlowPresent got %+v, want exactly the 25 ms present", got)
	}
}

func TestTimePresentIgnoresAFastPresent(t *testing.T) {
	c := &phaseClock{t: time.Unix(1000, 0)}
	called := false
	SlowPresent = func(PresentInfo) { called = true }
	defer func() { SlowPresent = nil }()

	timePresent(c.now,
		func() { c.spend(4 * time.Millisecond) },
		func() {},
		func() { c.spend(3 * time.Millisecond) })

	if called {
		t.Fatal("a 7 ms present was reported as slow")
	}
}
