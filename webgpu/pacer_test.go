package webgpu

import (
	"testing"
	"time"
)

// fakeClock is a clock whose sleeps wake late by a fixed overshoot, as a
// browser timer does (2.7 ms measured in Chromium), and whose work advances it.
type fakeClock struct {
	t         time.Time
	overshoot time.Duration
}

func (c *fakeClock) now() time.Time        { return c.t }
func (c *fakeClock) sleep(d time.Duration) { c.t = c.t.Add(d + c.overshoot) }

// emits runs n frames through a pacer, each costing work(i) before its present,
// and returns the time of each present.
func emits(interval, overshoot time.Duration, n int, work func(i int) time.Duration) []time.Time {
	c := &fakeClock{t: time.Unix(1000, 0), overshoot: overshoot}
	p := &pacer{interval: interval, now: c.now, sleep: c.sleep}
	var out []time.Time
	for i := 0; i < n; i++ {
		c.t = c.t.Add(work(i))
		p.wait()
		out = append(out, c.t)
	}
	return out
}

func mean(ts []time.Time) time.Duration { return ts[len(ts)-1].Sub(ts[0]) / time.Duration(len(ts)-1) }

const sixty = time.Second / 60

// The mean interval is FrameInterval, within 0.5 ms, however late the timer
// wakes: the lateness is taken out of the next sleep.
func TestPacerMeanIntervalWithLateWakeups(t *testing.T) {
	for _, over := range []time.Duration{0, 1 * time.Millisecond, 2700 * time.Microsecond, 5 * time.Millisecond} {
		for _, work := range []time.Duration{0, 3 * time.Millisecond, 13 * time.Millisecond} {
			ts := emits(sixty, over, 120, func(int) time.Duration { return work })
			if m := mean(ts); m < sixty-500*time.Microsecond || m > sixty+500*time.Microsecond {
				t.Errorf("overshoot %v, work %v: mean interval %v, want %v +- 0.5 ms", over, work, m, sixty)
			}
		}
	}
}

// A frame whose work exceeds the interval is not delayed further, and one that
// misses its deadline does not make the next frames rush: they land on the
// schedule's next slot ahead.
func TestPacerSkipsAheadAfterAMissedDeadline(t *testing.T) {
	ts := emits(sixty, 2*time.Millisecond, 60, func(i int) time.Duration {
		if i == 20 {
			return 100 * time.Millisecond // six intervals
		}
		return 2 * time.Millisecond
	})
	for i := 1; i < len(ts); i++ {
		d := ts[i].Sub(ts[i-1])
		switch {
		case i == 20:
			if d < 100*time.Millisecond {
				t.Errorf("slow frame %d: interval %v", i, d)
			}
		case i > 21:
			// back on the schedule, with no burst of catch-up frames
			if d < sixty-500*time.Microsecond || d > sixty+500*time.Microsecond {
				t.Errorf("frame %d after the stall: interval %v, want %v", i, d, sixty)
			}
		}
	}
	// Work alone above the interval: the frame time is the work.
	ts = emits(sixty, 2*time.Millisecond, 30, func(int) time.Duration { return 25 * time.Millisecond })
	if m := mean(ts); m != 25*time.Millisecond {
		t.Errorf("work 25 ms: mean %v, want 25 ms", m)
	}
}

func TestPacerFirstFrameIsNotDelayed(t *testing.T) {
	c := &fakeClock{t: time.Unix(1000, 0)}
	p := &pacer{interval: sixty, now: c.now, sleep: c.sleep}
	start := c.t
	p.wait()
	if c.t != start {
		t.Errorf("first frame waited %v", c.t.Sub(start))
	}
}
