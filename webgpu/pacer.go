package webgpu

import "time"

// pacer spaces a surface's frames FrameInterval apart on a fixed schedule.
//
// Each frame has a deadline; the next one is the previous deadline plus the
// interval, never "now plus the interval". A sleep that wakes late (a browser
// timer wakes 2 to 3 ms after its deadline) therefore shortens the next sleep
// instead of stretching the frame, and the mean interval is the interval. A
// frame that misses its deadline by a whole interval or more does not make the
// following ones rush to catch up: the schedule skips to the next slot ahead.
type pacer struct {
	interval time.Duration
	now      func() time.Time
	sleep    func(time.Duration)

	next time.Time // the deadline of the coming frame; zero before the first
}

// wait blocks until the coming frame's deadline (at once for the first frame
// or a late one) and schedules the frame after it.
func (p *pacer) wait() {
	now := p.now()
	if p.next.IsZero() {
		p.next = now
	}
	if d := p.next.Sub(now); d > 0 {
		p.sleep(d)
		now = p.now()
	}
	p.next = p.next.Add(p.interval)
	if !p.next.After(now) {
		// Missed by a whole interval: the next slot ahead of now.
		p.next = p.next.Add((now.Sub(p.next)/p.interval + 1) * p.interval)
	}
}
