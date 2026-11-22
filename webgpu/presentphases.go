package webgpu

import "time"

// SlowPresentMin is the duration of one context.present from which it is
// reported to SlowPresent.
const SlowPresentMin = 10 * time.Millisecond

// PresentInfo is where the time of one context.present went. Together the
// three phases are the call's whole execution in the provider.
type PresentInfo struct {
	// Paced is the pacer's sleep until the frame's deadline (Browser.FrameInterval).
	Paced time.Duration
	// Acked is the wait for the page to acknowledge an earlier frame, when
	// MaxFramesInFlight were unacknowledged.
	Acked time.Duration
	// Handed is the work after both waits: moving the frame out of the canvas
	// (transferToImageBitmap) and giving it to the page.
	Handed time.Duration
}

// Total is the call's whole execution.
func (i PresentInfo) Total() time.Duration { return i.Paced + i.Acked + i.Handed }

// SlowPresent, when set, is called with every present that took SlowPresentMin
// or more, so a measurement can say whether a slow present was the pacer's
// sleep, the wait for the page or the hand-over. It is set once, before a
// module serves; the browser build keeps the list in self.__presentSlow
// (presentphases_js.go).
var SlowPresent func(PresentInfo)

// timePresent runs the three phases of a present in order and reports how long
// each took on now's clock.
func timePresent(now func() time.Time, pace, ack, hand func()) PresentInfo {
	t0 := now()
	pace()
	t1 := now()
	ack()
	t2 := now()
	hand()
	t3 := now()
	i := PresentInfo{Paced: t1.Sub(t0), Acked: t2.Sub(t1), Handed: t3.Sub(t2)}
	if SlowPresent != nil && i.Total() >= SlowPresentMin {
		SlowPresent(i)
	}
	return i
}
