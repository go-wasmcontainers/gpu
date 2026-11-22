package gpubatch

import (
	"time"

	"github.com/go-virtio/gpu/wgpubatch"
)

// SlowSubmitMin is the time on the GPU's turn from which a submit is reported
// to SlowSubmit.
const SlowSubmitMin = 10 * time.Millisecond

// SubmitInfo is what one submit was: how long its frame ran (the turn, not the
// wait for it), how long it waited for the turn, and what the frame held.
type SubmitInfo struct {
	Ran, Waited time.Duration
	Bytes       int // the frame's size
	Commands    int // its decoded messages
	Creates     int // of which object creations (shaders, pipelines, buffers)
}

// SlowSubmit, when set, is called with every submit whose frame ran SlowSubmitMin
// or more, so a measurement can say how large the batch behind a slow call was.
// It is set once, before a module serves; the browser build sets it to keep the
// list in self.__batchSlow (slow_js.go).
var SlowSubmit func(SubmitInfo)

// infoOf describes a decoded frame.
func infoOf(frame []byte, nmsgs, creates int, waited, ran time.Duration) SubmitInfo {
	return SubmitInfo{Ran: ran, Waited: waited, Bytes: len(frame), Commands: nmsgs, Creates: creates}
}

func countCreates(msgs []wgpubatch.Message) (n int) {
	for _, m := range msgs {
		if m.Type == wgpubatch.MsgCreate {
			n++
		}
	}
	return n
}
