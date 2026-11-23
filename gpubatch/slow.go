package gpubatch

import (
	"strconv"
	"time"

	"github.com/go-virtio/gpu/wgpubatch"
)

// SlowSubmitMin is the time on the GPU's turn from which a submit is reported
// to SlowSubmit.
//
// It is a variable only so that a test can raise it above the jitter of a loaded
// runner; nothing else changes it.
var SlowSubmitMin = 10 * time.Millisecond

// SubmitInfo is what one submit was: how long its frame ran (the turn, not the
// wait for it), how long it waited for the turn, and what the frame held.
type SubmitInfo struct {
	Ran, Waited time.Duration
	Bytes       int // the frame's size
	Commands    int // its decoded messages
	Creates     int // of which object creations (shaders, pipelines, buffers)
	// Creations is what each creation of the frame was and cost, in order.
	Creations []Creation
}

// Creation is one object creation of a slow frame: what it made, how large its
// descriptor was (a shader's source, a texture's size) and how long the
// provider took.
type Creation struct {
	Kind  string
	Bytes int
	Ran   time.Duration
}

// kindNames are the wire kinds of a creation descriptor (go-gfx/webgpu/wire).
var kindNames = [...]string{1: "buffer", 2: "texture", 3: "texture-view", 4: "sampler", 5: "shader-module", 6: "bind-group-layout", 7: "pipeline-layout", 8: "bind-group", 9: "render-pipeline", 10: "compute-pipeline"}

func kindName(k uint32) string {
	if int(k) < len(kindNames) && kindNames[k] != "" {
		return kindNames[k]
	}
	return "kind-" + strconv.FormatUint(uint64(k), 10)
}

// SlowSubmit, when set, is called with every submit whose frame ran SlowSubmitMin
// or more, so a measurement can say how large the batch behind a slow call was.
// It is set once, before a module serves; the browser build sets it to keep the
// list in self.__batchSlow (slow_js.go).
var SlowSubmit func(SubmitInfo)

// Created, when set, is called with the number of objects each submit that
// creates any has created, fast or slow: a steady frame creates none, and a
// test can say so by reading a count that does not move. The browser build adds
// to self.__batchCreated (slow_js.go).
var Created func(n int)

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
