package webgpu

import (
	"fmt"
	"time"
)

// WatchdogReport is the line the host writes (to stderr, which the browser
// shows as console.error and the guest's output carries) when a submit
// overran GPUTime and the process's devices are destroyed. It names what the
// next reader needs to tell a slow GPU from a starved completion signal
// (Firefox resolves onSubmittedWorkDone on a 100 ms poll tick): the
// submission that timed out, how many later ones were queued behind it, how
// long the wait ran against the limit, and when, on the host's clock.
func WatchdogReport(seq, latest uint64, waited, limit time.Duration, at time.Time) string {
	return fmt.Sprintf("gpu watchdog: submit #%d overran its GPU time budget (waited %s of %s, %d later submit(s) queued behind it, at %s); the process's devices are destroyed and the grant revoked",
		seq, waited.Round(time.Millisecond), limit, latest-seq, at.UTC().Format("15:04:05.000Z"))
}
