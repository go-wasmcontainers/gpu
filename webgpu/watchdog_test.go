package webgpu

import (
	"testing"
	"time"
)

func TestWatchdogReport(t *testing.T) {
	at := time.Date(2026, 11, 24, 17, 20, 1, 123_000_000, time.UTC)
	got := WatchdogReport(41, 44, 2003400*time.Microsecond, 2*time.Second, at)
	want := "gpu watchdog: submit #41 overran its GPU time budget (waited 2.003s of 2s, 3 later submit(s) queued behind it, at 17:20:01.123Z); the process's devices are destroyed and the grant revoked"
	if got != want {
		t.Fatalf("WatchdogReport =\n%s\nwant\n%s", got, want)
	}
}
