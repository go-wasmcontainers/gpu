package wasigpu

import (
	"testing"
	"time"
)

// Slowest keeps the longest of the calls since ResetSlowest.
func TestSlowestKeepsTheLongestUntilReset(t *testing.T) {
	ResetSlowest()
	if n, d := Slowest(); n != "" || d != 0 {
		t.Fatalf("after reset: %q %v", n, d)
	}
	slowest.name, slowest.d = "[method]context.present", 3*time.Millisecond
	if n, d := Slowest(); n != "[method]context.present" || d != 3*time.Millisecond {
		t.Fatalf("got %q %v", n, d)
	}
	ResetSlowest()
	if n, _ := Slowest(); n != "" {
		t.Fatalf("not reset: %q", n)
	}
}
