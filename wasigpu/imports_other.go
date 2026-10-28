//go:build !wasip1

package wasigpu

// calls is empty outside wasip1: every call fails with ErrUnavailable.
var calls map[string]func(a []uint64) uint64
