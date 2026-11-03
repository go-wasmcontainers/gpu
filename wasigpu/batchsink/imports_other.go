//go:build !wasip1 && !js

package batchsink

// Open returns ErrUnavailable: only a wasip1 program has the host module.
// NewSink drives a Sink over any Host.
func Open() (*Sink, error) { return nil, ErrUnavailable }
