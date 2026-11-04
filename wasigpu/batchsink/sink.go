//go:build !js

// Package batchsink is the guest side of the host module wasi_webgpu_batch
// (package gpubatch): a webgpu.ReadSink over its four imports, so that a
// wasip1 Go program writes against the go-gfx/webgpu API unchanged.
//
//	s, err := batchsink.Open()          // caps + context_create
//	dev := s.Device("my-container")      // a webgpu remote device on the sink
//	... CreateBuffer, NewRecorder, Dispatch, ReadBuffer, Submit ...
//
// The sink batches. Object creations and destructions are not sent when
// they happen: they are appended to the frame being built (go-virtio/gpu's
// wgpubatch.Encoder) and travel with the next Submit, which appends its
// stream and crosses into the host once, with the read window the stream's
// ReadBuffer commands fill. One Submit is one crossing; a frame that would
// exceed the host's max_frame is sent early, so a program that uploads
// weights in chunks pays one crossing per chunk.
//
// Errors the host reports lose the context (the host releases what it
// held); every later call returns the same error.
//
// Outside GOOS=wasip1 the package builds and Open returns ErrUnavailable;
// tests drive a Sink over any host with NewSink.
package batchsink

import (
	"errors"
	"fmt"

	"github.com/go-gfx/webgpu"
	"github.com/go-gfx/webgpu/wire"
	"github.com/go-virtio/gpu/wgpubatch"
)

// ErrUnavailable: there is no wasi_webgpu_batch module here (not a wasip1
// program).
var ErrUnavailable = errors.New("batchsink: wasi_webgpu_batch is not available outside wasip1")

// MaxRead is the host's read window per submit (gpubatch.DefaultLimits).
const MaxRead = 1 << 20

// Errno is the host module's result code (gpubatch.Errno).
type Errno uint32

// Errno values of wasi_webgpu_batch.
const (
	Success         Errno = 0
	InvalidArgument Errno = 1
	Denied          Errno = 2 // no gpu capability: run --gpu
	Unsupported     Errno = 3 // no GPU here, or only a fallback adapter
	TooLarge        Errno = 4
	QuotaExceeded   Errno = 5
	Refused         Errno = 6 // the frame or stream was refused; the context is lost
	ContextLost     Errno = 7
	Timeout         Errno = 8 // a submit overran its time budget; the device is destroyed and the grant revoked
	Fault           Errno = 9
	Failed          Errno = 10
	InvalidShader   Errno = 11 // WGSL that does not compile, or an object WebGPU rejects; the context is lost, Error.Msg says why
	Revoked         Errno = 12 // an earlier submit overran its budget: no GPU for the rest of the container's life
	Reserved        Errno = 13 // the guest used the bind group the host reserves for its loop budget (group 3); the context is lost
	Truncated       Errno = 14 // a shader hit the host's loop budget and was cut: its results are partial; the context is lost and the grant revoked
)

var errnoNames = [...]string{"success", "invalid argument", "denied (the container has no GPU: run --gpu)",
	"unsupported (no GPU here, or only a fallback adapter)", "too large", "quota exceeded", "refused",
	"context lost", "timeout (a submit overran its GPU time budget; the GPU grant is revoked)", "fault", "failed",
	"invalid shader or object (the GPU refused it)", "revoked (an earlier submit overran its GPU time budget)",
	"reserved (the guest used bind group 3, which the host keeps for its loop budget)",
	"truncated (a shader hit the host's loop budget and was cut: its results are partial; the GPU grant is revoked)"}

// Error is a host error.
type Error struct {
	Op    string
	Errno Errno
	// Msg is the host's message for the failure (last_error): for
	// InvalidShader, the WGSL compiler's text with line and column.
	Msg string
}

func (e *Error) Error() string {
	n := fmt.Sprintf("errno %d", e.Errno)
	if int(e.Errno) < len(errnoNames) {
		n = errnoNames[e.Errno]
	}
	if e.Msg != "" {
		return fmt.Sprintf("wasi_webgpu_batch: %s: %s: %s", e.Op, n, e.Msg)
	}
	return fmt.Sprintf("wasi_webgpu_batch: %s: %s", e.Op, n)
}

// Is reports whether target is an *Error of the same Errno, so that
// errors.Is(err, &Error{Errno: Refused}) tests a class.
func (e *Error) Is(target error) bool {
	t, ok := target.(*Error)
	return ok && t.Errno == e.Errno && (t.Op == "" || t.Op == e.Op)
}

// Host is the four imports of wasi_webgpu_batch, over Go slices.
type Host interface {
	Caps(buf []byte) Errno
	ContextCreate() (ctx uint32, e Errno)
	// Submit sends a frame; the host fills read before returning.
	Submit(ctx uint32, frame, read []byte) Errno
	ContextDestroy(ctx uint32) Errno
}

// ErrorReporter is implemented by a Host that can say why its last call
// failed (the wasi_webgpu_batch import last_error). A Sink asks it after a
// failed submit and puts the message in the Error.
type ErrorReporter interface {
	LastError(ctx uint32) string
}

// fail builds the error of a failed host call, with the host's message when
// it has one.
func (s *Sink) fail(op string, e Errno) *Error {
	err := &Error{Op: op, Errno: e}
	if er, ok := s.h.(ErrorReporter); ok && e != Success {
		err.Msg = er.LastError(s.ctx)
	}
	return err
}

// Sink is a webgpu.ReadSink on one host context.
type Sink struct {
	h    Host
	ctx  uint32
	caps wgpubatch.Caps
	enc  *wgpubatch.Encoder
	size uint64 // bytes of the frame being built
	n    int    // its messages
	err  error  // the error that lost the context
	subs int
	done bool
}

// NewSink queries the host's capset and creates a context on it.
func NewSink(h Host) (*Sink, error) {
	buf := make([]byte, 256)
	if e := h.Caps(buf); e != Success {
		return nil, &Error{Op: "caps", Errno: e}
	}
	caps, err := wgpubatch.DecodeCaps(buf)
	if err != nil {
		return nil, err
	}
	if caps.StreamVersion != wire.StreamVersion {
		return nil, fmt.Errorf("batchsink: the host replays stream version %d, this program records version %d", caps.StreamVersion, wire.StreamVersion)
	}
	ctx, e := h.ContextCreate()
	if e != Success {
		return nil, &Error{Op: "context_create", Errno: e}
	}
	s := &Sink{h: h, ctx: ctx, caps: caps, enc: wgpubatch.NewEncoder(caps.MaxFrame)}
	s.reset()
	return s, nil
}

// Caps is what the host granted: frame and object limits, the GPU memory
// quota, the optional features (wgpubatch.FeatureShaderF16, FeatureSubgroups)
// and the device limits.
func (s *Sink) Caps() wgpubatch.Caps { return s.caps }

// Err is the error that lost the context (or closed the sink), or nil.
func (s *Sink) Err() error { return s.err }

// Submits is the number of host crossings made by Submit so far.
func (s *Sink) Submits() int { return s.subs }

// Device returns a go-gfx/webgpu device that records into the sink, with
// the checks and the GPU memory quota of the host's grant applied at the
// call.
func (s *Sink) Device(label string) *webgpu.Device {
	return webgpu.NewRemoteDevice(s, label, &webgpu.Quota{
		MemoryBytes: int64(min(s.caps.MaxMemory, 1<<62)), Pipelines: 256, ShaderBytes: 1 << 20, ReadBytes: MaxRead,
	})
}

func (s *Sink) reset() {
	s.enc.Reset()
	s.size, s.n = wgpubatch.FrameHeaderSize, 0
}

func pad4(n int) uint64 { return (uint64(n) + 3) &^ 3 }

// room makes the frame hold a message of size bytes (header included),
// sending what is pending first if it would not fit.
func (s *Sink) room(size uint64) error {
	if s.err != nil {
		return s.err
	}
	if wgpubatch.FrameHeaderSize+size > uint64(s.caps.MaxFrame) {
		return fmt.Errorf("batchsink: a message of %d bytes cannot fit the host's %d-byte frame", size, s.caps.MaxFrame)
	}
	if s.size+size > uint64(s.caps.MaxFrame) {
		return s.flush(nil)
	}
	return nil
}

func (s *Sink) added(size uint64) { s.size += size; s.n++ }

// flush sends the pending frame (if any) with an optional read window.
func (s *Sink) flush(window []byte) error {
	if s.err != nil {
		return s.err
	}
	if s.n == 0 && len(window) == 0 {
		return nil
	}
	frame, err := s.enc.Frame()
	if err != nil {
		return s.lose(err)
	}
	if e := s.h.Submit(s.ctx, frame, window); e != Success {
		return s.lose(s.fail("submit", e))
	}
	s.subs++
	s.reset()
	return nil
}

func (s *Sink) lose(err error) error {
	s.err = err
	return err
}

// Create implements webgpu.RemoteSink: the creation travels with the next
// Submit (or Flush).
func (s *Sink) Create(id uint32, kind wire.Kind, desc []byte) error {
	sz := wgpubatch.MessageHeaderSize + 12 + pad4(len(desc))
	if err := s.room(sz); err != nil {
		return err
	}
	s.enc.Create(id, uint32(kind), desc)
	s.added(sz)
	return nil
}

// Destroy implements webgpu.RemoteSink: it travels with the next Submit.
func (s *Sink) Destroy(id uint32) error {
	sz := uint64(wgpubatch.MessageHeaderSize + 4)
	if err := s.room(sz); err != nil {
		return err
	}
	s.enc.Destroy(id)
	s.added(sz)
	return nil
}

// Submit implements webgpu.RemoteSink: one crossing, which returns once the
// GPU has executed the stream.
func (s *Sink) Submit(words []uint32, data []byte) error { return s.SubmitRead(words, data, nil) }

// SubmitRead implements webgpu.ReadSink: Submit, then the stream's
// ReadBuffer commands have filled window (at most MaxRead bytes).
func (s *Sink) SubmitRead(words []uint32, data []byte, window []byte) error {
	if len(window) > MaxRead {
		return fmt.Errorf("batchsink: a %d-byte read window above the host's %d", len(window), MaxRead)
	}
	sz := wgpubatch.MessageHeaderSize + 8 + 4*uint64(len(words)) + pad4(len(data))
	if err := s.room(sz); err != nil {
		return err
	}
	s.enc.Stream(words, data)
	s.added(sz)
	return s.flush(window)
}

// Flush sends pending creations and destructions without a stream.
func (s *Sink) Flush() error { return s.flush(nil) }

// Raw sends frame as it is, with a read window, bypassing the encoder:
// for tools and tests that must send a frame the encoder would not build.
// Pending messages are sent first.
func (s *Sink) Raw(frame, window []byte) error {
	if err := s.flush(nil); err != nil {
		return err
	}
	if e := s.h.Submit(s.ctx, frame, window); e != Success {
		return s.lose(s.fail("submit", e))
	}
	s.subs++
	return nil
}

// Close sends what is pending and destroys the host context. The Sink is
// unusable afterwards.
func (s *Sink) Close() error {
	if s.done {
		return nil
	}
	s.done = true
	err := s.flush(nil)
	if e := s.h.ContextDestroy(s.ctx); e != Success && err == nil {
		err = &Error{Op: "context_destroy", Errno: e}
	}
	cerr := err
	if s.err == nil {
		s.err = errors.New("batchsink: closed")
	}
	return cerr
}
