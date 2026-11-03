//go:build !js

package batchsink

import (
	"errors"
	"strings"
	"testing"

	"github.com/go-gfx/webgpu"
	"github.com/go-gfx/webgpu/wire"
	"github.com/go-virtio/gpu/wgpubatch"
)

// fakeHost records the frames it is sent and answers a script.
type fakeHost struct {
	caps      wgpubatch.Caps
	capsErr   Errno
	createErr Errno
	submitErr Errno
	frames    [][]byte
	windows   []int
	destroyed []uint32
	destroyE  Errno
}

func (h *fakeHost) Caps(buf []byte) Errno {
	if h.capsErr != Success {
		return h.capsErr
	}
	copy(buf, h.caps.Encode())
	return Success
}
func (h *fakeHost) ContextCreate() (uint32, Errno) { return 7, h.createErr }
func (h *fakeHost) Submit(ctx uint32, frame, read []byte) Errno {
	h.frames = append(h.frames, append([]byte(nil), frame...))
	h.windows = append(h.windows, len(read))
	for i := range read {
		read[i] = byte(i) // the "GPU"'s bytes
	}
	return h.submitErr
}
func (h *fakeHost) ContextDestroy(ctx uint32) Errno {
	h.destroyed = append(h.destroyed, ctx)
	return h.destroyE
}

func testCaps() wgpubatch.Caps {
	return wgpubatch.Caps{Version: 2, StreamVersion: wire.StreamVersion, MaxFrame: 4096, MaxObjects: 64, MaxRings: 1, MaxMemory: 1 << 20}
}

func newSink(t *testing.T) (*Sink, *fakeHost) {
	t.Helper()
	h := &fakeHost{caps: testCaps()}
	s, err := NewSink(h)
	if err != nil {
		t.Fatal(err)
	}
	return s, h
}

func messages(t *testing.T, f []byte) []wgpubatch.Message {
	t.Helper()
	m, err := wgpubatch.Decode(f, wgpubatch.Limits{MaxFrame: 4096, MaxObjects: 64})
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func TestNewSinkRefusals(t *testing.T) {
	for name, tc := range map[string]struct {
		h    *fakeHost
		want string
	}{
		"denied":      {&fakeHost{caps: testCaps(), capsErr: Denied}, "caps: denied"},
		"unsupported": {&fakeHost{caps: testCaps(), capsErr: Unsupported}, "caps: unsupported"},
		"stream version": {&fakeHost{caps: func() wgpubatch.Caps { c := testCaps(); c.StreamVersion = 3; return c }()},
			"stream version 3"},
		"no context": {&fakeHost{caps: testCaps(), createErr: QuotaExceeded}, "context_create: quota exceeded"},
		"bad blob":   {&fakeHost{caps: wgpubatch.Caps{}}, "malformed"},
	} {
		_, err := NewSink(tc.h)
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: %v, want %q", name, err, tc.want)
		}
	}
	if _, err := Open(); !errors.Is(err, ErrUnavailable) {
		t.Errorf("Open outside wasip1: %v", err)
	}
	if !errors.Is(&Error{Op: "submit", Errno: Refused}, &Error{Errno: Refused}) || errors.Is(&Error{Op: "submit", Errno: Refused}, &Error{Errno: Failed}) ||
		errors.Is(&Error{Op: "submit", Errno: Refused}, errors.New("x")) {
		t.Error("Error.Is")
	}
	if !strings.Contains((&Error{Op: "x", Errno: 99}).Error(), "errno 99") {
		t.Error("unknown errno text")
	}
}

// Creations and destructions travel with the next submit: one crossing.
func TestBatching(t *testing.T) {
	s, h := newSink(t)
	d := s.Device("t")
	src, err := d.CreateBuffer(webgpu.BufferDescriptor{Size: 64, Usage: webgpu.BufferUsageCopySrc | webgpu.BufferUsageCopyDst})
	if err != nil {
		t.Fatal(err)
	}
	dst, _ := d.CreateBuffer(webgpu.BufferDescriptor{Size: 64, Usage: webgpu.BufferUsageMapRead | webgpu.BufferUsageCopyDst})
	if len(h.frames) != 0 || s.Submits() != 0 {
		t.Fatalf("%d crossings before any submit", len(h.frames))
	}
	r := d.NewRecorder()
	r.CopyBufferToBuffer(src, 0, dst, 0, 64)
	rb := r.ReadBuffer(dst, 8, 16)
	if _, err := r.Submit(); err != nil {
		t.Fatal(err)
	}
	if s.Submits() != 1 || len(h.frames) != 1 || h.windows[0] != 16 {
		t.Fatalf("%d crossings, windows %v", len(h.frames), h.windows)
	}
	m := messages(t, h.frames[0])
	if len(m) != 3 || m[0].Type != wgpubatch.MsgCreate || m[1].Type != wgpubatch.MsgCreate || m[2].Type != wgpubatch.MsgStream {
		t.Fatalf("messages %+v", m)
	}
	b, err := rb.Bytes()
	if err != nil || len(b) != 16 || b[0] != 0 || b[15] != 15 {
		t.Fatalf("read-back %v %v", b, err)
	}
	// A destruction rides with the next submit; Flush sends it alone.
	dst.Destroy()
	if len(h.frames) != 1 {
		t.Fatal("a destroy crossed by itself")
	}
	if err := s.Flush(); err != nil || len(h.frames) != 2 || messages(t, h.frames[1])[0].Type != wgpubatch.MsgDestroy {
		t.Fatalf("flush: %v, %d frames", err, len(h.frames))
	}
	if err := s.Flush(); err != nil || len(h.frames) != 2 {
		t.Fatal("an empty flush crossed")
	}
}

// A frame that would pass max_frame is sent early, in order.
func TestFlushWhenFull(t *testing.T) {
	s, h := newSink(t)
	d := s.Device("t")
	for i := 0; i < 400; i++ { // 400 creations: far over the 4096-byte frame
		if _, err := d.CreateBuffer(webgpu.BufferDescriptor{Size: 16, Usage: webgpu.BufferUsageCopyDst}); err != nil {
			t.Fatal(err)
		}
	}
	if len(h.frames) < 1 {
		t.Fatal("no early flush")
	}
	for _, f := range h.frames {
		if len(f) > 4096 {
			t.Fatalf("a %d-byte frame above max_frame", len(f))
		}
	}
	// One message larger than the frame is refused, not truncated.
	_, err := d.CreateShaderModule(webgpu.ShaderModuleDescriptor{Code: strings.Repeat("x", 5000)})
	if err == nil || !strings.Contains(err.Error(), "cannot fit") {
		t.Errorf("oversized message: %v", err)
	}
	if s.Err() != nil {
		t.Error("a message that does not fit lost the context")
	}
}

func TestSubmitReadLimits(t *testing.T) {
	s, _ := newSink(t)
	if err := s.SubmitRead([]uint32{uint32(wire.Submit)}, nil, make([]byte, MaxRead+1)); err == nil {
		t.Error("a window above the host's")
	}
	if err := s.SubmitRead([]uint32{uint32(wire.Submit)}, make([]byte, 8192), nil); err == nil {
		t.Error("a stream above max_frame")
	}
}

func TestHostErrorsLoseTheContext(t *testing.T) {
	s, h := newSink(t)
	h.submitErr = Refused
	err := s.Submit([]uint32{uint32(wire.Submit)}, nil)
	var e *Error
	if !errors.As(err, &e) || e.Errno != Refused || e.Op != "submit" {
		t.Fatalf("%v", err)
	}
	n := len(h.frames)
	h.submitErr = Success
	// Every later call returns the same error and reaches no host.
	if err2 := s.Submit([]uint32{uint32(wire.Submit)}, nil); err2 != err {
		t.Errorf("later submit: %v", err2)
	}
	if err2 := s.Create(1, wire.KindBuffer, nil); err2 != err {
		t.Errorf("later create: %v", err2)
	}
	if err2 := s.Destroy(1); err2 != err {
		t.Errorf("later destroy: %v", err2)
	}
	if err2 := s.Raw(make([]byte, 16), nil); err2 != err {
		t.Errorf("later raw: %v", err2)
	}
	if len(h.frames) != n {
		t.Error("a lost context reached the host")
	}
	if err := s.Close(); err == nil {
		t.Error("Close of a lost sink reported nothing")
	}
	if len(h.destroyed) != 1 || h.destroyed[0] != 7 {
		t.Errorf("context_destroy calls %v", h.destroyed)
	}
	if s.Close() != nil || len(h.destroyed) != 1 {
		t.Error("second Close")
	}
}

func TestRawAndClose(t *testing.T) {
	s, h := newSink(t)
	if err := s.Create(3, wire.KindBuffer, []byte{1, 2, 3}); err != nil {
		t.Fatal(err)
	}
	// Raw sends what is pending first, then the frame as given.
	if err := s.Raw([]byte("not a frame"), make([]byte, 4)); err != nil {
		t.Fatal(err)
	}
	if len(h.frames) != 2 || string(h.frames[1]) != "not a frame" || h.windows[1] != 4 || s.Submits() != 2 {
		t.Fatalf("%d frames", len(h.frames))
	}
	if err := s.Destroy(3); err != nil {
		t.Fatal(err)
	}
	h.destroyE = Failed
	if err := s.Close(); err == nil { // pending destroy flushed, context_destroy failed
		t.Error("context_destroy error lost")
	}
	if len(h.frames) != 3 {
		t.Errorf("Close did not flush: %d frames", len(h.frames))
	}
	if err := s.Submit([]uint32{uint32(wire.Submit)}, nil); err == nil {
		t.Error("submit after Close")
	}
	// A raw submit the host refuses loses the sink.
	s2, h2 := newSink(t)
	h2.submitErr = TooLarge
	if err := s2.Raw(make([]byte, 16), nil); err == nil {
		t.Error("raw refusal")
	}
	// An id the encoder cannot carry loses the sink at the frame.
	s3, _ := newSink(t)
	if err := s3.Create(0, wire.KindBuffer, nil); err != nil {
		t.Fatal(err) // queued: the encoder reports at Frame
	}
	if err := s3.Flush(); err == nil {
		t.Error("an id-0 creation was sent")
	}
}

func TestDeviceQuotaFromCaps(t *testing.T) {
	s, _ := newSink(t)
	d := s.Device("q")
	if _, err := d.CreateBuffer(webgpu.BufferDescriptor{Size: 2 << 20, Usage: webgpu.BufferUsageStorage}); err == nil {
		t.Error("a 2 MiB buffer under a 1 MiB grant")
	}
	if s.Caps().MaxMemory != 1<<20 {
		t.Error("Caps")
	}
}

type reportingHost struct {
	fakeHost
	msg string
}

func (h *reportingHost) Submit(ctx uint32, frame, read []byte) Errno { return InvalidShader }
func (h *reportingHost) LastError(ctx uint32) string                 { return h.msg }

// A host that can say why gives the error its message.
func TestInvalidShaderMessage(t *testing.T) {
	h := &reportingHost{msg: "Error while parsing WGSL: :1:42 error: 'meta' is a reserved keyword"}
	h.fakeHost = *(func() *fakeHost { _, f := newSink(t); return f })()
	s, err := NewSink(h)
	if err != nil {
		t.Fatal(err)
	}
	d := s.Device("t")
	m, err := d.CreateShaderModule(webgpu.ShaderModuleDescriptor{Code: "x"})
	_ = m
	if err != nil {
		t.Fatal(err)
	}
	err = s.Flush()
	var e *Error
	if !errors.As(err, &e) || e.Errno != InvalidShader || !strings.Contains(e.Msg, "reserved keyword") || !strings.Contains(e.Error(), "reserved keyword") {
		t.Fatalf("%v", err)
	}
}
