// Package gpubatch is the host side of the batched GPU command stream for
// WASI containers: the import module "wasi_webgpu_batch", served by the
// engine's host-module mechanism (wasi.HostModule), natively and in the
// browser. A guest records go-gfx/webgpu work with a remote device (package
// wasigpu/batchsink does it for Go guests), builds a frame with
// go-virtio/gpu's wgpubatch, and crosses into the host once per submit.
//
// The host does not interpret the work: it copies the frame out of guest
// memory, decodes it (wgpubatch.Decode), and hands creations, destructions
// and streams to a go-gfx/webgpu batch.Context, which remaps the guest's
// private ids, checks the stream and its buffer ranges (wire.CheckRanges),
// and replays it on the container's device. The guest is hostile and the
// replay is the trust boundary; see docs/llm-design.md.
//
// Functions (all i32 parameters, an Errno result):
//
//	caps(buf, len)                              the capset blob (wgpubatch.Caps, version 2)
//	context_create(out)                         a context; its handle is stored at out
//	submit(ctx, frame, frame_len, read, read_len)  run a frame, then fill [read, read+read_len)
//	context_destroy(ctx)                        release everything the context holds
//	last_error(ctx, buf, len, out)              the message of the context's last failure
//	                                            (at most 1 KiB, UTF-8): copied to buf, its
//	                                            full length stored at out; Success with
//	                                            length 0 when there is none
//
// A process without the "gpu" capability (run --gpu) gets Denied from
// every function. The container's GPU memory quota is the capability
// "gpu-memory=BYTES" (run --gpu-memory), 256 MiB if absent, at most
// 1.5 GiB. Where there is no GPU backend (a Provider of nil) the answer is
// Unsupported, and a fallback (software) adapter is refused the same way.
package gpubatch

import (
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/go-gfx/webgpu"
	"github.com/go-gfx/webgpu/batch"
	"github.com/go-gfx/webgpu/wire"
	"github.com/go-virtio/gpu/wgpubatch"
	"github.com/go-wasmcontainers/engine/wasi"
)

// ModuleName is the import module.
const ModuleName = "wasi_webgpu_batch"

// Errno is the result of every function.
type Errno = uint32

// Errno values.
const (
	Success         Errno = 0
	InvalidArgument Errno = 1  // a bad handle, or a buffer too small for the capset
	Denied          Errno = 2  // the container has no gpu capability
	Unsupported     Errno = 3  // no GPU backend here, or only a fallback adapter
	TooLarge        Errno = 4  // a frame or a read window above its limit
	QuotaExceeded   Errno = 5  // contexts, GPU memory, pipelines, WGSL size
	Refused         Errno = 6  // a frame or stream the host refuses; the context is lost
	ContextLost     Errno = 7  // the context was lost earlier
	Timeout         Errno = 8  // a submit over its time budget: the device is destroyed, the grant revoked
	Fault           Errno = 9  // a guest address range outside its memory
	Failed          Errno = 10 // the backend failed
	// InvalidShader: the GPU refused an object the guest described: WGSL that
	// does not compile, or a layout, pipeline or bind group WebGPU rejects.
	// The context is lost; last_error returns the compiler's message (WGSL
	// line and column included).
	InvalidShader Errno = 11
	// Revoked: the grant was revoked by an earlier Timeout: every call but
	// context_destroy answers it, for the container's life.
	Revoked Errno = 12
	// Reserved: the guest used the bind group the host reserves for its loop
	// budget (group 3 on a default device): a pipeline layout with more than
	// 3 bind group layouts, a WGSL module with a binding in @group(3), or a
	// SetBindGroup of index 3 or more. The context is lost, the grant is not
	// revoked; last_error names the slot.
	Reserved Errno = 13
	// Truncated: a shader invocation hit the host's loop budget and was cut
	// (layer 4): the stream's results are partial and are not returned. The
	// context is lost and the grant is revoked, as for Timeout; the engine
	// emits a "sanction" event with kind "truncated".
	Truncated Errno = 14
	// Unbounded: the host cannot bound the module's loops, so it does not run
	// it: the WGSL does not parse with the host's parser though the browser
	// accepts it, it has a loop in a module with a vertex or fragment entry
	// point, or the browser refused the loop-budget rewrite (for example a
	// shader that already binds as many storage buffers as the device allows:
	// the capset reports one less). The context is lost, the grant is not
	// revoked; last_error says which.
	Unbounded Errno = 15
)

// MaxErrorMessage bounds the text last_error returns.
const MaxErrorMessage = 1024

// Limits bound one container's use of the module.
type Limits struct {
	Contexts   int    // live contexts (default 4)
	MaxFrame   uint32 // largest frame, bytes (default 32 MiB)
	MaxObjects uint32 // guest ids per context (default 4096)
	MaxRead    uint32 // read window of one submit (default 1 MiB, at most wire.MaxReadWindow)
	// GPUTime is the hard cap of one submit's time on the GPU (default 2 s):
	// what a frame that creates objects (shader and pipeline compilation
	// included) may take. Budget is the steady-state budget of a frame that
	// only streams commands (default 500 ms: the longest legitimate submit
	// measured is 226 ms in Firefox (its 200 ms floor), 43 ms in Chromium; docs/gpu-scheduling.md), at most GPUTime. Both run from
	// the moment the submit gets its turn, not from when it was queued.
	GPUTime, Budget time.Duration
	// DrainTime bounds how long the host waits, after a sanction, for the
	// destroyed device's remaining work to leave the GPU before it serves
	// the next turn (default 30 s).
	DrainTime time.Duration
	// MaxDispatchInvocations and MaxStreamInvocations are the compute quota
	// of one context: no Dispatch, and no stream, may ask for more
	// invocations (workgroup counts times the device's largest workgroup;
	// wire.CheckInvocations). 0: DefaultLimits'.
	MaxDispatchInvocations, MaxStreamInvocations uint64
	// MaxDrawVertices and MaxStreamVertices are the draw quota of one
	// context: no Draw or DrawIndexed, and no stream, may ask for more vertex
	// invocations (index or vertex count times instance count;
	// wire.CheckDraws). They do not depend on the frame size. 0: DefaultLimits'.
	MaxDrawVertices, MaxStreamVertices uint64
	// Memory is the GPU memory quota when the container made no grant
	// (default 256 MiB). MaxMemory is the largest grant honoured (1.5 GiB).
	Memory, MaxMemory int64
}

// DefaultLimits are the quotas of the design (docs/llm-design.md).
var DefaultLimits = Limits{
	Contexts: 4, MaxFrame: 32 << 20, MaxObjects: 4096, MaxRead: 1 << 20, GPUTime: 2 * time.Second, Budget: 500 * time.Millisecond, DrainTime: 30 * time.Second,
	MaxDispatchInvocations: 1 << 26, MaxStreamInvocations: 1 << 30,
	// Draws: a Quake e1m1 frame (go-quake1/engine, 640x480 or any size, the
	// count is of vertices, not pixels) is a few thousand faces of at most
	// 186 indices each, about 60 thousand indices a frame, and an alias model
	// draws a few thousand indices per instance; 1 Mi per draw and 16 Mi per
	// stream leave more than two orders of magnitude of headroom.
	MaxDrawVertices: 1 << 20, MaxStreamVertices: 1 << 24,
	Memory: 256 << 20, MaxMemory: 1610612736,
}

// Provider opens the container's GPU device. Open may wait on the
// platform; the module calls it off the host call.
type Provider interface {
	// Open makes a device whose GPU memory is bounded by memory bytes.
	// ErrFallback or webgpu.ErrFallbackAdapter: only a software adapter;
	// any error becomes Unsupported (ErrFallback) or Failed.
	Open(memory int64) (Device, error)
}

// ErrFallback: the provider refuses a fallback (software) adapter.
var ErrFallback = errors.New("gpubatch: only a fallback adapter")

// Drainer is a Provider that can wait until the GPU has finished the work
// of a destroyed device (WebGPU's destroy() does not cancel submitted work).
// Drain returns when a trivial submit on a device the provider owns has
// completed, or after timeout. A Provider without it gets no drain wait.
type Drainer interface {
	Drain(timeout time.Duration) error
}

// Device is one container's GPU device, shared by its contexts.
type Device interface {
	// Caps describes what the guest may use of it: the adapter's optional
	// features granted, and the limits the device has.
	Caps() (features uint32, limits wgpubatch.DeviceLimits)
	// Backend returns a fresh backend for one context (a batch.Reader if it
	// can read back). Its Close releases what the context created.
	Backend() batch.Backend
	// Destroy loses the device: every context's work fails from now on.
	Destroy()
}

// Module is the wasi_webgpu_batch host module.
type Module struct {
	// Provider opens devices; nil: Unsupported everywhere (native).
	Provider Provider
	Limits   Limits

	sched   sched    // the GPU's turns, shared by every container
	revoked sync.Map // container label -> true: grants revoked for the container's life
	smu     sync.Mutex
	stats   Stats
}

// Stats returns what the module measured so far.
func (m *Module) Stats() Stats {
	m.smu.Lock()
	defer m.smu.Unlock()
	return m.stats
}

func (m *Module) record(f func(s *Stats)) {
	m.smu.Lock()
	f(&m.stats)
	m.smu.Unlock()
}

// NewModule returns the module over a provider (nil: no GPU).
func NewModule(p Provider) *Module { return &Module{Provider: p, Limits: DefaultLimits} }

// HostModules is NewModule as the engine's host-module list.
func HostModules(p Provider) []wasi.HostModule { return []wasi.HostModule{NewModule(p)} }

// Module implements wasi.HostModule.
func (*Module) Module() string { return ModuleName }

// Function indexes, in Funcs order.
const (
	fnCaps = iota
	fnContextCreate
	fnSubmit
	fnContextDestroy
	fnLastError
)

// Funcs implements wasi.HostModule: every argument an i32, Errno returned.
func (*Module) Funcs() []wasi.HostFunc { return slices.Clone(funcs) }

var funcs = []wasi.HostFunc{
	{Name: "caps", Params: "ii", Result: "i"},
	{Name: "context_create", Params: "i", Result: "i"},
	{Name: "submit", Params: "iiiii", Result: "i"},
	{Name: "context_destroy", Params: "i", Result: "i"},
	{Name: "last_error", Params: "iiii", Result: "i"},
}

func (m *Module) limits() Limits {
	l, d := m.Limits, DefaultLimits
	if l.Contexts <= 0 {
		l.Contexts = d.Contexts
	}
	if l.MaxFrame == 0 {
		l.MaxFrame = d.MaxFrame
	}
	if l.MaxObjects == 0 {
		l.MaxObjects = d.MaxObjects
	}
	if l.MaxRead == 0 {
		l.MaxRead = d.MaxRead
	}
	l.MaxRead = min(l.MaxRead, wire.MaxReadWindow)
	if l.GPUTime <= 0 {
		l.GPUTime = d.GPUTime
	}
	if l.Budget <= 0 {
		l.Budget = d.Budget
	}
	l.Budget = min(l.Budget, l.GPUTime)
	if l.DrainTime <= 0 {
		l.DrainTime = d.DrainTime
	}
	if l.MaxDispatchInvocations == 0 {
		l.MaxDispatchInvocations = d.MaxDispatchInvocations
	}
	if l.MaxStreamInvocations == 0 {
		l.MaxStreamInvocations = d.MaxStreamInvocations
	}
	if l.MaxDrawVertices == 0 {
		l.MaxDrawVertices = d.MaxDrawVertices
	}
	if l.MaxStreamVertices == 0 {
		l.MaxStreamVertices = d.MaxStreamVertices
	}
	if l.Memory <= 0 {
		l.Memory = d.Memory
	}
	if l.MaxMemory <= 0 {
		l.MaxMemory = d.MaxMemory
	}
	return l
}

// Open implements wasi.HostModule.
func (m *Module) Open(p *wasi.Proc) wasi.Instance {
	in := &instance{m: m, lim: m.limits(), gpu: p.HasCap("gpu"), ctxs: map[uint32]*gctx{}, proc: p, key: p.Label(), quit: make(chan struct{})}
	in.memory = in.lim.Memory
	if in.key != "" {
		if _, ok := m.revoked.Load(in.key); ok {
			in.revoked.Store(true) // a restarted container is still sanctioned
		}
	}
	if v, ok := p.CapValue("gpu-memory"); ok {
		// A grant that is not a number in range is a configuration error:
		// the module fails closed rather than falling back to a default.
		n, err := strconv.ParseInt(v, 10, 64)
		if err != nil || n <= 0 || n > in.lim.MaxMemory {
			in.gpu = false
		} else {
			in.memory = n
		}
	}
	return in
}

// gctx is one guest context: a batch.Context over its own backend.
type gctx struct {
	c    *batch.Context
	be   batch.Backend // the context's backend: layer 4's counters are read from it
	loop loopStats     // the counters already added to the module's Stats
	lost atomic.Bool
	// closed: release was called; no frame runs on a closed context.
	closed             atomic.Bool
	peak, peakDispatch atomic.Uint64 // batch.Context's peaks after the last frame
	mu                 sync.Mutex    // held while a frame runs: a batch.Context is not concurrent
	errMu              sync.Mutex
	errMsg             string // the last failure's message
}

// release closes the context's backend. If a frame is still running (the
// process ended during a submit) the close waits for it off the caller.
func (c *gctx) release() {
	c.closed.Store(true) // a frame that has not started yet will not run
	if c.mu.TryLock() {
		c.c.Close()
		c.mu.Unlock()
		return
	}
	go func() {
		c.mu.Lock()
		c.c.Close()
		c.mu.Unlock()
	}()
}

type instance struct {
	m      *Module
	lim    Limits
	gpu    bool
	memory int64

	proc    *wasi.Proc
	key     string        // the container's label: revocations outlive a process
	quit    chan struct{} // closed by Close: a submit waiting for its turn gives up
	revoked atomic.Bool

	openMu sync.Mutex // serialises Provider.Open, which waits on the platform
	mu     sync.Mutex
	dev    Device
	ctxs   map[uint32]*gctx
	next   uint32
	closed bool
}

func r(e Errno) wasi.Result { return wasi.Result{Ret: uint64(e)} }

// later runs f off the host call and resumes with then; f runs while the
// guest is blocked, then writes whatever it must to guest memory.
func later(f func() Errno, then func(e Errno, m wasi.Memory) wasi.Result) wasi.Result {
	done := make(chan struct{})
	var e Errno
	go func() {
		e = f()
		close(done)
	}()
	return wasi.Result{Wait: done, Resume: func(m wasi.Memory) wasi.Result { return then(e, m) }}
}

// Call implements wasi.Instance.
func (in *instance) Call(fn int, a []uint64, m wasi.Memory) wasi.Result {
	if fn < 0 || fn > fnLastError || len(a) != len(funcs[fn].Params) {
		return wasi.Result{Trap: ModuleName + ": no such function"}
	}
	u := func(i int) uint32 { return uint32(a[i]) }
	in.mu.Lock()
	closed := in.closed
	in.mu.Unlock()
	switch {
	case closed:
		return r(Failed)
	case !in.gpu:
		return r(Denied)
	case in.revoked.Load() && fn != fnContextDestroy:
		return r(Revoked)
	case in.m.Provider == nil:
		return r(Unsupported)
	}
	switch fn {
	case fnCaps:
		return in.caps(u(0), u(1), m)
	case fnContextCreate:
		return in.contextCreate(u(0), m)
	case fnSubmit:
		return in.submit(u(0), u(1), u(2), u(3), u(4), m)
	case fnLastError:
		return in.lastError(u(0), u(1), u(2), u(3), m)
	default:
		return r(in.contextDestroy(u(0)))
	}
}

// device returns the container's device, opening it at first use. It is
// called off the host call.
func (in *instance) device() (Device, Errno) {
	in.openMu.Lock()
	defer in.openMu.Unlock()
	in.mu.Lock()
	closed, dev := in.closed, in.dev
	in.mu.Unlock()
	if closed {
		return nil, Failed
	}
	if dev != nil {
		return dev, Success
	}
	d, err := in.m.Provider.Open(in.memory)
	switch {
	case errors.Is(err, ErrFallback), errors.Is(err, webgpu.ErrFallbackAdapter), errors.Is(err, webgpu.ErrNoAdapter), errors.Is(err, webgpu.ErrNoWebGPU):
		return nil, Unsupported
	case err != nil:
		// caps has no context, so no last_error: stderr says why.
		fmt.Fprintf(os.Stderr, "gpubatch: opening the GPU device failed: %v\n", err)
		return nil, Failed
	}
	in.mu.Lock()
	if in.closed { // the process ended while the device was opening
		in.mu.Unlock()
		d.Destroy()
		return nil, Failed
	}
	in.dev = d
	in.mu.Unlock()
	return d, Success
}

// capsBlob is the capset the guest sees: the container's grant and the
// device's features and limits.
func (in *instance) capsBlob(d Device) []byte {
	features, limits := d.Caps()
	return wgpubatch.Caps{
		Version: wgpubatch.CapsVersion, StreamVersion: wire.StreamVersion,
		MaxFrame: in.lim.MaxFrame, MaxObjects: in.lim.MaxObjects, MaxRings: 1,
		Features: features &^ wgpubatch.FeatureRing, MaxMemory: uint64(in.memory), Device: limits,
	}.Encode()
}

func (in *instance) caps(buf, n uint32, m wasi.Memory) wasi.Result {
	if n < wgpubatch.CapsSizeV2 {
		return r(InvalidArgument)
	}
	if _, ok := m.Read(buf, wgpubatch.CapsSizeV2); !ok {
		return r(Fault)
	}
	var blob []byte
	return later(func() Errno {
		d, e := in.device()
		if e == Success {
			blob = in.capsBlob(d)
		}
		return e
	}, func(e Errno, m wasi.Memory) wasi.Result {
		if e != Success {
			return r(e)
		}
		if !m.Write(buf, blob) {
			return r(Fault)
		}
		return r(Success)
	})
}

func (in *instance) contextCreate(out uint32, m wasi.Memory) wasi.Result {
	if _, ok := m.Read(out, 4); !ok {
		return r(Fault)
	}
	in.mu.Lock()
	n := len(in.ctxs)
	in.mu.Unlock()
	if n >= in.lim.Contexts {
		return r(QuotaExceeded)
	}
	var h uint32
	return later(func() Errno {
		d, e := in.device()
		if e != Success {
			return e
		}
		_, dl := d.Caps()
		be := d.Backend()
		c := &gctx{be: be, c: batch.New(be, batch.Config{MaxObjects: in.lim.MaxObjects, MaxRead: in.lim.MaxRead, MaxWorkgroups: dl.MaxComputeWorkgroupsPerDimension,
			WorkgroupInvocations: dl.MaxComputeInvocationsPerWorkgroup, MaxDispatchInvocations: in.lim.MaxDispatchInvocations, MaxStreamInvocations: in.lim.MaxStreamInvocations,
			MaxDrawVertices: in.lim.MaxDrawVertices, MaxStreamVertices: in.lim.MaxStreamVertices})}
		in.mu.Lock()
		defer in.mu.Unlock()
		if in.closed || len(in.ctxs) >= in.lim.Contexts {
			c.release()
			return QuotaExceeded
		}
		for {
			in.next++
			if _, used := in.ctxs[in.next]; in.next != 0 && !used {
				break
			}
		}
		h = in.next
		in.ctxs[h] = c
		return Success
	}, func(e Errno, m wasi.Memory) wasi.Result {
		if e != Success {
			return r(e)
		}
		var b [4]byte
		binary.LittleEndian.PutUint32(b[:], h)
		if !m.Write(out, b[:]) {
			in.contextDestroy(h)
			return r(Fault)
		}
		return r(Success)
	})
}

func (in *instance) contextDestroy(h uint32) Errno {
	in.mu.Lock()
	c := in.ctxs[h]
	delete(in.ctxs, h)
	in.mu.Unlock()
	if c == nil {
		return InvalidArgument
	}
	c.release()
	return Success
}

// submit runs one frame. The frame is copied out of guest memory before
// anything reads it; the read window is a host buffer the stream's reads
// fill, copied to the guest only in Resume, once the stream succeeded.
func (in *instance) submit(h, frame, flen, read, rlen uint32, m wasi.Memory) wasi.Result {
	in.mu.Lock()
	c := in.ctxs[h]
	in.mu.Unlock()
	switch {
	case c == nil:
		return r(InvalidArgument)
	case c.lost.Load():
		return r(ContextLost)
	case flen < wgpubatch.FrameHeaderSize || flen > in.lim.MaxFrame || rlen > in.lim.MaxRead:
		return r(TooLarge)
	}
	src, ok := m.Read(frame, flen)
	if !ok {
		return r(Fault)
	}
	fr := append([]byte(nil), src...) // copy, then decode
	if rlen > 0 {
		if _, ok := m.Read(read, rlen); !ok {
			return r(Fault)
		}
	}
	window := make([]byte, rlen)

	done := make(chan struct{})
	var e Errno
	go func() {
		e = in.runSubmit(c, fr, window)
		close(done)
	}()
	return wasi.Result{Wait: done, Resume: func(m wasi.Memory) wasi.Result {
		if e != Success {
			return r(e)
		}
		if rlen > 0 && !m.Write(read, window) {
			return r(Fault)
		}
		return r(Success)
	}}
}

// runSubmit is one submit's life on the host: decode (a refusal costs no
// turn), wait for the GPU's turn (one submit in flight in the whole module,
// round-robin between containers), run the frame under its time budget, and
// on an overrun sanction the container. The budget runs from the start of
// the turn, so a container never pays for the turns of others.
func (in *instance) runSubmit(c *gctx, frame, window []byte) Errno {
	lim := wgpubatch.Limits{MaxFrame: in.lim.MaxFrame, MaxObjects: in.lim.MaxObjects}
	msgs, err := wgpubatch.Decode(frame, lim)
	if err != nil {
		c.lost.Store(true)
		c.release()
		return Refused
	}
	// A frame that creates objects may compile shaders and pipelines, which
	// no budget of the steady state fits: it gets the hard cap.
	budget, creates := in.lim.Budget, false
	for _, msg := range msgs {
		if msg.Type == wgpubatch.MsgCreate {
			budget, creates = in.lim.GPUTime, true
			break
		}
	}
	queued := time.Now()
	rel, err := in.m.sched.acquire(in, in.quit)
	if err != nil {
		return Failed // the process ended while it waited
	}
	if in.revoked.Load() { // sanctioned while this submit waited behind its own
		rel(nil)
		return Revoked
	}
	t0 := time.Now()
	res := make(chan Errno, 1)
	go func() { res <- in.runFrame(c, msgs, lim, window) }()
	timer := time.NewTimer(budget)
	defer timer.Stop()
	select {
	case e := <-res:
		el := time.Since(t0)
		rel(nil)
		in.m.record(func(s *Stats) {
			s.Submits++
			s.MaxWait = max(s.MaxWait, t0.Sub(queued))
			s.MaxSubmit = max(s.MaxSubmit, el)
			if creates {
				s.MaxCreate = max(s.MaxCreate, el)
			} else {
				s.MaxStream = max(s.MaxStream, el)
			}
			s.MaxInvocations = max(s.MaxInvocations, c.peak.Load())
			s.MaxDispatchInvocations = max(s.MaxDispatchInvocations, c.peakDispatch.Load())
		})
		if e == Truncated {
			// Layer 4 cut a loop: the work is already off the GPU, so there
			// is nothing to drain; the grant goes like a timeout's.
			in.m.record(func(s *Stats) { s.Truncations++ })
			kind := "stream"
			if creates {
				kind = "create"
			}
			in.sanction(el, budget, kind, "loop-budget", "a shader exceeded its loop budget and was truncated; the GPU grant is revoked")
		}
		return e
	case <-timer.C:
		el := time.Since(t0)
		if lw, ok := in.m.Provider.(LossWatcher); ok {
			if lt := lw.LastBrowserLoss(); !lt.IsZero() && lt.After(t0.Add(-CollateralWindow)) {
				// Not an overrun of this container: the browser destroyed
				// another device (its watchdog) and this submit, queued
				// behind it, may never complete (Safari does that, #32).
				// The device is reset, the grant kept.
				in.m.record(func(s *Stats) { s.Resets++ })
				in.resetDevice(el, budget)
				rel(nil)
				return ContextLost
			}
		}
		kind := "stream"
		if creates {
			kind = "create"
		}
		in.sanction(el, budget, kind, "time-budget", "a GPU submit overran its time budget; the GPU grant is revoked")
		// The GPU still holds the destroyed device's work: keep the turn
		// until it has left, so the next container's clock starts on an
		// idle GPU.
		rel(in.drain)
		return Timeout
	}
}

// drain waits for the work of a destroyed device to leave the GPU, if the
// provider can tell.
func (in *instance) drain() {
	d, ok := in.m.Provider.(Drainer)
	if !ok {
		return
	}
	in.m.record(func(s *Stats) { s.Drains++ })
	d.Drain(in.lim.DrainTime)
}

// sanction destroys the container's device, revokes its GPU grant for the
// rest of its life (even across a restart of the same container, until the
// engine itself restarts), and reports it to the engine (event and inspect).
// Every context of the container is lost with the device.
func (in *instance) sanction(el, budget time.Duration, frame, cause, reason string) {
	in.revoked.Store(true)
	if in.key != "" {
		in.m.revoked.Store(in.key, true)
	}
	in.m.record(func(s *Stats) { s.Sanctions++ })
	in.loseDevice()
	if in.proc != nil {
		in.proc.Emit("gpu", "sanction", map[string]string{
			"reason":    reason,
			"submit_ms": strconv.FormatInt(el.Milliseconds(), 10), "budget_ms": strconv.FormatInt(budget.Milliseconds(), 10),
			"frame": frame, "cause": cause,
		})
	}
}

// LossWatcher is a Provider that knows when the browser (not the host) last
// destroyed one of its devices.
type LossWatcher interface {
	LastBrowserLoss() time.Time
}

// CollateralWindow is how far before a submit's turn a browser loss of
// another device still explains that the submit never completed.
const CollateralWindow = 10 * time.Second

// resetDevice destroys the container's device without revoking its grant, and
// forgets it: the next context_create opens a new one. Every context is lost
// (ContextLost, which is what a guest answers by recreating its context).
func (in *instance) resetDevice(el, budget time.Duration) {
	in.loseDevice()
	in.mu.Lock()
	in.dev = nil
	in.mu.Unlock()
	if in.proc != nil {
		in.proc.Emit("gpu", "device-reset", map[string]string{
			"reason":    "the browser destroyed another container's device while this submit waited; this container's device is reset, its GPU grant is kept",
			"submit_ms": strconv.FormatInt(el.Milliseconds(), 10), "budget_ms": strconv.FormatInt(budget.Milliseconds(), 10),
		})
	}
}

// loseDevice destroys the container's device: every context is lost, and
// the module opens no new device.
func (in *instance) loseDevice() {
	in.mu.Lock()
	d := in.dev
	for _, c := range in.ctxs {
		c.lost.Store(true)
	}
	in.mu.Unlock()
	if d != nil {
		d.Destroy()
	}
}

// runFrame applies a decoded frame. Any error loses the context and
// releases what it held.
func (in *instance) runFrame(c *gctx, msgs []wgpubatch.Message, lim wgpubatch.Limits, window []byte) (e Errno) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed.Load() {
		return ContextLost
	}
	defer func() {
		if e != Success && e != ContextLost {
			c.lost.Store(true)
			c.c.Close()
		}
	}()
	defer in.recordLoop(c) // a refusal is counted too
	for _, msg := range msgs {
		var err error
		switch msg.Type {
		case wgpubatch.MsgCreate:
			var cr wgpubatch.Create
			if cr, err = msg.Create(lim); err == nil {
				err = c.c.Create(cr.ID, cr.Kind, cr.Descriptor)
			}
		case wgpubatch.MsgDestroy:
			var ids []uint32
			if ids, err = msg.Destroy(lim); err == nil {
				err = c.c.Destroy(ids...)
			}
		case wgpubatch.MsgStream:
			var s wgpubatch.Stream
			if s, err = msg.Stream(); err == nil {
				err = c.c.Stream(s.Words, s.Data, window)
			}
		default: // BIND_RESOURCE: a container has no virtio resources to bind
			err = errors.New("message type not served here")
		}
		if err != nil {
			c.setError(err)
			if e := errno(err); e == Unbounded || e == Reserved {
				// What the host refused, and why, for whoever runs the
				// container (the guest has last_error).
				if in.proc != nil {
					in.proc.Emit("gpu", "refused", map[string]string{"errno": strconv.Itoa(int(e)), "reason": strings.ToValidUTF8(err.Error()[:min(len(err.Error()), 600)], "")})
				}
				return e
			}
			return errno(err)
		}
	}
	c.peak.Store(c.c.PeakInvocations())
	c.peakDispatch.Store(c.c.PeakDispatchInvocations())
	in.recordLoop(c)
	return Success
}

// loopStats are layer 4's counters of one backend.
type loopStats struct {
	modules, rewritten, refused int
	spent                       time.Duration
}

// recordLoop adds to the module's Stats what layer 4 did since the last frame
// of the context (the backend, if it has a loop budget, keeps the totals).
func (in *instance) recordLoop(c *gctx) {
	lb, ok := c.be.(interface {
		LoopBudgetStats() (modules, rewritten, refused int, spent time.Duration)
	})
	if !ok {
		return
	}
	var now loopStats
	now.modules, now.rewritten, now.refused, now.spent = lb.LoopBudgetStats()
	d := loopStats{now.modules - c.loop.modules, now.rewritten - c.loop.rewritten, now.refused - c.loop.refused, now.spent - c.loop.spent}
	c.loop = now
	if d == (loopStats{}) {
		return
	}
	in.m.record(func(s *Stats) {
		s.LoopModules += int64(d.modules)
		s.LoopRewritten += int64(d.rewritten)
		s.LoopRefused += int64(d.refused)
		s.LoopSpent += d.spent
	})
}

// setError records the message of a failure for last_error.
func (c *gctx) setError(err error) {
	m := err.Error()
	if len(m) > MaxErrorMessage {
		m = strings.ToValidUTF8(m[:MaxErrorMessage], "")
	}
	c.errMu.Lock()
	c.errMsg = m
	c.errMu.Unlock()
}

// lastError copies a context's last failure message to the guest.
func (in *instance) lastError(h, buf, n, out uint32, m wasi.Memory) wasi.Result {
	in.mu.Lock()
	c := in.ctxs[h]
	in.mu.Unlock()
	if c == nil {
		return r(InvalidArgument)
	}
	if _, ok := m.Read(out, 4); !ok {
		return r(Fault)
	}
	c.errMu.Lock()
	msg := c.errMsg
	c.errMu.Unlock()
	var l [4]byte
	binary.LittleEndian.PutUint32(l[:], uint32(len(msg)))
	if !m.Write(out, l[:]) {
		return r(Fault)
	}
	if k := min(uint32(len(msg)), n); k > 0 && !m.Write(buf, []byte(msg[:k])) {
		return r(Fault)
	}
	return r(Success)
}

// errno classifies a batch error.
func errno(err error) Errno {
	var q *webgpu.QuotaError
	switch {
	case errors.Is(err, batch.ErrTruncated):
		return Truncated
	case errors.Is(err, batch.ErrReserved):
		return Reserved
	case errors.Is(err, batch.ErrUnbounded):
		return Unbounded
	case errors.Is(err, batch.ErrInvalid):
		return InvalidShader
	case errors.As(err, &q):
		return QuotaExceeded
	case errors.Is(err, batch.ErrLost):
		return ContextLost
	case errors.Is(err, wgpubatch.ErrFrame), errors.Is(err, batch.ErrRefused), errors.Is(err, batch.ErrRead), errors.Is(err, wire.ErrRange):
		return Refused
	}
	var b *webgpu.BoundsError
	if errors.As(err, &b) {
		return Refused
	}
	// A browser exception, a WebGPU error or a destroyed device is the
	// platform's: the guest did nothing the host could have refused.
	var js *webgpu.JSError
	var ge *webgpu.GPUError
	if errors.As(err, &js) || errors.As(err, &ge) || errors.Is(err, webgpu.ErrDestroyed) {
		return Failed
	}
	// What is left is a stream or a descriptor the decoders refused
	// (wire.Decode, Remap, DecodeDescriptor): their errors are plain.
	return Refused
}

// Close implements wasi.Instance: it releases every context and the
// device. A Resume that runs later answers Failed or does nothing.
func (in *instance) Close() {
	in.mu.Lock()
	if !in.closed {
		close(in.quit)
	}
	in.closed = true
	ctxs, d := in.ctxs, in.dev
	in.ctxs, in.dev = map[uint32]*gctx{}, nil
	in.mu.Unlock()
	for _, c := range ctxs {
		c.release()
	}
	if d != nil {
		d.Destroy()
	}
}
