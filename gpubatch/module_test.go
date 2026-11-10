package gpubatch

import (
	"encoding/binary"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/go-gfx/webgpu"
	"github.com/go-gfx/webgpu/batch"
	"github.com/go-gfx/webgpu/wire"
	"github.com/go-virtio/gpu/wgpubatch"
	"github.com/go-wasmcontainers/engine/rootfs"
	"github.com/go-wasmcontainers/engine/wasi"
)

// mem is a guest's linear memory.
type mem []byte

func (m mem) Read(off, n uint32) ([]byte, bool) {
	if uint64(off)+uint64(n) > uint64(len(m)) {
		return nil, false
	}
	return m[off : off+n], true
}

func (m mem) Write(off uint32, b []byte) bool {
	if uint64(off)+uint64(len(b)) > uint64(len(m)) {
		return false
	}
	copy(m[off:], b)
	return true
}

// fakeDevice is a Device without a GPU: a memory quota and buffers that
// read back a pattern of their offset.
type fakeDevice struct {
	mu        sync.Mutex
	memory    int64
	used      int64
	replays   int
	destroyed bool
	delay     time.Duration // each replay takes this long
	delayN    int           // ... but only the first delayN replays, if positive
	live      int           // buffers alive
	features  uint32
	truncate  bool // each replay answers batch.ErrTruncated (layer 4)
	plain     bool // the backend has no loop-budget counters
}

func (d *fakeDevice) Caps() (uint32, wgpubatch.DeviceLimits) {
	return d.features, wgpubatch.DeviceLimits{MaxStorageBufferBindingSize: 128 << 20, MaxComputeWorkgroupStorageSize: 16384,
		MaxComputeInvocationsPerWorkgroup: 256, MaxComputeWorkgroupsPerDimension: 65535, MaxStorageBuffersPerShaderStage: 8}
}
func (d *fakeDevice) Backend() batch.Backend {
	b := &fakeBackend{d: d, sizes: map[uint32]int64{}}
	if d.plain {
		return plainBackend{b}
	}
	return b
}

// plainBackend hides the loop-budget counters of fakeBackend.
type plainBackend struct{ batch.Backend }

// LoopBudgetStats are layer 4's counters: one module, rewritten, per replay,
// 2 ms each.
func (b *fakeBackend) LoopBudgetStats() (modules, rewritten, refused int, spent time.Duration) {
	b.d.mu.Lock()
	defer b.d.mu.Unlock()
	return b.d.replays, b.d.replays, 0, time.Duration(b.d.replays) * 2 * time.Millisecond
}
func (d *fakeDevice) Destroy() { d.mu.Lock(); d.destroyed = true; d.mu.Unlock() }
func (d *fakeDevice) snapshot() (replays, live int, destroyed bool) {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.replays, d.live, d.destroyed
}

type fakeBackend struct {
	d     *fakeDevice
	next  uint32
	sizes map[uint32]int64
}

func (b *fakeBackend) Create(kind uint32, desc []byte) (uint32, error) {
	b.d.mu.Lock()
	defer b.d.mu.Unlock()
	var size int64
	if wire.Kind(kind) == wire.KindShaderModule {
		d, err := webgpu.DecodeDescriptor(wire.KindShaderModule, desc, nil)
		if err != nil {
			return 0, err
		}
		// The real backend asks the GPU; the fake refuses the sources a
		// browser refuses: a reserved word, a syntax error.
		if code := d.(webgpu.ShaderModuleDescriptor).Code; strings.Contains(code, "meta") || strings.Contains(code, "fn main(") {
			return 0, fmt.Errorf("%w: Error while parsing WGSL: :1:42 error: bad source", batch.ErrInvalid)
		}
	}
	if wire.Kind(kind) == wire.KindBuffer {
		d, err := webgpu.DecodeDescriptor(wire.KindBuffer, desc, nil)
		if err != nil {
			return 0, err
		}
		size = int64(d.(webgpu.BufferDescriptor).Size)
		if b.d.memory > 0 && size > b.d.memory-b.d.used {
			return 0, &webgpu.QuotaError{Resource: "memory", Limit: b.d.memory, Want: b.d.used + size}
		}
		b.d.used += size
		b.d.live++
	}
	b.next++
	b.sizes[b.next] = size
	return b.next, nil
}

func (b *fakeBackend) Destroy(id uint32) {
	b.d.mu.Lock()
	defer b.d.mu.Unlock()
	if s, ok := b.sizes[id]; ok {
		if s > 0 {
			b.d.used -= s
			b.d.live--
		}
		delete(b.sizes, id)
	}
}

func (b *fakeBackend) Replay(words []uint32, data []byte) error {
	b.d.mu.Lock()
	b.d.replays++
	delay := b.d.delay
	if b.d.delayN > 0 {
		if b.d.replays > b.d.delayN {
			delay = 0
		}
	}
	truncate := b.d.truncate
	b.d.mu.Unlock()
	time.Sleep(delay)
	if truncate {
		return batch.ErrTruncated
	}
	return nil
}

func (b *fakeBackend) Read(id uint32, off uint64, dst []byte) error {
	for i := range dst {
		dst[i] = byte(off) + byte(i)
	}
	return nil
}

func (b *fakeBackend) Close() {
	for id := range b.sizes {
		b.Destroy(id)
	}
}

type fakeProvider struct {
	loss  time.Time // when the browser last destroyed a device (LossWatcher)
	dev   *fakeDevice
	err   error
	opens int
	mem   int64
}

func (p *fakeProvider) LastBrowserLoss() time.Time { return p.loss }

func (p *fakeProvider) Open(memory int64) (Device, error) {
	p.opens++
	p.mem = memory
	if p.err != nil {
		return nil, p.err
	}
	p.dev.memory = memory
	return p.dev, nil
}

// guest is one process: a module instance and its memory.
type guest struct {
	t    *testing.T
	in   wasi.Instance
	m    mem
	prov *fakeProvider
	dev  *fakeDevice
	mod  *Module
}

func newGuest(t *testing.T, caps []string, tune func(*Module, *fakeProvider)) *guest {
	t.Helper()
	dev := &fakeDevice{}
	prov := &fakeProvider{dev: dev}
	mod := NewModule(prov)
	if tune != nil {
		tune(mod, prov)
	}
	p := wasi.New(wasi.Config{FS: rootfs.New(), Caps: caps})
	in := mod.Open(p)
	t.Cleanup(in.Close)
	return &guest{t: t, in: in, m: make(mem, 1<<20), prov: prov, dev: dev, mod: mod}
}

// call runs one import to completion, as the executor does: waiting, then
// resuming.
func (g *guest) call(fn int, args ...uint64) Errno {
	g.t.Helper()
	r := g.in.Call(fn, args, g.m)
	for r.Wait != nil {
		<-r.Wait
		r = r.Resume(g.m)
	}
	if r.Trap != "" {
		g.t.Fatalf("trap: %s", r.Trap)
	}
	return Errno(r.Ret)
}

func (g *guest) newContext() uint32 {
	g.t.Helper()
	if e := g.call(fnContextCreate, 16); e != Success {
		g.t.Fatalf("context_create: errno %d", e)
	}
	return binary.LittleEndian.Uint32(g.m[16:])
}

// put copies a frame into guest memory at off and submits it with a read
// window of rlen bytes at roff.
func (g *guest) submit(ctx uint32, f []byte, off, roff, rlen uint32) Errno {
	copy(g.m[off:], f)
	return g.call(fnSubmit, uint64(ctx), uint64(off), uint64(len(f)), uint64(roff), uint64(rlen))
}

func bufferFrame(t *testing.T, id uint32, size uint64, usage webgpu.BufferUsage, stream []uint32, data []byte) []byte {
	t.Helper()
	k, d, err := webgpu.EncodeDescriptor(webgpu.BufferDescriptor{Size: size, Usage: usage})
	if err != nil {
		t.Fatal(err)
	}
	e := wgpubatch.NewEncoder(1 << 20)
	e.Create(id, uint32(k), d)
	if stream != nil {
		e.Stream(stream, data)
	}
	f, err := e.Frame()
	if err != nil {
		t.Fatal(err)
	}
	return append([]byte(nil), f...)
}

var gpuCaps = []string{"gpu"}

func TestDeniedAndUnsupported(t *testing.T) {
	for _, tc := range []struct {
		name string
		caps []string
		tune func(*Module, *fakeProvider)
		want Errno
	}{
		{"no gpu capability", nil, nil, Denied},
		{"only a memory grant", []string{"gpu-memory=1048576"}, nil, Denied},
		{"grant above the ceiling", []string{"gpu", "gpu-memory=1610612737"}, nil, Denied},
		{"grant that is not a number", []string{"gpu", "gpu-memory=lots"}, nil, Denied},
		{"zero grant", []string{"gpu", "gpu-memory=0"}, nil, Denied},
		{"no GPU backend (native)", gpuCaps, func(m *Module, _ *fakeProvider) { m.Provider = nil }, Unsupported},
		{"only a fallback adapter", gpuCaps, func(_ *Module, p *fakeProvider) { p.err = ErrFallback }, Unsupported},
		{"fallback adapter from go-gfx", gpuCaps, func(_ *Module, p *fakeProvider) { p.err = webgpu.ErrFallbackAdapter }, Unsupported},
		{"no adapter", gpuCaps, func(_ *Module, p *fakeProvider) { p.err = webgpu.ErrNoAdapter }, Unsupported},
		{"the platform fails", gpuCaps, func(_ *Module, p *fakeProvider) { p.err = errors.New("boom") }, Failed},
	} {
		g := newGuest(t, tc.caps, tc.tune)
		if e := g.call(fnCaps, 0, 256); e != tc.want {
			t.Errorf("%s: caps errno %d, want %d", tc.name, e, tc.want)
		}
		if e := g.call(fnContextCreate, 16); e != tc.want {
			t.Errorf("%s: context_create errno %d, want %d", tc.name, e, tc.want)
		}
		// Without the capability or a backend every function answers alike;
		// where only the device failed to open, no context exists to name.
		submit, destroy := tc.want, tc.want
		if g.prov.opens > 0 {
			submit, destroy = InvalidArgument, InvalidArgument
		}
		if e := g.call(fnSubmit, 1, 0, 32, 0, 0); e != submit {
			t.Errorf("%s: submit errno %d, want %d", tc.name, e, submit)
		}
		if e := g.call(fnContextDestroy, 1); e != destroy {
			t.Errorf("%s: context_destroy errno %d, want %d", tc.name, e, destroy)
		}
	}
}

func TestUnknownFunctionTraps(t *testing.T) {
	g := newGuest(t, gpuCaps, nil)
	for _, c := range []struct {
		fn   int
		args []uint64
	}{{-1, nil}, {4, nil}, {fnCaps, []uint64{1}}, {fnSubmit, nil}} {
		if r := g.in.Call(c.fn, c.args, g.m); r.Trap == "" {
			t.Errorf("Call(%d, %v) did not trap", c.fn, c.args)
		}
	}
}

func TestCaps(t *testing.T) {
	g := newGuest(t, gpuCaps, func(_ *Module, p *fakeProvider) { p.dev.features = wgpubatch.FeatureShaderF16 })
	if e := g.call(fnCaps, 64, wgpubatch.CapsSizeV2-1); e != InvalidArgument {
		t.Errorf("short buffer: %d", e)
	}
	if e := g.call(fnCaps, uint64(len(g.m))-8, 256); e != Fault {
		t.Errorf("buffer outside memory: %d", e)
	}
	if e := g.call(fnCaps, 64, 256); e != Success {
		t.Fatalf("caps: %d", e)
	}
	c, err := wgpubatch.DecodeCaps(g.m[64 : 64+wgpubatch.CapsSizeV2])
	if err != nil {
		t.Fatal(err)
	}
	if c.Version != 2 || c.StreamVersion != wire.StreamVersion || c.MaxFrame != 32<<20 || c.MaxObjects != 4096 ||
		c.MaxMemory != 256<<20 || c.Features != wgpubatch.FeatureShaderF16 || c.Device.MaxComputeWorkgroupsPerDimension != 65535 {
		t.Errorf("caps %+v", c)
	}
	if g.prov.mem != 256<<20 {
		t.Errorf("device opened with %d bytes", g.prov.mem)
	}
	// A grant is what the guest is told and what the device is opened with.
	g2 := newGuest(t, []string{"gpu", "gpu-memory=1610612736"}, nil)
	if e := g2.call(fnCaps, 0, 64); e != Success {
		t.Fatal(e)
	}
	if c, _ := wgpubatch.DecodeCaps(g2.m[:64]); c.MaxMemory != 1610612736 || g2.prov.mem != 1610612736 {
		t.Errorf("grant: caps %d, device %d", c.MaxMemory, g2.prov.mem)
	}
}

func TestContexts(t *testing.T) {
	g := newGuest(t, gpuCaps, nil)
	var hs []uint32
	for i := 0; i < 4; i++ {
		hs = append(hs, g.newContext())
	}
	if e := g.call(fnContextCreate, 16); e != QuotaExceeded {
		t.Errorf("fifth context: %d", e)
	}
	if e := g.call(fnContextCreate, uint64(len(g.m))-2); e != Fault {
		t.Errorf("handle outside memory: %d", e)
	}
	if g.prov.opens != 1 {
		t.Errorf("%d devices for one container", g.prov.opens)
	}
	if e := g.call(fnContextDestroy, uint64(hs[0])); e != Success {
		t.Error(e)
	}
	if e := g.call(fnContextDestroy, uint64(hs[0])); e != InvalidArgument {
		t.Errorf("double destroy: %d", e)
	}
	if e := g.call(fnContextDestroy, 999); e != InvalidArgument {
		t.Errorf("unknown handle: %d", e)
	}
	g.newContext() // the slot is free again
}

// A guest's submit fills its window, and only its window.
func TestSubmitReadBack(t *testing.T) {
	g := newGuest(t, gpuCaps, nil)
	ctx := g.newContext()
	for i := range g.m {
		g.m[i] = 0xAA
	}
	f := bufferFrame(t, 1, 256, webgpu.BufferUsageMapRead|webgpu.BufferUsageCopyDst,
		[]uint32{uint32(wire.ReadBuffer), 1, 8, 0, 16, 4}, nil)
	before := append(mem(nil), g.m...)
	if e := g.submit(ctx, f, 4096, 8192, 32); e != Success {
		t.Fatalf("submit: %d", e)
	}
	// Window offset 4, 16 bytes of buffer offset 8.
	for i := 0; i < 16; i++ {
		if g.m[8192+4+i] != byte(8+i) {
			t.Fatalf("window byte %d = %d", i, g.m[8192+4+i])
		}
	}
	// Outside [8192, 8192+32) only the frame's own copy changed.
	for i := range g.m {
		inWindow := i >= 8192 && i < 8192+32
		inFrame := i >= 4096 && i < 4096+len(f)
		if g.m[i] != before[i] && !inWindow && !inFrame {
			t.Fatalf("guest byte %d outside the window changed", i)
		}
	}
	if r, live, _ := g.dev.snapshot(); r != 1 || live != 1 {
		t.Errorf("%d replays, %d buffers", r, live)
	}
}

// The frame is copied out of guest memory before it is decoded: a guest
// that rewrites it while the host works changes nothing.
func TestFrameCopiedBeforeDecode(t *testing.T) {
	g := newGuest(t, gpuCaps, func(_ *Module, p *fakeProvider) { p.dev.delay = 20 * time.Millisecond })
	ctx := g.newContext()
	f := bufferFrame(t, 1, 64, webgpu.BufferUsageCopyDst, []uint32{uint32(wire.WriteBuffer), 1, 0, 0, 0, 4}, []byte{1, 2, 3, 4})
	copy(g.m[100:], f)
	r := g.in.Call(fnSubmit, []uint64{uint64(ctx), 100, uint64(len(f)), 0, 0}, g.m)
	if r.Wait == nil {
		t.Fatal("submit did not wait")
	}
	clear(g.m[100 : 100+len(f)]) // the guest's other thread scribbles over the frame
	<-r.Wait
	if got := r.Resume(g.m); Errno(got.Ret) != Success {
		t.Fatalf("errno %d", got.Ret)
	}
}

func TestRefusedFramesLoseTheContext(t *testing.T) {
	buf := func(usage webgpu.BufferUsage, w []uint32, d []byte) []byte { return bufferFrame(t, 1, 64, usage, w, d) }
	for _, tc := range []struct {
		name  string
		frame []byte
		rlen  uint32
		want  Errno
		// replayed is whether the backend was reached.
		replayed bool
	}{
		{"write past the buffer (M4)", buf(webgpu.BufferUsageCopyDst, []uint32{uint32(wire.WriteBuffer), 1, 60, 0, 0, 16}, make([]byte, 16)), 0, Refused, false},
		{"copy past the buffer (M4)", buf(webgpu.BufferUsageCopyDst|webgpu.BufferUsageCopySrc, []uint32{uint32(wire.CopyBufferToBuffer), 1, 0, 0, 1, 32, 0, 64, 0}, nil), 0, Refused, false},
		{"dispatch above the limit (M4)", buf(webgpu.BufferUsageCopyDst, []uint32{uint32(wire.BeginComputePass), uint32(wire.Dispatch), 65536, 1, 1, uint32(wire.EndPass)}, nil), 0, Refused, false},
		{"id the context did not create", buf(webgpu.BufferUsageCopyDst, []uint32{uint32(wire.WriteBuffer), 9, 0, 0, 0, 4}, make([]byte, 4)), 0, Refused, false},
		{"read outside the window", buf(webgpu.BufferUsageMapRead, []uint32{uint32(wire.ReadBuffer), 1, 0, 0, 16, 8}, nil), 16, Refused, false},
		{"read of a buffer without MAP_READ", buf(webgpu.BufferUsageCopyDst, []uint32{uint32(wire.ReadBuffer), 1, 0, 0, 16, 0}, nil), 16, Refused, false},
		{"bad magic", make([]byte, 64), 0, Refused, false},
		{"truncated", buf(webgpu.BufferUsageCopyDst, nil, nil)[:30], 0, Refused, false},
		{"bind resource", func() []byte {
			e := wgpubatch.NewEncoder(1 << 20)
			e.BindResource(wgpubatch.BindResource{ID: 1, ResourceID: 1, Offset: 0, Size: 4})
			f, _ := e.Frame()
			return append([]byte(nil), f...)
		}(), 0, Refused, false},
	} {
		g := newGuest(t, gpuCaps, nil)
		ctx := g.newContext()
		if e := g.submit(ctx, tc.frame, 4096, 8192, tc.rlen); e != tc.want {
			t.Errorf("%s: errno %d, want %d", tc.name, e, tc.want)
		}
		if r, live, _ := g.dev.snapshot(); (r > 0) != tc.replayed || live != 0 {
			t.Errorf("%s: %d replays, %d buffers left (the lost context must release them)", tc.name, r, live)
		}
		if e := g.submit(ctx, buf(webgpu.BufferUsageCopyDst, nil, nil), 4096, 0, 0); e != ContextLost {
			t.Errorf("%s: after the refusal, errno %d, want ContextLost", tc.name, e)
		}
		// Other contexts of the container are unaffected.
		other := g.newContext()
		if e := g.submit(other, bufferFrame(t, 1, 64, webgpu.BufferUsageCopyDst, nil, nil), 4096, 0, 0); e != Success {
			t.Errorf("%s: another context: %d", tc.name, e)
		}
	}
}

func TestSubmitLimitsAndFaults(t *testing.T) {
	g := newGuest(t, gpuCaps, nil)
	ctx := g.newContext()
	valid := bufferFrame(t, 1, 64, webgpu.BufferUsageCopyDst, nil, nil)
	copy(g.m, valid)
	n := uint64(len(valid))
	for _, tc := range []struct {
		name string
		args []uint64
		want Errno
	}{
		{"unknown context", []uint64{99, 0, n, 0, 0}, InvalidArgument},
		{"frame above max_frame", []uint64{uint64(ctx), 0, 32<<20 + 4, 0, 0}, TooLarge},
		{"frame below a header", []uint64{uint64(ctx), 0, 8, 0, 0}, TooLarge},
		{"frame length 0xffffffff", []uint64{uint64(ctx), 0, 0xffffffff, 0, 0}, TooLarge},
		{"window above 1 MiB", []uint64{uint64(ctx), 0, n, 0, 1<<20 + 4}, TooLarge},
		{"window length 0xffffffff", []uint64{uint64(ctx), 0, n, 0, 0xffffffff}, TooLarge},
		{"frame outside memory", []uint64{uint64(ctx), uint64(len(g.m)) - 4, n, 0, 0}, Fault},
		{"frame address 0xffffffff", []uint64{uint64(ctx), 0xffffffff, n, 0, 0}, Fault},
		{"window outside memory", []uint64{uint64(ctx), 0, n, uint64(len(g.m)) - 4, 16}, Fault},
	} {
		if e := g.call(fnSubmit, tc.args...); e != tc.want {
			t.Errorf("%s: errno %d, want %d", tc.name, e, tc.want)
		}
	}
	// None of those refusals lost the context: they are the call's, not the frame's.
	if e := g.submit(ctx, valid, 0, 0, 0); e != Success {
		t.Errorf("after argument refusals: %d", e)
	}
}

func TestQuotaAndIsolation(t *testing.T) {
	g := newGuest(t, []string{"gpu", "gpu-memory=1048576"}, nil)
	a, b := g.newContext(), g.newContext()
	if e := g.submit(a, bufferFrame(t, 1, 2<<20, webgpu.BufferUsageStorage, nil, nil), 4096, 0, 0); e != QuotaExceeded {
		t.Errorf("a 2 MiB buffer under a 1 MiB grant: %d", e)
	}
	// The grant is summed over contexts: b may hold 512 KiB, a second 768 KiB is refused.
	if e := g.submit(b, bufferFrame(t, 1, 512<<10, webgpu.BufferUsageStorage, nil, nil), 4096, 0, 0); e != Success {
		t.Fatalf("512 KiB: %d", e)
	}
	c := g.newContext()
	if e := g.submit(c, bufferFrame(t, 1, 768<<10, webgpu.BufferUsageStorage, nil, nil), 4096, 0, 0); e != QuotaExceeded {
		t.Errorf("768 KiB on top of 512 KiB: %d", e)
	}
	// Ids are per context: c's id 1 is not b's id 1.
	d := g.newContext()
	if e := g.submit(d, bufferFrame(t, 2, 64, webgpu.BufferUsageCopyDst,
		[]uint32{uint32(wire.WriteBuffer), 1, 0, 0, 0, 4}, make([]byte, 4)), 4096, 0, 0); e != Refused {
		t.Errorf("an id of another context: %d", e)
	}
	// Destroying a context returns its memory.
	if e := g.call(fnContextDestroy, uint64(b)); e != Success {
		t.Fatal(e)
	}
	e2 := g.newContext()
	if e := g.submit(e2, bufferFrame(t, 1, 768<<10, webgpu.BufferUsageStorage, nil, nil), 4096, 0, 0); e != Success {
		t.Errorf("after the release: %d", e)
	}
}

func TestGPUTimeOverrun(t *testing.T) {
	g := newGuest(t, gpuCaps, func(m *Module, p *fakeProvider) {
		m.Limits.GPUTime = 30 * time.Millisecond
		p.dev.delay = 300 * time.Millisecond
	})
	a, b := g.newContext(), g.newContext()
	t0 := time.Now()
	slow := bufferFrame(t, 1, 64, webgpu.BufferUsageCopyDst, []uint32{uint32(wire.WriteBuffer), 1, 0, 0, 0, 4}, make([]byte, 4))
	if e := g.submit(a, slow, 4096, 0, 0); e != Timeout {
		t.Fatalf("errno %d, want Timeout", e)
	}
	if d := time.Since(t0); d > 250*time.Millisecond {
		t.Errorf("the guest waited %v for a 30 ms deadline", d)
	}
	if _, _, destroyed := g.dev.snapshot(); !destroyed {
		t.Error("the device was not destroyed")
	}
	// Every context of the container is lost with the device.
	// ... and the grant is revoked for good.
	if e := g.submit(b, bufferFrame(t, 1, 64, webgpu.BufferUsageCopyDst, nil, nil), 4096, 0, 0); e != Revoked {
		t.Errorf("another context: %d, want Revoked", e)
	}
}

func TestCloseDuringSubmit(t *testing.T) {
	g := newGuest(t, gpuCaps, func(_ *Module, p *fakeProvider) { p.dev.delay = 30 * time.Millisecond })
	ctx := g.newContext()
	f := bufferFrame(t, 1, 64, webgpu.BufferUsageCopyDst, []uint32{uint32(wire.WriteBuffer), 1, 0, 0, 0, 4}, make([]byte, 4))
	copy(g.m, f)
	r := g.in.Call(fnSubmit, []uint64{uint64(ctx), 0, uint64(len(f)), 0, 0}, g.m)
	g.in.Close() // the process ended while the host worked
	<-r.Wait
	r.Resume(g.m) // must not panic
	// The context is released once the running frame is done.
	var live int
	var destroyed bool
	for i := 0; i < 200; i++ {
		if _, live, destroyed = g.dev.snapshot(); live == 0 && destroyed {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if live != 0 || !destroyed {
		t.Errorf("%d buffers left, destroyed %v", live, destroyed)
	}
	if e := g.call(fnCaps, 0, 64); e != Failed {
		t.Errorf("a call after Close: %d", e)
	}
}

func TestErrno(t *testing.T) {
	for _, tc := range []struct {
		err  error
		want Errno
	}{
		{&webgpu.QuotaError{Resource: "memory"}, QuotaExceeded},
		{batch.ErrLost, ContextLost},
		{batch.ErrRefused, Refused},
		{batch.ErrRead, Refused},
		{wire.ErrRange, Refused},
		{&webgpu.BoundsError{Op: "x"}, Refused},
		{wgpubatch.ErrFrame, Refused},
		{errors.New("replay: unknown object 3"), Refused},
		{&webgpu.JSError{Op: "x"}, Failed},
		{&webgpu.GPUError{Kind: "validation"}, Failed},
		{fmt.Errorf("%w: x", batch.ErrInvalid), InvalidShader},
		{batch.ErrTruncated, Truncated},
		{fmt.Errorf("%w: group 3", batch.ErrReserved), Reserved},
		{webgpu.ErrDestroyed, Failed},
	} {
		if got := errno(tc.err); got != tc.want {
			t.Errorf("errno(%v) = %d, want %d", tc.err, got, tc.want)
		}
	}
}

func TestLimitsDefaultsAndHostModules(t *testing.T) {
	l := (&Module{}).limits()
	if l != DefaultLimits {
		t.Errorf("zero Limits give %+v, want %+v", l, DefaultLimits)
	}
	l = (&Module{Limits: Limits{MaxRead: 1 << 30}}).limits()
	if l.MaxRead != wire.MaxReadWindow {
		t.Errorf("MaxRead %d above the wire's window", l.MaxRead)
	}
	// The draw quota (go-gfx/webgpu#30, G3) has defaults, and a set value is kept.
	if l.MaxDrawVertices != 1<<20 || l.MaxStreamVertices != 1<<24 {
		t.Errorf("draw quota %d / %d", l.MaxDrawVertices, l.MaxStreamVertices)
	}
	if l = (&Module{Limits: Limits{MaxDrawVertices: 5, MaxStreamVertices: 7}}).limits(); l.MaxDrawVertices != 5 || l.MaxStreamVertices != 7 {
		t.Errorf("a set draw quota became %d / %d", l.MaxDrawVertices, l.MaxStreamVertices)
	}
	hm := HostModules(nil)
	if len(hm) != 1 || hm[0].Module() != "wasi_webgpu_batch" || len(hm[0].Funcs()) != 5 {
		t.Errorf("%v", hm)
	}
	for i, f := range hm[0].Funcs() {
		if f.Result != "i" || len(f.Params) != len(funcs[i].Params) {
			t.Errorf("func %d: %+v", i, f)
		}
	}
}

func shaderFrame(t *testing.T, id uint32, code string) []byte {
	t.Helper()
	k, d, err := webgpu.EncodeDescriptor(webgpu.ShaderModuleDescriptor{Code: code})
	if err != nil {
		t.Fatal(err)
	}
	e := wgpubatch.NewEncoder(1 << 20)
	e.Create(id, uint32(k), d)
	f, err := e.Frame()
	if err != nil {
		t.Fatal(err)
	}
	return append([]byte(nil), f...)
}

// lastError reads last_error into guest memory at 100 and returns the message
// and the length the host reported.
func (g *guest) lastError(ctx uint32, capacity uint32) (string, uint32) {
	g.t.Helper()
	if e := g.call(fnLastError, uint64(ctx), 100, uint64(capacity), 32); e != Success {
		g.t.Fatalf("last_error: errno %d", e)
	}
	n := binary.LittleEndian.Uint32(g.m[32:])
	return string(g.m[100 : 100+min(n, capacity)]), n
}

// An invalid shader fails loudly: its own errno, the message through
// last_error, the context lost; a valid one is the control.
func TestInvalidShaderIsLoud(t *testing.T) {
	for _, tc := range []struct {
		name, code string
		want       Errno
	}{
		{"valid control", "@compute @workgroup_size(1) fn k() {}", Success},
		{"reserved word", "var<storage, read> meta: array<u32>;", InvalidShader},
		{"syntax error", "@compute @workgroup_size(1) fn main( { }", InvalidShader},
	} {
		g := newGuest(t, gpuCaps, nil)
		ctx := g.newContext()
		if m, n := g.lastError(ctx, 64); m != "" || n != 0 {
			t.Errorf("%s: a fresh context has the error %q", tc.name, m)
		}
		if e := g.submit(ctx, shaderFrame(t, 1, tc.code), 4096, 0, 0); e != tc.want {
			t.Errorf("%s: errno %d, want %d", tc.name, e, tc.want)
		}
		msg, n := g.lastError(ctx, 1024)
		if tc.want == Success {
			if msg != "" {
				t.Errorf("%s: unexpected message %q", tc.name, msg)
			}
			continue
		}
		if !strings.Contains(msg, "Error while parsing WGSL") || int(n) != len(msg) {
			t.Errorf("%s: message %q (length %d)", tc.name, msg, n)
		}
		// A short buffer gets a prefix and the full length.
		if short, full := g.lastError(ctx, 5); short != msg[:5] || full != n {
			t.Errorf("%s: short read %q of %d", tc.name, short, full)
		}
		if e := g.submit(ctx, shaderFrame(t, 2, "@compute @workgroup_size(1) fn k() {}"), 4096, 0, 0); e != ContextLost {
			t.Errorf("%s: after the failure, errno %d, want ContextLost", tc.name, e)
		}
	}
	g := newGuest(t, gpuCaps, nil)
	if e := g.call(fnLastError, 99, 100, 8, 32); e != InvalidArgument {
		t.Errorf("last_error of no context: %d", e)
	}
	ctx := g.newContext()
	if e := g.call(fnLastError, uint64(ctx), 100, 8, uint64(len(g.m)-1)); e != Fault {
		t.Errorf("last_error with the length outside memory: %d", e)
	}
}

// Layer 4: a stream whose flag was set answers Truncated, the container is
// sanctioned like for a timeout (device destroyed, grant revoked), and the
// module counts it.
func TestTruncatedIsASanction(t *testing.T) {
	g := newGuest(t, gpuCaps, func(_ *Module, p *fakeProvider) { p.dev.truncate = true })
	a, b := g.newContext(), g.newContext()
	f := bufferFrame(t, 1, 64, webgpu.BufferUsageCopyDst, []uint32{uint32(wire.WriteBuffer), 1, 0, 0, 0, 4}, make([]byte, 4))
	if e := g.submit(a, f, 4096, 0, 0); e != Truncated {
		t.Fatalf("errno %d, want Truncated", e)
	}
	if _, _, destroyed := g.dev.snapshot(); !destroyed {
		t.Error("the device was not destroyed")
	}
	if e := g.submit(b, bufferFrame(t, 1, 64, webgpu.BufferUsageCopyDst, nil, nil), 4096, 0, 0); e != Revoked {
		t.Errorf("another context: %d, want Revoked", e)
	}
	if s := g.mod.Stats(); s.Truncations != 1 || s.Sanctions != 1 {
		t.Errorf("stats %+v", s)
	}
}

// The loop-budget counters of the backend are added to the module's Stats
// once per frame, as differences.
func TestLoopStats(t *testing.T) {
	g := newGuest(t, gpuCaps, nil)
	a := g.newContext()
	f := bufferFrame(t, 1, 64, webgpu.BufferUsageCopyDst, []uint32{uint32(wire.WriteBuffer), 1, 0, 0, 0, 4}, make([]byte, 4))
	for i := 0; i < 2; i++ {
		if e := g.submit(a, f, 4096, 0, 0); e != Success {
			t.Fatalf("errno %d", e)
		}
		f = streamFrame(t) // the second frame only streams
	}
	if s := g.mod.Stats(); s.LoopModules != 2 || s.LoopRewritten != 2 || s.LoopRefused != 0 || s.LoopSpent != 4*time.Millisecond {
		t.Errorf("stats %+v", s)
	}
	// A backend without the counters adds nothing and does not fail.
	p := newGuest(t, gpuCaps, func(_ *Module, p *fakeProvider) { p.dev.plain = true })
	c := p.newContext()
	if e := p.submit(c, bufferFrame(t, 1, 64, webgpu.BufferUsageCopyDst, nil, nil), 4096, 0, 0); e != Success {
		t.Fatalf("plain backend: errno %d", e)
	}
	if s := p.mod.Stats(); s.LoopModules != 0 {
		t.Errorf("stats %+v", s)
	}
}

// A submit that never completes because the browser destroyed ANOTHER
// device is not an overrun: no sanction, no revocation; the device is reset
// and the guest answered ContextLost, then opens a new context (#32).
func TestCollateralLossIsNotASanction(t *testing.T) {
	g := newGuest(t, gpuCaps, func(m *Module, p *fakeProvider) {
		m.Limits.GPUTime, m.Limits.Budget = 60*time.Millisecond, 60*time.Millisecond
		p.dev.delay, p.dev.delayN = 400*time.Millisecond, 1
		p.loss = time.Now() // another device was lost by the browser just now
	})
	a := g.newContext()
	f := bufferFrame(t, 1, 64, webgpu.BufferUsageCopyDst, []uint32{uint32(wire.WriteBuffer), 1, 0, 0, 0, 4}, make([]byte, 4))
	if e := g.submit(a, f, 4096, 0, 0); e != ContextLost {
		t.Fatalf("errno %d, want ContextLost", e)
	}
	if s := g.mod.Stats(); s.Sanctions != 0 || s.Resets != 1 {
		t.Errorf("stats %+v: a collateral loss is a reset, not a sanction", s)
	}
	// The grant is kept: a new context works on a new device.
	b := g.newContext()
	if e := g.submit(b, bufferFrame(t, 1, 64, webgpu.BufferUsageCopyDst, nil, nil), 4096, 0, 0); e != Success {
		t.Errorf("after the reset: errno %d", e)
	}
	// Without a recent browser loss the same stall IS an overrun.
	h := newGuest(t, gpuCaps, func(m *Module, p *fakeProvider) {
		m.Limits.GPUTime, m.Limits.Budget = 60*time.Millisecond, 60*time.Millisecond
		p.dev.delay = 400 * time.Millisecond
		p.loss = time.Now().Add(-time.Hour)
	})
	if e := h.submit(h.newContext(), f, 4096, 0, 0); e != Timeout {
		t.Errorf("an old loss: errno %d, want Timeout", e)
	}
}
