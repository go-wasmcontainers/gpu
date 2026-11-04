package gpubatch

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-gfx/webgpu"
	"github.com/go-gfx/webgpu/wire"
	"github.com/go-virtio/gpu/wgpubatch"
	"github.com/go-wasmcontainers/engine/rootfs"
	"github.com/go-wasmcontainers/engine/wasi"
)

// TestSchedRoundRobin: with owner A holding the GPU and three more submits of
// A queued before one of B, B is served right after A's current turn, not
// after all of A's: no owner fills the queue.
func TestSchedRoundRobin(t *testing.T) {
	var s sched
	rel, err := s.acquire("A", nil)
	if err != nil {
		t.Fatal(err)
	}
	var mu sync.Mutex
	var order []string
	var wg sync.WaitGroup
	enqueue := func(owner string) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			r, err := s.acquire(owner, nil)
			if err != nil {
				t.Error(err)
				return
			}
			mu.Lock()
			order = append(order, owner)
			mu.Unlock()
			r(nil)
		}()
	}
	waiting := func(n int) {
		for i := 0; i < 500; i++ {
			s.mu.Lock()
			total := 0
			for _, q := range s.queues {
				total += len(q)
			}
			s.mu.Unlock()
			if total == n {
				return
			}
			time.Sleep(2 * time.Millisecond)
		}
		t.Fatalf("never %d waiting", n)
	}
	for i := 0; i < 3; i++ {
		enqueue("A")
		waiting(i + 1)
	}
	enqueue("B")
	waiting(4)
	rel(nil)
	wg.Wait()
	if len(order) != 4 || order[0] != "A" || order[1] != "B" {
		t.Errorf("served %v, want A then B then the rest of A", order)
	}
}

// TestSchedAbortAndDrain: an abandoned wait costs nobody a turn, and a
// release with a drain keeps the GPU reserved until the drain returns.
func TestSchedAbortAndDrain(t *testing.T) {
	var s sched
	rel, _ := s.acquire("A", nil)
	abort := make(chan struct{})
	errc := make(chan error, 1)
	go func() { _, err := s.acquire("B", abort); errc <- err }()
	for i := 0; i < 500; i++ {
		s.mu.Lock()
		n := len(s.ring)
		s.mu.Unlock()
		if n == 1 {
			break
		}
		time.Sleep(time.Millisecond)
	}
	close(abort)
	if err := <-errc; err != errAborted {
		t.Fatalf("abort: %v", err)
	}
	var drained atomic.Bool
	release := make(chan struct{})
	rel(func() { <-release; drained.Store(true) })
	got := make(chan struct{})
	go func() {
		r, _ := s.acquire("C", nil)
		if !drained.Load() {
			t.Error("C ran before the drain returned")
		}
		r(nil)
		close(got)
	}()
	time.Sleep(30 * time.Millisecond)
	select {
	case <-got:
		t.Fatal("C was served during the drain")
	default:
	}
	close(release)
	<-got
}

type drainProvider struct {
	*fakeProvider
	wait    time.Duration
	drained atomic.Int32
}

func (p *drainProvider) Drain(time.Duration) error { p.drained.Add(1); time.Sleep(p.wait); return nil }

// streamFrame is a frame with only a stream (no creation).
func streamFrame(t *testing.T) []byte {
	t.Helper()
	e := wgpubatch.NewEncoder(1 << 20)
	e.Stream([]uint32{uint32(wire.WriteBuffer), 1, 0, 0, 0, 4}, make([]byte, 4))
	f, err := e.Frame()
	if err != nil {
		t.Fatal(err)
	}
	return append([]byte(nil), f...)
}

// TestBudgetAndCap: a stream-only frame is held to Budget, a frame that
// creates objects to the hard cap GPUTime.
func TestBudgetAndCap(t *testing.T) {
	g := newGuest(t, gpuCaps, func(m *Module, p *fakeProvider) {
		m.Limits.GPUTime, m.Limits.Budget = 400*time.Millisecond, 40*time.Millisecond
		p.dev.delay = 100 * time.Millisecond
		p.dev.delayN = 0
	})
	c := g.newContext()
	// 100 ms creation frame: over Budget, under the cap: fine.
	if e := g.submit(c, bufferFrame(t, 1, 64, webgpu.BufferUsageCopyDst, []uint32{uint32(wire.WriteBuffer), 1, 0, 0, 0, 4}, make([]byte, 4)), 4096, 0, 0); e != Success {
		t.Fatalf("create frame within the cap: %d", e)
	}
	if st := g.mod.Stats(); st.Sanctions != 0 || st.MaxCreate < 90*time.Millisecond || st.MaxStream != 0 {
		t.Errorf("stats %+v", st)
	}
	// The same 100 ms as a stream-only frame: over the steady budget.
	if e := g.submit(c, streamFrame(t), 4096, 0, 0); e != Timeout {
		t.Fatalf("stream frame over the budget: %d, want Timeout", e)
	}
	if e := g.call(fnContextCreate, 16); e != Revoked {
		t.Errorf("context_create after the sanction: %d, want Revoked", e)
	}
	if e := g.call(fnCaps, 0, 256); e != Revoked {
		t.Errorf("caps after the sanction: %d, want Revoked", e)
	}
	if e := g.call(fnContextDestroy, uint64(c)); e != Success {
		t.Errorf("context_destroy stays allowed: %d", e)
	}
	if st := g.mod.Stats(); st.Sanctions != 1 {
		t.Errorf("stats %+v", st)
	}
}

// TestSanctionEventAndRestart: the sanction is reported through Proc.Emit,
// and a new process of the same container (same label) starts revoked.
func TestSanctionEventAndRestart(t *testing.T) {
	var mu sync.Mutex
	var got []map[string]string
	prov := &fakeProvider{dev: &fakeDevice{delay: 200 * time.Millisecond}}
	mod := NewModule(prov)
	mod.Limits.GPUTime, mod.Limits.Budget = 30*time.Millisecond, 30*time.Millisecond
	open := func() wasi.Instance {
		p := wasi.New(wasi.Config{FS: rootfs.New(), Caps: gpuCaps, Label: "c-1", OnEvent: func(module, action string, attrs map[string]string) {
			mu.Lock()
			defer mu.Unlock()
			if module == "gpu" && action == "sanction" {
				got = append(got, attrs)
			}
		}})
		in := mod.Open(p)
		t.Cleanup(in.Close)
		return in
	}
	g := &guest{t: t, in: open(), m: make(mem, 1<<20), prov: prov, dev: prov.dev, mod: mod}
	c := g.newContext()
	if e := g.submit(c, bufferFrame(t, 1, 64, webgpu.BufferUsageCopyDst, []uint32{uint32(wire.WriteBuffer), 1, 0, 0, 0, 4}, make([]byte, 4)), 4096, 0, 0); e != Timeout {
		t.Fatalf("errno %d", e)
	}
	mu.Lock()
	n, a := len(got), map[string]string{}
	if n > 0 {
		a = got[0]
	}
	mu.Unlock()
	if n != 1 || a["budget_ms"] != "30" || a["submit_ms"] == "" || a["reason"] == "" {
		t.Fatalf("events %d, attrs %v", n, a)
	}
	g2 := &guest{t: t, in: open(), m: make(mem, 1<<20), prov: prov, dev: prov.dev, mod: mod}
	if e := g2.call(fnContextCreate, 16); e != Revoked {
		t.Errorf("a restart of the sanctioned container: %d, want Revoked", e)
	}
	other := wasi.New(wasi.Config{FS: rootfs.New(), Caps: gpuCaps, Label: "c-2"})
	in := mod.Open(other)
	defer in.Close()
	g3 := &guest{t: t, in: in, m: make(mem, 1<<20), prov: prov, dev: prov.dev, mod: mod}
	if e := g3.call(fnContextCreate, 16); e != Success {
		t.Errorf("another container: %d", e)
	}
}

// TestVictimNotChargedForDrain: after a sanction the destroyed device's work
// is waited out before the next turn, and a victim's budget starts after it:
// a submit that waited longer than its whole budget behind the hostile's
// remaining work is not sanctioned (the false positive of go-wasmcontainers/gpu#26).
func TestVictimNotChargedForDrain(t *testing.T) {
	dev := &fakeDevice{delay: 150 * time.Millisecond, delayN: 1}
	prov := &drainProvider{fakeProvider: &fakeProvider{dev: dev}, wait: 300 * time.Millisecond}
	mod := NewModule(prov)
	mod.Limits.GPUTime, mod.Limits.Budget = 40*time.Millisecond, 40*time.Millisecond
	mk := func(label string) *guest {
		in := mod.Open(wasi.New(wasi.Config{FS: rootfs.New(), Caps: gpuCaps, Label: label}))
		t.Cleanup(in.Close)
		return &guest{t: t, in: in, m: make(mem, 1<<20), prov: prov.fakeProvider, dev: dev, mod: mod}
	}
	hostile, victim := mk("hostile"), mk("victim")
	hc, vc := hostile.newContext(), victim.newContext()
	f := bufferFrame(t, 1, 64, webgpu.BufferUsageCopyDst, []uint32{uint32(wire.WriteBuffer), 1, 0, 0, 0, 4}, make([]byte, 4))
	res := make(chan Errno, 1)
	go func() { res <- hostile.submit(hc, f, 4096, 0, 0) }()
	time.Sleep(15 * time.Millisecond) // the hostile holds the turn
	t0 := time.Now()
	if e := victim.submit(vc, f, 4096, 0, 0); e != Success {
		t.Fatalf("victim: errno %d, want Success", e)
	}
	waited := time.Since(t0)
	if e := <-res; e != Timeout {
		t.Fatalf("hostile: errno %d, want Timeout", e)
	}
	if waited < 250*time.Millisecond {
		t.Errorf("the victim waited only %v: the drain was not waited out", waited)
	}
	if prov.drained.Load() != 1 {
		t.Errorf("%d drains", prov.drained.Load())
	}
	if st := mod.Stats(); st.Sanctions != 1 || st.Drains != 1 {
		t.Errorf("stats %+v", st)
	}
}

// TestCompute quota is enforced by the module: a dispatch over the quota is
// refused (Refused), and the refusal is not a sanction.
func TestComputeQuota(t *testing.T) {
	g := newGuest(t, gpuCaps, func(m *Module, _ *fakeProvider) {
		m.Limits.MaxDispatchInvocations = 256 * 10
	})
	c := g.newContext()
	disp := func(n uint32) []uint32 {
		return []uint32{uint32(wire.BeginComputePass), uint32(wire.Dispatch), n, 1, 1, uint32(wire.EndPass)}
	}
	if e := g.submit(c, bufferFrame(t, 1, 64, webgpu.BufferUsageCopyDst, disp(10), nil), 4096, 0, 0); e != Success {
		t.Fatalf("a dispatch at the quota: %d", e)
	}
	if st := g.mod.Stats(); st.MaxInvocations != 2560 {
		t.Errorf("max invocations %d", st.MaxInvocations)
	}
	if e := g.submit(c, bufferFrame(t, 2, 64, webgpu.BufferUsageCopyDst, disp(11), nil), 4096, 0, 0); e != Refused {
		t.Errorf("a dispatch over the quota: %d, want Refused", e)
	}
	if st := g.mod.Stats(); st.Sanctions != 0 {
		t.Errorf("a refusal sanctioned: %+v", st)
	}
}
