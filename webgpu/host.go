// Package webgpu is the host side of wasi:webgpu (wasi-gfx,
// wasi:webgpu@0.3.0-rc.2) for core modules: every function of the WIT
// interface is a typed import of one host module, lowered and lifted with
// the Canonical ABI (package wit) by hand, without a component-model
// runtime. Resources are handles in a per-process table.
//
// The functions run against a Backend: in the browser, WebGPU in the
// engine's worker (backend_js.go); elsewhere none, so get-gpu works and
// request-adapter answers "none", and a guest can test for a GPU
// portably. A process without the container's "gpu" capability (run
// --gpu) gets no adapter either.
package webgpu

import (
	"errors"
	"fmt"
	"sync"

	"github.com/go-wasmcontainers/engine/wasi"
	witdata "github.com/go-wasmcontainers/gpu/webgpu/wit"
	"github.com/go-wasmcontainers/gpu/wit"
)

// Interface is the parsed wasi:webgpu interface.
var Interface = func() *wit.Interface {
	ifs, err := wit.Parse(witdata.WebGPU)
	if err != nil {
		panic(err)
	}
	return ifs["webgpu"]
}()

// SurfaceInterfaces are wasi-gfx:surface's surface and surface-webgpu, as
// this module declares and serves them (webgpu/wit/surface.wit).
var SurfaceInterfaces = func() map[string]*wit.Interface {
	ifs, err := wit.ParseFiles([]*wit.Interface{Interface}, witdata.Surface)
	if err != nil {
		panic(err)
	}
	return ifs
}()

// export is one import the module serves.
type export struct {
	name     string // core import name
	resource string // owning resource, "" for get-gpu
	fn       *wit.Func
	drop     bool // [resource-drop]
}

// Limits bound one process's use of the GPU.
type Limits struct {
	Handles     int   // live resources (default 4096)
	MemoryBytes int64 // buffer and texture bytes alive (default 256 MiB)
	ShaderBytes int   // one WGSL source (default 1 MiB)
	Pipelines   int   // pipelines alive (default 256)
}

// DefaultLimits are the per-process quotas.
var DefaultLimits = Limits{Handles: 4096, MemoryBytes: 256 << 20, ShaderBytes: 1 << 20, Pipelines: 256}

// Process is what a Backend knows of the process it serves.
type Process struct {
	GPU       bool   // the process has the gpu capability
	Container string // its container's label (wasi.Config.Label), for surfaces
}

// Backend serves the functions for one process.
type Backend interface {
	// Open starts a process's session.
	Open(p Process, lim Limits, h *Handles) Session
}

// Session is one process's GPU state.
type Session interface {
	// Call runs import name with lifted arguments and returns the result
	// value (nil for none). A *TrapError traps the guest.
	Call(name string, args []any) (any, error)
	// Async reports whether name waits on the platform (WebGPU promises):
	// such calls run off the host call.
	Async(name string) bool
	// Close releases everything (the process ended).
	Close()
}

// TrapError traps the guest (the Canonical ABI's answer to an invalid
// handle, or a quota exceeded where the API has no error to report).
type TrapError struct{ Msg string }

func (e *TrapError) Error() string { return "wasi:webgpu: " + e.Msg }

// Trapf makes a TrapError.
func Trapf(format string, a ...any) error { return &TrapError{fmt.Sprintf(format, a...)} }

// Module is a host module serving one WIT interface: wasi:webgpu, or one
// of wasi-gfx:surface's. The modules of one Backend share, per process,
// one handle table and one session, so a texture a surface context
// returns is a wasi:webgpu resource like any other.
type Module struct {
	Backend Backend
	Limits  Limits
	iface   *wit.Interface
	exports []export
	shared  *sharedState
}

// NewModule serves wasi:webgpu with a backend (nil: no GPU anywhere).
func NewModule(b Backend) *Module { return NewModules(b)[0] }

// NewModules serves wasi:webgpu and wasi-gfx:surface (surface,
// surface-webgpu) with one backend: register all three.
func NewModules(b Backend) []*Module {
	if b == nil {
		b = noGPU{}
	}
	sh := &sharedState{procs: map[*wasi.Proc]*procState{}}
	var out []*Module
	for _, i := range []*wit.Interface{Interface, SurfaceInterfaces["surface"], SurfaceInterfaces["surface-webgpu"]} {
		m := &Module{Backend: b, Limits: DefaultLimits, iface: i, shared: sh}
		for _, f := range i.Funcs {
			m.exports = append(m.exports, export{name: f.ImportName(""), fn: f})
		}
		for _, r := range i.ResourceNames() {
			for _, f := range i.Resources[r].Funcs {
				m.exports = append(m.exports, export{name: f.ImportName(r), resource: r, fn: f})
			}
			m.exports = append(m.exports, export{name: "[resource-drop]" + r, resource: r, drop: true})
		}
		out = append(out, m)
	}
	return out
}

// HostModules is NewModules as the engine's host-module list.
func HostModules(b Backend) []wasi.HostModule {
	var out []wasi.HostModule
	for _, m := range NewModules(b) {
		out = append(out, m)
	}
	return out
}

// sharedState is a backend's per-process state, shared by its modules.
type sharedState struct {
	mu    sync.Mutex
	procs map[*wasi.Proc]*procState
}

type procState struct {
	h    *Handles
	s    Session
	refs int
}

func (sh *sharedState) acquire(p *wasi.Proc, m *Module) *procState {
	sh.mu.Lock()
	defer sh.mu.Unlock()
	ps := sh.procs[p]
	if ps == nil {
		h := newHandles(m.Limits.Handles)
		ps = &procState{h: h, s: m.Backend.Open(Process{GPU: p.HasCap("gpu"), Container: p.Label()}, m.Limits, h)}
		sh.procs[p] = ps
	}
	ps.refs++
	return ps
}

func (sh *sharedState) release(p *wasi.Proc, ps *procState) {
	sh.mu.Lock()
	ps.refs--
	last := ps.refs == 0
	if last {
		delete(sh.procs, p)
	}
	sh.mu.Unlock()
	if last {
		ps.s.Close()
		ps.h.closeAll()
	}
}

// Module implements wasi.HostModule.
func (m *Module) Module() string { return m.iface.ImportModule() }

// Funcs implements wasi.HostModule: every function of the interface, with
// its Canonical ABI core signature, and a drop per resource.
func (m *Module) Funcs() []wasi.HostFunc {
	out := make([]wasi.HostFunc, len(m.exports))
	for i, e := range m.exports {
		if e.drop {
			out[i] = wasi.HostFunc{Name: e.name, Params: "i"}
			continue
		}
		s := wit.Sig(e.fn)
		out[i] = wasi.HostFunc{Name: e.name, Params: string(s.Params), Result: string(s.Results)}
	}
	return out
}

// Open implements wasi.HostModule.
func (m *Module) Open(p *wasi.Proc) wasi.Instance {
	ps := m.shared.acquire(p, m)
	return &instance{m: m, h: ps.h, s: ps.s, p: p, ps: ps}
}

type instance struct {
	m    *Module
	h    *Handles
	s    Session
	p    *wasi.Proc
	ps   *procState
	once sync.Once
}

func (in *instance) Close() {
	in.once.Do(func() { in.m.shared.release(in.p, in.ps) })
}

func trapResult(err error) wasi.Result { return wasi.Result{Trap: err.Error()} }

func (in *instance) Call(fn int, args []uint64, m wasi.Memory) wasi.Result {
	if fn < 0 || fn >= len(in.m.exports) {
		return wasi.Result{Trap: "wasi:webgpu: no such function"}
	}
	e := in.m.exports[fn]
	if e.drop {
		if err := in.h.Drop(uint32(args[0]), e.resource); err != nil {
			return trapResult(err)
		}
		return wasi.Result{}
	}
	vals, err := wit.LiftParams(e.fn, args, m)
	if err != nil {
		return trapResult(err)
	}
	finish := func(v any, err error, m wasi.Memory) wasi.Result {
		if err != nil {
			return trapResult(err)
		}
		return in.lowerResult(e.fn, v, args, m)
	}
	if in.s.Async(e.name) {
		done := make(chan struct{})
		var v any
		go func() {
			v, err = in.s.Call(e.name, vals)
			close(done)
		}()
		return wasi.Result{Wait: done, Resume: func(m wasi.Memory) wasi.Result { return finish(v, err, m) }}
	}
	v, err := in.s.Call(e.name, vals)
	return finish(v, err, m)
}

// lowerResult lowers v, obtaining the guest allocations it needs one by
// one through Result.Realloc.
func (in *instance) lowerResult(f *wit.Func, v any, args []uint64, m wasi.Memory) wasi.Result {
	l, err := wit.PlanResult(f, v, args)
	if err != nil {
		return trapResult(err)
	}
	ptrs := make([]uint32, 0, len(l.Reqs))
	var step func(m wasi.Memory) wasi.Result
	step = func(m wasi.Memory) wasi.Result {
		if len(ptrs) == len(l.Reqs) {
			ret, err := l.Apply(m, ptrs)
			if err != nil {
				return trapResult(err)
			}
			return wasi.Result{Ret: ret}
		}
		r := l.Reqs[len(ptrs)]
		return wasi.Result{Realloc: &wasi.Realloc{Align: r.Align, Size: r.Size, Then: func(p uint32, m wasi.Memory) wasi.Result {
			ptrs = append(ptrs, p)
			return step(m)
		}}}
	}
	return step(m)
}

// Handles is a process's resource table: handles start at 1 (0 is never
// valid), each tagged with its resource type.
type Handles struct {
	mu    sync.Mutex
	limit int
	next  uint32
	tab   map[uint32]handle
}

type handle struct {
	res string
	obj any
}

// Destroyer is implemented by objects that release something when their
// handle is dropped or the process ends.
type Destroyer interface{ Destroy() }

func newHandles(limit int) *Handles {
	if limit <= 0 {
		limit = DefaultLimits.Handles
	}
	return &Handles{limit: limit, tab: map[uint32]handle{}}
}

// ErrHandles is the quota error of New.
var ErrHandles = errors.New("too many live resources")

// New registers obj as a resource of type res and returns its handle.
func (h *Handles) New(res string, obj any) (uint32, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if len(h.tab) >= h.limit {
		return 0, Trapf("%v (%d)", ErrHandles, h.limit)
	}
	h.next++
	for h.next == 0 || h.tab[h.next].res != "" {
		h.next++
	}
	h.tab[h.next] = handle{res: res, obj: obj}
	return h.next, nil
}

// Get returns handle n's object, which must be a res.
func (h *Handles) Get(n uint32, res string) (any, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	e, ok := h.tab[n]
	if !ok || e.res != res {
		return nil, Trapf("invalid %s handle %d", res, n)
	}
	return e.obj, nil
}

// Drop removes handle n (a res), destroying its object.
func (h *Handles) Drop(n uint32, res string) error {
	h.mu.Lock()
	e, ok := h.tab[n]
	if !ok || e.res != res {
		h.mu.Unlock()
		return Trapf("drop of invalid %s handle %d", res, n)
	}
	delete(h.tab, n)
	h.mu.Unlock()
	if d, ok := e.obj.(Destroyer); ok {
		d.Destroy()
	}
	return nil
}

// Len is the number of live handles.
func (h *Handles) Len() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return len(h.tab)
}

func (h *Handles) closeAll() {
	h.mu.Lock()
	tab := h.tab
	h.tab = map[uint32]handle{}
	h.mu.Unlock()
	for _, e := range tab {
		if d, ok := e.obj.(Destroyer); ok {
			d.Destroy()
		}
	}
}

// noGPU is the backend where there is none: a gpu object whose adapter
// request answers none.
type noGPU struct{}

func (noGPU) Open(_ Process, _ Limits, h *Handles) Session { return &noSession{h: h} }

type noSession struct{ h *Handles }

func (noSession) Async(string) bool { return false }
func (noSession) Close()            {}

func (s *noSession) Call(name string, args []any) (any, error) {
	switch name {
	case "get-gpu":
		return s.h.New("gpu", nil)
	case "[method]gpu.request-adapter":
		if _, err := s.h.Get(args[0].(uint32), "gpu"); err != nil {
			return nil, err
		}
		return nil, nil // option::none: no adapter here
	case "[method]gpu.get-preferred-canvas-format":
		return uint32(0), nil
	case "[method]gpu.wgsl-language-features":
		return s.h.New("wgsl-language-features", nil)
	case "[method]wgsl-language-features.has":
		return false, nil
	}
	return nil, Trapf("%s: no GPU here (no adapter was given)", name)
}
