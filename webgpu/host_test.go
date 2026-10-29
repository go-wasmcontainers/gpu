package webgpu

import (
	"encoding/binary"
	"strings"
	"testing"

	"github.com/go-wasmcontainers/engine/rootfs"
	"github.com/go-wasmcontainers/engine/wasi"
	"github.com/go-wasmcontainers/gpu/wit"
)

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

func index(t *testing.T, m *Module, name string) int {
	t.Helper()
	for i, e := range m.exports {
		if e.name == name {
			return i
		}
	}
	t.Fatalf("no export %s", name)
	return -1
}

// TestModuleShape: every function of the interface is exported once,
// with a valid core signature, and the module's imports pass the
// engine's validation.
func TestModuleShape(t *testing.T) {
	m := NewModule(nil)
	if m.Module() != "wasi:webgpu/webgpu@0.3.0-rc.2" {
		t.Errorf("module %q", m.Module())
	}
	if _, err := wasi.Imports([]wasi.HostModule{m}); err != nil {
		t.Fatal(err)
	}
	n := len(Interface.Funcs) + len(Interface.Resources)
	for _, r := range Interface.Resources {
		n += len(r.Funcs)
		for _, f := range r.Funcs {
			if f.Result != nil && f.Result.Kind == wit.Stream {
				n += 5 // its stream built-ins
			}
		}
	}
	if len(m.Funcs()) != n || n < 200 {
		t.Errorf("%d functions exported, want %d", len(m.Funcs()), n)
	}
}

// TestNoGPU: without a backend, get-gpu works, request-adapter answers
// none, and anything needing a device traps; handles are checked.
func TestNoGPU(t *testing.T) {
	m := NewModule(nil)
	in := m.Open(wasi.New(wasi.Config{FS: rootfs.New(), Caps: []string{"gpu"}}))
	defer in.Close()
	mm := make(mem, 1024)
	r := in.Call(index(t, m, "get-gpu"), nil, mm)
	if r.Trap != "" || r.Ret == 0 {
		t.Fatalf("get-gpu: %+v", r)
	}
	gpu := r.Ret
	mm[100] = 0xff
	r = in.Call(index(t, m, "[method]gpu.request-adapter"), []uint64{gpu, 0, 0, 0, 0, 100}, mm)
	_ = r
	// request-adapter(self, options: option<record>) -> option<own>: the
	// options flatten to several values, the result through a return area
	// (the last argument).
	sig := m.Funcs()[index(t, m, "[method]gpu.request-adapter")]
	args := make([]uint64, len(sig.Params))
	args[0] = gpu
	args[len(args)-1] = 100
	if r = in.Call(index(t, m, "[method]gpu.request-adapter"), args, mm); r.Trap != "" || mm[100] != 0 {
		t.Fatalf("request-adapter: %+v, discriminant %d", r, mm[100])
	}
	args[0] = 999
	if r = in.Call(index(t, m, "[method]gpu.request-adapter"), args, mm); !strings.Contains(r.Trap, "invalid gpu handle") {
		t.Errorf("forged handle: %+v", r)
	}
	if r = in.Call(index(t, m, "[resource-drop]gpu"), []uint64{gpu}, mm); r.Trap != "" {
		t.Errorf("drop: %+v", r)
	}
	if r = in.Call(index(t, m, "[resource-drop]gpu"), []uint64{gpu}, mm); r.Trap == "" {
		t.Error("double drop accepted")
	}
	if r = in.Call(index(t, m, "[method]gpu-device.queue"), []uint64{1}, mm); r.Trap == "" {
		t.Error("a device call without a GPU did not trap")
	}
	_ = binary.LittleEndian
}

// TestSurfaceModules: the surface interfaces resolve their uses of
// wasi:webgpu (a context's texture is a wasi:webgpu resource), the three
// modules validate together, and they share one handle table per process.
func TestSurfaceModules(t *testing.T) {
	ctx := SurfaceInterfaces["surface-webgpu"].Resources["context"]
	var tex *wit.Func
	for _, f := range ctx.Funcs {
		if f.Name == "get-current-texture" {
			tex = f
		}
	}
	if tex == nil || tex.Result.Kind != wit.Own || tex.Result.Name != "gpu-texture" || tex.Result.Owner != "wasi:webgpu/webgpu@0.3.0-rc.2" {
		t.Fatalf("get-current-texture result %+v", tex)
	}
	if c := ctx.Funcs[0]; c.Kind != wit.Constructor || c.Params[0].Type.Kind != wit.Borrow || c.Params[0].Type.Owner != "wasi-gfx:surface/surface@0.2.0" {
		t.Errorf("context constructor %+v", c.Params[0].Type)
	}
	mods := HostModules(nil)
	if _, err := wasi.Imports(mods); err != nil {
		t.Fatal(err)
	}
	p := wasi.New(wasi.Config{FS: rootfs.New()})
	a, b := mods[0].Open(p).(*instance), mods[2].Open(p).(*instance)
	if a.h != b.h {
		t.Error("the modules of one process do not share a handle table")
	}
	a.Close()
	if _, err := b.h.New("gpu", nil); err != nil {
		t.Error("the table closed while a module still used it")
	}
	b.Close()
	if b.h.Len() != 0 {
		t.Error("the table outlived its last module")
	}
}

type recordBackend struct{ got *[]Process }

func (r recordBackend) Open(p Process, l Limits, h *Handles) Session {
	*r.got = append(*r.got, p)
	return noGPU{}.Open(p, l, h)
}

// TestBackendProcess: a backend opens once per process, with the gpu
// capability and the container label the engine set.
func TestBackendProcess(t *testing.T) {
	var got []Process
	mods := HostModules(recordBackend{&got})
	p := wasi.New(wasi.Config{FS: rootfs.New(), Caps: []string{"gpu"}, Label: "c1"})
	for _, m := range mods {
		defer m.Open(p).Close()
	}
	if len(got) != 1 || got[0] != (Process{GPU: true, Container: "c1"}) {
		t.Errorf("backend opened with %+v", got)
	}
}

func TestHandles(t *testing.T) {
	h := newHandles(2)
	a, _ := h.New("gpu", nil)
	b, _ := h.New("gpu-buffer", nil)
	if _, err := h.New("gpu", nil); err == nil {
		t.Error("handle quota not enforced")
	}
	if _, err := h.Get(a, "gpu-buffer"); err == nil {
		t.Error("a gpu handle passed as a buffer")
	}
	if _, err := h.Get(b, "gpu-buffer"); err != nil {
		t.Error(err)
	}
	if err := h.Drop(0, "gpu"); err == nil {
		t.Error("handle 0 dropped")
	}
}
