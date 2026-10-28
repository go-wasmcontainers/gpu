package webgpu

import (
	"encoding/binary"
	"strings"
	"testing"

	"github.com/go-wasmcontainers/engine/rootfs"
	"github.com/go-wasmcontainers/engine/wasi"
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
