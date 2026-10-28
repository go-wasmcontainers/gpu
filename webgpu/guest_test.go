package webgpu

import (
	"bytes"
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"testing"

	"github.com/go-wasmcontainers/engine/rootfs"
	"github.com/go-wasmcontainers/engine/runtime"
	"github.com/go-wasmcontainers/engine/runtime/wazero"
)

type syncBuf struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (s *syncBuf) Write(p []byte) (int, error) { s.mu.Lock(); defer s.mu.Unlock(); return s.b.Write(p) }
func (s *syncBuf) String() string              { s.mu.Lock(); defer s.mu.Unlock(); return s.b.String() }

// TestComputeExampleWithoutGPU runs examples/compute natively, where no
// backend gives an adapter: the guest's generated imports and its wit
// lowering meet the host's (get-gpu, request-adapter with a return area,
// resource drops) and it reports "no GPU adapter", exit 2.
func TestComputeExampleWithoutGPU(t *testing.T) {
	if testing.Short() {
		t.Skip("builds a wasip1 guest")
	}
	out := filepath.Join(t.TempDir(), "compute.wasm")
	cmd := exec.Command("go", "build", "-o", out, "../examples/compute")
	cmd.Env = append(os.Environ(), "GOOS=wasip1", "GOARCH=wasm", "CGO_ENABLED=0")
	if b, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build: %v\n%s", err, b)
	}
	wasm, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	fsys := rootfs.New()
	fsys.MkdirAll("/bin", 0o755)
	fsys.WriteFile("/bin/compute.wasm", wasm, 0o755)
	e := wazero.New(wazero.Options{HostModules: HostModules(nil)})
	defer e.Close(context.Background())
	var so, se syncBuf
	p, err := e.Start(context.Background(), runtime.Spec{FS: fsys, Path: "/bin/compute.wasm", Stdout: &so, Stderr: &se, Caps: []string{"gpu"}})
	if err != nil {
		t.Fatal(err)
	}
	res := p.Wait()
	if res.ExitCode != 2 || so.String() != "no GPU adapter\n" {
		t.Fatalf("exit %d (%s), stdout %q, stderr %q", res.ExitCode, res.Error, so.String(), se.String())
	}
}
