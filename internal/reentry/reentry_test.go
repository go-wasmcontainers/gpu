package reentry

import (
	"bytes"
	"context"
	"encoding/binary"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tetratelabs/wazero"
	"github.com/tetratelabs/wazero/api"
	"github.com/tetratelabs/wazero/imports/wasi_snapshot_preview1"
)

// buildGuest compiles testdata/guest for wasip1 with the toolchain running
// the test.
func buildGuest(t *testing.T) []byte {
	t.Helper()
	out := filepath.Join(t.TempDir(), "guest.wasm")
	cmd := exec.Command("go", "build", "-o", out, "./testdata/guest")
	cmd.Env = append(os.Environ(), "GOOS=wasip1", "GOARCH=wasm", "CGO_ENABLED=0")
	if b, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("building the guest: %v\n%s", err, b)
	}
	wasm, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	return wasm
}

// TestReallocDuringImport is R1: the host function "give" calls the guest's
// cabi_realloc while the guest is blocked in it, 2000 times, with garbage
// collections in between, and the guest checks every byte.
func TestReallocDuringImport(t *testing.T) {
	if testing.Short() {
		t.Skip("builds a wasip1 guest")
	}
	wasm := buildGuest(t)
	ctx := context.Background()
	r := wazero.NewRuntime(ctx)
	defer r.Close(ctx)
	wasi_snapshot_preview1.MustInstantiate(ctx, r)

	calls := 0
	var hostErr error
	_, err := r.NewHostModuleBuilder("probe").NewFunctionBuilder().
		WithGoModuleFunction(api.GoModuleFunc(func(ctx context.Context, mod api.Module, stack []uint64) {
			n, ret := uint32(stack[0]), uint32(stack[1])
			p, err := Realloc(ctx, mod, 1, n)
			if err != nil {
				hostErr = err
				panic(err)
			}
			b := make([]byte, n)
			for j := range b {
				b[j] = byte(j*7 + calls)
			}
			var rec [8]byte
			binary.LittleEndian.PutUint32(rec[0:], p)
			binary.LittleEndian.PutUint32(rec[4:], n)
			if !mod.Memory().Write(p, b) || !mod.Memory().Write(ret, rec[:]) {
				hostErr = os.ErrInvalid
				panic(hostErr)
			}
			calls++
		}), []api.ValueType{api.ValueTypeI32, api.ValueTypeI32}, nil).
		Export("give").Instantiate(ctx)
	if err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	cfg := wazero.NewModuleConfig().WithStdout(&stdout).WithStderr(&stderr).WithName("guest")
	_, err = r.InstantiateWithConfig(ctx, wasm, cfg)
	if hostErr != nil {
		t.Fatalf("host: %v", hostErr)
	}
	if err != nil {
		t.Fatalf("guest: %v\nstdout: %s\nstderr: %s", err, stdout.String(), stderr.String())
	}
	if got := strings.TrimSpace(stdout.String()); got != "ok 2000" {
		t.Fatalf("guest said %q (stderr %q)", got, stderr.String())
	}
	if calls != 2000 {
		t.Fatalf("%d host calls, want 2000", calls)
	}
}

// TestReallocMissing: a guest without the export is refused, not called.
func TestReallocMissing(t *testing.T) {
	ctx := context.Background()
	r := wazero.NewRuntime(ctx)
	defer r.Close(ctx)
	// The smallest module with a memory and no exports but the memory.
	wasm := []byte{0, 'a', 's', 'm', 1, 0, 0, 0, 5, 3, 1, 0, 1, 7, 10, 1, 6, 'm', 'e', 'm', 'o', 'r', 'y', 2, 0}
	mod, err := r.Instantiate(ctx, wasm)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Realloc(ctx, mod, 1, 8); err != ErrNoRealloc {
		t.Fatalf("got %v, want ErrNoRealloc", err)
	}
}
