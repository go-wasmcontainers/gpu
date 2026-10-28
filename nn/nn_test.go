package nn

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/go-wasmcontainers/engine/rootfs"
	"github.com/go-wasmcontainers/engine/runtime"
	"github.com/go-wasmcontainers/engine/runtime/wazero"
	"github.com/go-wasmcontainers/engine/wasi"
)

type syncBuf struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (s *syncBuf) Write(p []byte) (int, error) { s.mu.Lock(); defer s.mu.Unlock(); return s.b.Write(p) }
func (s *syncBuf) String() string              { s.mu.Lock(); defer s.mu.Unlock(); return s.b.String() }

func buildExample(t *testing.T) []byte {
	t.Helper()
	out := filepath.Join(t.TempDir(), "mnist.wasm")
	cmd := exec.Command("go", "build", "-o", out, "../examples/mnist")
	cmd.Env = append(os.Environ(), "GOOS=wasip1", "GOARCH=wasm", "CGO_ENABLED=0")
	if b, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build: %v\n%s", err, b)
	}
	b, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// digitLabel is the MNIST test set's label of examples/mnist/digit.png
// (test picture 0, extracted by internal/cmd/mnistpng).
const digitLabel = 7

func runExample(t *testing.T, wasm []byte, caps []string, args ...string) (string, string, runtime.Result) {
	t.Helper()
	fsys := rootfs.New()
	fsys.MkdirAll("/bin", 0o755)
	fsys.WriteFile("/bin/mnist.wasm", wasm, 0o755)
	pics, _ := filepath.Glob("testdata/mnist-*.png")
	for _, p := range pics {
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		fsys.WriteFile("/"+filepath.Base(p), b, 0o644)
	}
	e := wazero.New(wazero.Options{HostModules: []wasi.HostModule{NewModule()}})
	defer e.Close(context.Background())
	var out, errb syncBuf
	p, err := e.Start(context.Background(), runtime.Spec{FS: fsys, Path: "/bin/mnist.wasm", Args: append([]string{"mnist"}, args...),
		Stdout: &out, Stderr: &errb, Caps: caps, MemoryLimitBytes: 256 << 20})
	if err != nil {
		t.Fatal(err)
	}
	res := p.Wait()
	return out.String(), errb.String(), res
}

// TestMNISTExample runs examples/mnist as a container natively: the
// digit it reads must be the reference output's class, on the CPU; a GPU
// target fails plainly, with or without --gpu, since no GPU backend is
// registered natively.
func TestMNISTExample(t *testing.T) {
	if testing.Short() {
		t.Skip("builds a wasip1 guest")
	}
	wasm := buildExample(t)
	out, errOut, res := runExample(t, wasm, nil)
	if want := fmt.Sprintf("digit %d (p=", digitLabel); !strings.HasPrefix(out, want) || res.ExitCode != 0 {
		t.Fatalf("cpu: stdout %q stderr %q %+v, want %q…", out, errOut, res, want)
	}
	t.Logf("embedded picture: %s", strings.TrimSpace(out))
	// Three more pictures of the MNIST test set (pictures 1-3, labels
	// 2, 1, 0), given as a file argument.
	for pic, label := range map[string]int{"/mnist-1.png": 2, "/mnist-2.png": 1, "/mnist-3.png": 0} {
		out, errOut, res := runExample(t, wasm, nil, "cpu", pic)
		if want := fmt.Sprintf("digit %d (p=", label); !strings.HasPrefix(out, want) || res.ExitCode != 0 {
			t.Errorf("%s: stdout %q stderr %q, want %q…", pic, out, errOut, want)
		}
	}
	for _, caps := range [][]string{nil, {"gpu"}} {
		out, errOut, res = runExample(t, wasm, caps, "gpu")
		if res.ExitCode != 1 || !strings.Contains(errOut, "unsupported operation") || out != "" {
			t.Errorf("gpu with caps %v: stdout %q stderr %q %+v", caps, out, errOut, res)
		}
	}
}

// mem is a flat guest memory.
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

func le(m mem, off, v uint32) { binary.LittleEndian.PutUint32(m[off:], v) }

// TestHostCalls drives the module call by call with a test memory: a
// model loaded, an input set, compute, outputs read, and each refusal.
func TestHostCalls(t *testing.T) {
	model, err := os.ReadFile("../onnx/testdata/models/mnist-12/mnist-12.onnx")
	if err != nil {
		t.Fatal(err)
	}
	mod := NewModule()
	mod.Named = map[string]Named{"mnist": {Encoding: ONNX, Builders: [][]byte{model}}}
	in := mod.Open(wasi.New(wasi.Config{FS: rootfs.New()})).(*instance)
	m := make(mem, 1<<20)
	copy(m[100000:], model)
	le(m, 0, 100000)
	le(m, 4, uint32(len(model)))
	call := func(fn int, a ...uint32) uint32 {
		args := make([]uint64, len(a))
		for i, v := range a {
			args[i] = uint64(v)
		}
		return uint32(in.Call(fn, args, m).Ret)
	}
	if e := call(0, 0, 1, ONNX, CPU, 16); e != Success {
		t.Fatalf("load: %d", e)
	}
	g := binary.LittleEndian.Uint32(m[16:])
	for name, c := range map[string]struct {
		got, want uint32
	}{
		"no builders":      {call(0, 0, 0, ONNX, CPU, 16), InvalidArgument},
		"bad pointer":      {call(0, 1<<21, 1, ONNX, CPU, 16), InvalidArgument},
		"other encoding":   {call(0, 0, 1, TensorFlow, CPU, 16), InvalidEncoding},
		"gpu without cap":  {call(0, 0, 1, ONNX, GPU, 16), UnsupportedOperation},
		"bad target":       {call(0, 0, 1, ONNX, 7, 16), InvalidArgument},
		"not a model":      {func() uint32 { le(m, 4, 10); defer le(m, 4, uint32(len(model))); return call(0, 0, 1, ONNX, CPU, 16) }(), InvalidEncoding},
		"unknown name":     {func() uint32 { copy(m[200:], "nope"); return call(1, 200, 4, 16) }(), NotFound},
		"named":            {func() uint32 { copy(m[200:], "mnist"); return call(1, 200, 5, 16) }(), Success},
		"unknown graph":    {call(2, 999, 20), InvalidArgument},
		"unknown context":  {call(5, 999), InvalidArgument},
		"output too early": {func() uint32 { call(2, g, 20); return call(4, binary.LittleEndian.Uint32(m[20:]), 0, 300, 40, 24) }(), InvalidArgument},
	} {
		if c.got != c.want {
			t.Errorf("%s: %d, want %d", name, c.got, c.want)
		}
	}
	if e := call(2, g, 20); e != Success {
		t.Fatal(e)
	}
	ctx := binary.LittleEndian.Uint32(m[20:])
	// The tensor record at 500: dims at 600, data at 700.
	dims := []uint32{1, 1, 28, 28}
	for i, d := range dims {
		le(m, 600+uint32(4*i), d)
	}
	le(m, 500, 600)
	le(m, 504, 4)
	m[508] = TypeF32
	le(m, 512, 700)
	le(m, 516, 28*28*4)
	if e := call(3, ctx, 0, 500); e != Success {
		t.Fatalf("set_input: %d", e)
	}
	if e := call(3, ctx, 5, 500); e != InvalidArgument {
		t.Errorf("set_input index 5: %d", e)
	}
	m[508] = TypeU8
	if e := call(3, ctx, 0, 500); e != UnsupportedOperation {
		t.Errorf("u8 input: %d", e)
	}
	m[508] = TypeF32
	le(m, 516, 28*28*4-4)
	if e := call(3, ctx, 0, 500); e != InvalidArgument {
		t.Errorf("short data: %d", e)
	}
	le(m, 516, 28*28*4)
	le(m, 504, 100)
	if e := call(3, ctx, 0, 500); e != TooLarge {
		t.Errorf("rank 100: %d", e)
	}
	le(m, 504, 4)
	if e := call(5, ctx); e != Success {
		t.Fatalf("compute: %d", e)
	}
	if e := call(4, ctx, 0, 300, 39, 24); e != TooLarge {
		t.Errorf("small buffer: %d", e)
	}
	if e := call(4, ctx, 0, 300, 40, 24); e != Success || binary.LittleEndian.Uint32(m[24:]) != 40 {
		t.Fatalf("get_output: %d", e)
	}
	for i := 0; i < 10; i++ {
		if v := math.Float32frombits(binary.LittleEndian.Uint32(m[300+4*i:])); math.IsNaN(float64(v)) {
			t.Fatal("NaN score")
		}
	}
	// Handle limits.
	for i := 0; i < DefaultLimits.Contexts; i++ {
		call(2, g, 20)
	}
	if e := call(2, g, 20); e != Busy {
		t.Errorf("context limit: %d", e)
	}
	in.Close()
	if r := in.Call(5, []uint64{uint64(ctx)}, m); r.Ret != uint64(RuntimeError) {
		t.Errorf("call after close: %+v", r)
	}
}
