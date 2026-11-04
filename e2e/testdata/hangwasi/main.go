//go:build !js

// Command hangwasi is the hostile guest of the loop budget on the wasi:webgpu
// path (the host module with no scheduler): it compiles a compute shader whose
// loop would run for seconds, dispatches it, and reads the result back. With
// the loop budget on, the host cuts the loop and the read-back traps with the
// truncation; the guest never sees a partial result. It prints what it saw and
// how long it took.
package main

import (
	"fmt"
	"os"
	"time"

	"github.com/go-wasmcontainers/gpu/wasigpu"
)

const hostile = `
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  var x = id.x + id.y * 65537u + 1u;
  for (var i = 0u; i < 4000000000u; i = i + 1u) {
    x = x * 1664525u + 1013904223u;
    if (x == 0xFFFFFFFFu) { break; }
  }
  out[id.x %% arrayLength(&out)] = x;
}`

func main() {
	d, err := wasigpu.Open()
	if err != nil {
		fmt.Println("no GPU:", err)
		os.Exit(2)
	}
	size := uint64(4096)
	buf, err := d.Buffer(size, wasigpu.Storage|wasigpu.CopySrc)
	must(err)
	rd, err := d.Buffer(size, wasigpu.MapRead|wasigpu.CopyDst)
	must(err)
	p, err := d.ComputePipeline(fmt.Sprintf(hostile), "main")
	must(err)
	t0 := time.Now()
	must(d.Dispatch(p, []*wasigpu.Buffer{buf}, 4096, 1, 1))
	must(d.Copy(buf, rd, size))
	out, err := rd.Read() // a truncated result traps here, after the cut
	// Not reached with the loop budget on: the host traps the container.
	fmt.Printf("hostile read-back: %d bytes, error %v, after %v: NOT CUT\n", len(out), err, time.Since(t0).Round(time.Millisecond))
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}
