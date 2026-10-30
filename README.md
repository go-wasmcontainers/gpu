# gpu

[![CI](https://github.com/go-wasmcontainers/gpu/actions/workflows/ci.yml/badge.svg)](https://github.com/go-wasmcontainers/gpu/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/go-wasmcontainers/gpu.svg)](https://pkg.go.dev/github.com/go-wasmcontainers/gpu)

GPU access for [go-wasmcontainers](https://go-wasmcontainers.github.io/)
WASI containers: **wasi-nn** inference, **wasi:webgpu** compute, and
rendering, on WebGPU in the browser (the playground) and on the CPU
natively. Pure Go, `CGO_ENABLED=0`.

Design and references: [docs/DESIGN.md](docs/DESIGN.md).

## Status

| Part | State |
|---|---|
| `onnx`: ONNX model loader and CPU operators | **done** (v0.1.0) — see below |
| `nn`: wasi-nn host module (`wasi_ephemeral_nn`), CPU backend | **done** (v0.2.0) — see below |
| `wasinn`: wasi-nn guest bindings for Go, `examples/mnist` | **done** (v0.2.0) |
| `nn/webgpu`: wasi-nn GPU backend (WebGPU WGSL kernels, browser) | **done** (v0.3.0) — see below |
| `wit`, `webgpu`, `wasigpu`: wasi:webgpu compute (host and Go guest), `examples/compute` | **done** (v0.4.0) — see below |
| rendering: `wasi-gfx:surface` + `surface-webgpu`, frames to the page, `examples/triangle` | **done** (v0.5.0) — see below; the dashboard pane is the dashboard's part |
| surface input: pointer and key streams (`stream<…>` and the Canonical ABI's stream and waitable-set built-ins) | **done** — see below; forwarding events from the pane is the dashboard's part |
| `on-resize` / `on-frame` streams, vertex buffers, depth-stencil | not yet |

## wasi:webgpu: `wit`, `webgpu` (host), `wasigpu` (guest)

Containers are WASI preview 1 core modules, and the engine has no
component-model runtime; `wasi:webgpu@0.3.0-rc.2` is WIT. So:

- `wit` parses WIT and implements the **Canonical ABI** for a 32-bit
  memory (alignment, sizes, flattening, loads, stores, flat lifting and
  lowering; traps as errors), with the guest allocations of a result
  planned first and applied once the guest's `cabi_realloc` gave them
  (`FuzzLoad`).
- `webgpu.NewModule(backend)` is a host module serving **every function
  of the interface** (core names such as `[method]gpu-device.create-buffer`,
  signatures from the WIT), resources as per-process handles. Asynchronous
  functions (`request-adapter`, `map-async`, …) are lowered synchronously
  and run off the host call. `webgpu.Browser{}` maps them to WebGPU in the
  engine's worker through [go-gfx/webgpu](https://github.com/go-gfx/webgpu),
  the fleet's shared WebGPU library for `GOOS=js` (the wasi-nn GPU backend
  runs on it too): arguments are range checked before they reach the
  browser, and each process's memory, pipeline, WGSL and surface limits are
  a `webgpu.Quota` the library enforces. Elsewhere there is no backend: `get-gpu` works and
  `request-adapter` answers none. A process without `--gpu` gets no
  adapter. Quotas per process: 4096 handles, 256 MiB of buffers, 1 MiB per
  WGSL module, 256 pipelines (traps past them); a submission still running
  after 2 s has its device destroyed; fallback adapters are refused.
- `wasigpu` is the Go guest side (`GOOS=wasip1`), over the same `wit`
  code; its imports are generated from the WIT by `internal/cmd/witgen`
  (a test keeps them current).

[`examples/compute`](examples/compute) adds two 1M-float vectors and
multiplies two 256×256 matrices on the GPU, checking every element against
the CPU. `e2e/engine_test.go` runs it — and `examples/mnist` on the CPU and
on the GPU — as **real containers in the engine's browser executor**
(Chromium, a Web Worker per container): on an Apple Metal adapter every
element matches; matmul 15 ms on the GPU vs 42 ms on the guest's CPU, the
vector add 69 ms vs 3.5 ms (dominated by transfers; one run, not a
benchmark). Without `--gpu` the guest is told there is no adapter, and a
fifth 64 MiB buffer traps on the memory quota.

## Rendering: surfaces

`webgpu.HostModules(backend)` serves four modules sharing one handle table
per process: `wasi:webgpu/webgpu@0.3.0-rc.2`,
`wasi-gfx:surface/surface@0.2.0`, `wasi-gfx:surface/surface-webgpu@0.2.0`
and `$root` (the waitable-set built-ins; a guest using `wasigpu` imports
all four). In the browser each surface
is an `OffscreenCanvas` in the engine's worker; `present` turns it into an
`ImageBitmap` handed to `Browser.OnSurface` with a sequence number (the
`open` event names the container), for the
page to draw (in the playground, a dashboard pane). The page acknowledges
each frame (`webgpu.AckFrame`); at most 2 frames are unacknowledged
(past that, `present` waits for an acknowledgement at most 1 s, so a page
that stops acknowledging slows the guest to a frame per second without
stopping it), frames are paced to 60
per second, surfaces are at most 1920×1080. Render pipelines take their
vertices from the shader (no vertex buffers yet) and have no depth-stencil.

**Input.** A surface's `on-pointer-down`, `on-pointer-up`,
`on-pointer-move`, `on-key-down` and `on-key-up` each return a
`stream<pointer-event>` or `stream<key-event>` (UI Events key codes, the
text the key produces, modifiers). A core module reads it with the
Canonical ABI built-ins as wit-component names them:
`[stream-read-0]…` (blocks until an event), `[async-lower][stream-read-0]…`
(returns BLOCKED, completes through a waitable set), cancel-read,
drop-readable, and `$root`'s `[waitable-set-new|poll|wait|drop]` and
`[waitable-join]`. The page reports events on a pane with
`webgpu.SurfaceInput(surface, InputEvent)`; each stream keeps at most 256
undelivered events (the oldest dropped), a surface serves at most 64
streams, coordinates must be finite and key text at most 64 bytes of
UTF-8. Guest side: `(*Surface).Pointer(kind)` and `(*Surface).Keys(kind)`,
whose `Poll` returns what arrived without blocking (once per frame).

Guest side: `(*wasigpu.Device).Surface(w, h)`,
`(*Surface).RenderPipeline(wgsl, vs, fs)` and `(*Surface).Frame(pipeline,
uniforms, clear, vertices)`. [`examples/triangle`](examples/triangle) draws
a spinning triangle; `e2e/engine_test.go` runs it as a container on an Apple
Metal adapter and checks, at the page, 60 frames in order, the clear colour
in a corner, the triangle in the middle, frames that change, and a rate
within the 60 per second pacing (one run: 51 frames per second); with
`input`, the page clicks and types on the pane and the guest prints what it
read (`pointer-down 10,20`, `key-down key-a "A" shift=true`).
The same checks, pixels included, pass on the Linux SwiftShader lane.
**Headless Linux Chromium** can back a WebGPU canvas only with Skia on
Vulkan: launch it with `--enable-features=Vulkan --use-vulkan=swiftshader
--use-angle=swiftshader` (or a real Vulkan driver), otherwise the WebGPU
instance is torn down at the first canvas configure and frames are blank
(docs/DESIGN.md, "Surfaces"). `go run ./internal/cmd/gpuprobe --canvas`
follows one canvas frame from the GPU to the page to check a browser.

## wasi-nn: `nn` (host) and `wasinn` (guest)

`nn.NewModule()` is a host module for the engine (v0.13.0+,
`wazero.Options.HostModules` / `browser.Options.HostModules`) serving the
preview 1 wasi-nn interface, `wasi_ephemeral_nn` (`load`, `load_by_name`,
`init_execution_context`, `set_input`, `get_output`, `compute`), as
Wasmtime and the `wasi-nn` crate's bindings lower it. A graph goes to the
backend of its encoding (ONNX: the pure-Go CPU backend). The execution
target is checked: `gpu` needs the container's `gpu` capability
(`wctr run --gpu`) **and** a GPU backend, and otherwise fails with
`unsupported_operation` — never runs on the CPU in its place. Model bytes,
graphs and contexts per process, tensor rank and data are bounded;
`FuzzHostCalls` drives arbitrary call sequences over arbitrary memory.

`wasinn` gives Go guests (`GOOS=wasip1`) `Load`, `NewContext`,
`SetInputF32`, `Compute`, `OutputF32`; elsewhere it builds and returns
`ErrUnavailable`.

[`examples/mnist`](examples/mnist) is a container that reads a handwritten
digit: natively it classifies MNIST test pictures 0–3 (labels 7, 2, 1, 0)
correctly, p = 1.000 for the embedded one.

## wasi-nn on the GPU: `nn/webgpu`

In the browser, `webgpu.NewModule()` adds a "gpu" backend to the CPU one:
each ONNX node becomes one WGSL compute dispatch on a WebGPU device of the
graph's own (in the engine's worker), shapes and constants coming from
`onnx.Session.Plan` (a CPU dry run, so both executors agree on every shape
by construction). Its calls wait on WebGPU promises, so `nn` runs them off
the host call (`AsyncBackend`). A fallback (software) adapter is refused,
and each graph's GPU memory is bounded (256 MiB).

Checked twice: natively, `TestKernelsEmulated` runs each kernel's index
arithmetic in Go on all 122 vendored ONNX node tests (no GPU needed, every
platform); in a browser, `e2e/` runs the real WGSL on the same tests and
MNIST-12 through Playwright. Measured on an Apple Metal adapter (Chromium
143, new headless): **123/123 pass**, MNIST-12 within 3.1e-5 of the
reference. The CI lanes: macOS (must get a GPU adapter, or fail) and Linux
on SwiftShader (labelled fallback: proves the WGSL, not a GPU).
`internal/cmd/gpuprobe` reports what adapter a browser offers.

## `onnx`

Loads an ONNX model (`onnx.proto`, decoded by a bounded protobuf reader)
and runs it on the CPU. Operators: Add, AveragePool, Concat, Constant,
Conv, Div, Dropout (inference), Flatten, Gemm, Identity, MatMul, MaxPool,
Mul, Relu, Reshape, Softmax, Sub, Transpose — default domain, operator
sets 7 and later, float tensors (int64 for shapes).

- **Refuses, precisely.** A model using anything else fails at load with
  every unsupported operator and its node count, e.g.
  `onnx: unsupported operators: LSTM (2 nodes), com.example:Foo (1 node); supported: Add, …`.
  Unsupported variants (MaxPool indices, Dropout training mode, non-float
  graph values) are refused at load too, not mid-inference.
- **Checked op by op** against the ONNX node tests (onnx v1.22.0,
  `onnx/backend/test/data/node`): the 122 tests whose models load here are
  vendored by `internal/cmd/onnxvendor` (a derived list, not a hand-written
  one) and pass with the ONNX backend's tolerances, `|out − ref| ≤ 1e-7 +
  1e-3·|ref|`. MNIST-12 from the ONNX model zoo passes on its own test data.
- **Untrusted input.** The model size, node count, rank, elements per
  tensor and in total, string lengths, and the multiply-accumulates of one
  inference are bounded (`Limits`); `FuzzRun` (decode, load, run) and
  `FuzzDecodeTensor` run in CI.

Performance has not been measured yet (the machine's load was above 40);
convolutions are direct loops over precomputed taps, not im2col + GEMM.

## License

Apache-2.0 (LICENSE). Test data: ONNX node tests, Apache-2.0
(onnx/testdata/node/LICENSE); MNIST-12, MIT (onnx/testdata/models).
