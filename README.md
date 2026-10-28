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
| `nn/webgpu`: wasi-nn GPU backend (WebGPU WGSL kernels, browser) | **done** — see below |
| wasi:webgpu compute (browser) | next |
| rendering to a dashboard pane | planned |

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
