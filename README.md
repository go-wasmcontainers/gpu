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
| `onnx`: ONNX model loader and CPU operators | **done** — see below |
| wasi-nn host module (`wasi_ephemeral_nn`) and guest bindings | next |
| wasi:webgpu compute (browser) | planned |
| rendering to a dashboard pane | planned |

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
