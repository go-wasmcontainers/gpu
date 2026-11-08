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
| `gpubatch`, `wasigpu/batchsink`: the batched command stream (`wasi_webgpu_batch`, one host call per submit, read-back, `--gpu-memory`), `examples/batchcompute` | **done** — see below |
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

## Batched GPU commands: `gpubatch` (host) and `wasigpu/batchsink` (guest)

`wasi:webgpu` costs one host call per WebGPU call. The batched path costs
**one host call per submit**: a guest records go-gfx/webgpu work with a remote
device, the sink packs creations, destructions and the recorded stream into a
[go-virtio/gpu `wgpubatch`](https://github.com/go-virtio/gpu) frame, and the
host replays it on one WebGPU device per container, then fills the read window
(`Recorder.ReadBuffer`) before returning. The design and the plan are in
[docs/llm-design.md](docs/llm-design.md) (small language models inside a
container); this is its blocks M3 to M6.

- **Host module** `wasi_webgpu_batch`, `gpubatch.HostModules(gpubatch.Browser{})`:
  `caps(buf, len)` writes the capset (version 2: stream version, `max_frame`
  32 MiB, `max_objects` 4096, the GPU memory grant, `shader-f16` and
  `subgroups` when the adapter has them, the device limits),
  `context_create(out)` (at most 4 contexts, one device per container),
  `submit(ctx, frame, frame_len, read, read_len)` and `context_destroy(ctx)`,
  each answering an errno. The frame is copied out of guest memory before it
  is decoded; guest ids are private to the context and remapped
  (`go-gfx/webgpu/batch`); descriptors, streams and the buffer ranges
  (`wire.CheckRanges`: writes, copies, indirect arguments, dispatch sizes) are
  checked **before** WebGPU sees anything; a refused frame loses its context
  and releases what it held; the guest memory is written only inside the
  `read` window. GPU time is scheduled and budgeted
  ([docs/gpu-scheduling.md](docs/gpu-scheduling.md)): one submit in flight
  across all containers, served round-robin; each submit has a time budget
  from the moment it gets its turn (250 ms for a frame that only streams
  commands, 2 s for a frame that creates objects), and a container whose
  submit overruns has its device destroyed and its GPU grant revoked for the
  rest of its life (*timeout* once, then *revoked*; the engine reports a
  `gpu-sanction` event and `wctr inspect` shows `State.Sanction`); a compute
  quota (`wire.CheckInvocations`) refuses oversized dispatches up front; and a
**loop budget** (layer 4) rewrites every guest compute shader so that a loop
that would run for seconds is cut after `2^36 / invocations` iterations
(go-gfx/wgsl `transform.LoopBudget`, parse only; a truncated submit answers
errno 14 *truncated*, the grant is revoked, and the hostile shader's
victim waits tens of milliseconds, not 3 to 4.5 s). Nothing unbounded runs:
a module the host cannot bound (WGSL its parser does not accept, a loop in a
module with a vertex or fragment entry point, a rewrite the browser refuses)
is refused with errno 15 *unbounded*. Bind group 3 is the host's: a guest
that uses it gets errno 13 *reserved*. The `wasi:webgpu` module and the wasi-nn
GPU kernels go through the same guard. A submit that never completes
because the browser destroyed another container's device (Safari does) is
not an overrun: the device is reset, the guest answers *context lost* and
recreates its context, nobody is sanctioned. Without
  `--gpu` every function answers *denied*; without a GPU backend (native) or
  with only a software adapter, *unsupported*. `FuzzBatchCalls` drives
  arbitrary calls over arbitrary guest memory.
  **Invalid objects are loud**: WGSL that does not compile (a reserved
  word such as `meta`, a syntax error), or a layout, pipeline or bind group
  WebGPU rejects, fails the submit that carries it with errno **11,
  *invalid shader*** and loses the context; `last_error(ctx, buf, len, out)`
  (a fifth import) copies the context's last failure message (at most 1 KiB,
  UTF-8, the browser's WGSL line and column included) to `buf` and stores its
  full length at `out`. `batchsink.Error.Msg` carries it, so the guest sees
  `submit: invalid shader or object (the GPU refused it): ... :1:42 error:
  'meta' is a reserved keyword`. `examples/batchcompute shaders` runs a
  reserved word, a syntax error and a valid control in a container
  (e2e); go-gfx/webgpu v0.9.0 does the scoping.
- **GPU memory**: `wctr run --gpu --gpu-memory=SIZE` (default 256 MiB, at most
  1.5 GiB, refused without `--gpu`; the grant reaches the module as the
  capability `gpu-memory=BYTES`, engine v0.32.0) bounds the buffers and
  textures of all the container's contexts together.
- **Guest side**, `wasigpu/batchsink` (`GOOS=wasip1`):
  `s, _ := batchsink.Open(); dev := s.Device("name")` is a `webgpu.Device`
  whose submits cross once. Elsewhere `Open` returns `ErrUnavailable`. It
  lives here, not in [`llm`](https://github.com/go-wasmcontainers/llm), because
  it is the guest half of this repository's host module (one ABI, versioned
  and released together, tested against it) and any Go guest can use it;
  `llm` (the inference code) imports it.

[`examples/batchcompute`](examples/batchcompute) records a compute shader that
writes known values, copies them to a `MAP_READ` buffer and reads 1 MiB and
4 KiB back in one crossing each; `e2e/batch_test.go` runs it as **real
containers in the engine's browser executor** (Chromium on the Apple GPU, no
fallback adapter): every element matches. The same run checks the refusals:
no `--gpu`; an out-of-range write, a read outside the window, a dispatch
above the limit, an id of another context, a bad-magic frame, a truncated
frame (all *refused*), a frame above `max_frame` (*too large*), a buffer past
the quota (*quota exceeded*); a smaller `--gpu-memory`; and a forced
software adapter (*unsupported*).

**Cost of the guest crossing** (`e2e/batch_test.go` `TestBatchRoundTrip`,
Apple M4 Max, Chromium on Metal, 1-minute load 6.4 to 7.3, medians of 30
interleaved rounds, three runs): a read-back submit takes 315 to 375 µs at
4 KiB and 895 to 1 105 µs at 1 MiB from the guest, against 124 and 155 µs
host-only (go-gfx/webgpu `docs/readback.md`); compute pass + copy + read takes
735 to 800 µs and 1 160 to 1 380 µs, against 270 and 320 µs. The crossing adds
about 200 µs at 4 KiB and 750 to 950 µs at 1 MiB (the window and frame copies);
one crossing per submit is asserted (120 crossings for 120 submits).

### An LLM in a container

[go-wasmcontainers/llm](https://github.com/go-wasmcontainers/llm) runs
SmolLM2-135M (Q4_0) in a container over this module: weights from a
read-only `/models` mount (engine v0.33.0, `wctr run --model`), kernels in
f32 WGSL, one submit per token. The browser tests are here
(`LLM_MODELS=DIR LLM_REPO=CLONE go test -tags e2e ./e2e -run TestLLM`, they
skip without both): `TestLLMOracle` (GPU tokens against llama.cpp's, GPU
logits against the Go CPU path), `TestLLMShaders` (every kernel compiled in
the browser), `TestLLMBench` (interleaved with `llama-bench -t 1`, load at
most 8). Measured on the Apple M4 Max in Chromium: decode 171 tok/s and
prefill 311 tok/s, against 529 and 1286 for llama.cpp on one core (open
defect, in `docs/llm-design.md`). Firefox is a known-slow target for a
token-per-submit loop (about 200 ms per submit, gpu#25: about 5 tok/s); it
has not been measured with this container. An invalid WGSL source is not
reported by `CreateShaderModule` through `wasi_webgpu_batch`: validate
kernels with `TestLLMShaders`.

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
`internal/cmd/gpuprobe` reports what adapter a browser offers: vendor,
architecture, subgroup sizes, features and every numeric limit, for the
`--power high-performance|low-power` preference, and with `--executable` it
drives a system browser instead of Playwright's.

### Browser support

| Browser | Platform | WebGPU | e2e (batch module, ONNX tests, MNIST, refusals, 2 s deadline) |
|---|---|---|---|
| Chromium 151 | macOS, Apple M4 Max (Metal) | yes | passes (123/123 ONNX cases) |
| Chromium 157 (snap) | Ubuntu 24.04, AMD Radeon PRO W7700 (RDNA 3, Mesa RADV 25.2.8, Vulkan) | only with `--enable-unsafe-webgpu --enable-features=Vulkan --use-angle=vulkan --ignore-gpu-blocklist`, and `--use-webgpu-power-preference=force-high-performance` to get the W7700: the default and `low-power` pick the Ryzen iGPU (`rdna-2`) | measured (gpu#23): 123/123 ONNX cases, MNIST-12 within 3.1e-5, batch module (6 cases), LLM oracle (32 of 32 tokens equal to llama.cpp, logits within 1.2e-6 relative of the CPU path); the other e2e tests were not run |
| Firefox 159 | macOS 27, Apple M4 Max | yes in a normal window; `--headless` answers `requestAdapter` with null | passes, **headed** (123/123, MNIST-12 within 3.1e-5, 2 s deadline at 2000 ms) |
| Firefox 158.0.1 | macOS runner (GitHub Actions, `macos-latest`), headed | yes (adapter not nameable; `isFallbackAdapter` false) | passes in CI (batch module, 123/123 ONNX cases, 2 s deadline) |
| Firefox | Windows | since 141 | not run here |
| Firefox | macOS, Apple silicon | 145 (Tahoe), 147 (older macOS); not Intel | see the row above |
| Firefox | Linux | not supported (MDN browser-compat-data `api.GPU`, bug 2006676) | none |
| Safari 27.2 | macOS 27, Apple M4 Max | yes (since 26), adapter reported as vendor `apple`, architecture `apple`, not a fallback; `shader-f16` yes, `subgroups` no here | passes locally through safaridriver (batch module, refusals, quota, 123/123 ONNX cases, MNIST-12 within 3.1e-5; no GPU sanction on legitimate work, longest submit 11.7 ms; **the hostile-shader run does not pass**: WebKit's own watchdog destroys a 3 s shader and then refuses adapters, see docs/gpu-scheduling.md) |
| Safari 27.2 / 26.6.2 | GitHub macOS runners (`macos-15`, `macos-26`; virtual machines) | no: `macos-15` has no `navigator.gpu`, `macos-26` answers `requestAdapter` with null | no CI lane (the lane fails there; WebKit excludes VMs) |
| go-webengine engine v0.30.0 | pure Go, any platform | no WebGPU: since v0.30.0 Worker, SharedArrayBuffer and its views, Atomics, posting a SAB or a view, crossOriginIsolated (and its negative control: none without COOP/COEP), unsolicited worker messages, WebAssembly and `fetch` are green; OPFS, OffscreenCanvas and navigator.gpu are missing | **expected red** until go-webengine M4 (OPFS) and M5 (navigator.gpu); non-blocking CI job |

Adapter selection: every `requestAdapter` the module makes (`gpubatch`,
`webgpu`, `nn/webgpu`) passes `powerPreference: "high-performance"`
(`internal/adapter`). Without it, a browser on a PC with an integrated and a
discrete GPU returns the integrated one: Chromium 157 on Linux picked the
Ryzen iGPU (`rdna-2`) over a Radeon PRO W7700 (`rdna-3`) until this was set
(gpu#23). Browsers with one GPU, and software adapters, are unaffected.

The go-webengine lane (`GPU_E2E_BROWSER=webengine go test -tags e2e ./e2e/`)
renders the test pages with `engine.New()`, `EnableDoneSignal`,
`EnableWebGPU` (Metal, so macOS only), an OPFS root per run, raised wasm
caps, a 1 GiB fetch response cap (GGUF models), 2000 requests per page, a
4 GiB heap ceiling (the engine interprets wasm since v0.43.4 and stops a
render past the ceiling) and `JSLog` captured (the driver is
`e2e/webengine`, a module of its own pinned to engine v0.45.2, so this
module's dependency graph does not grow).
It fails, never skips, and every failure names the missing capabilities
("no Worker", "no navigator.gpu", ...) from a probe page that reports each
item separately (`TestBrowserCapabilities` runs the same probe on any
browser), including, inside a dedicated worker, every global the compute
workers use; OffscreenCanvas and a worker's Cache API are listed but not
required. The test workers log their milestones (fetched, instantiated,
failed with the stack) to the page's console, so a run that stops says
where. The worker probe also runs what go-gfx/webgpu needs for compute: a
storage-buffer dispatch with an explicit pipeline layout, read back with
`mapAsync`; and the container path: i64 arguments as BigInt, and a
worker's exported shared memory posted to its creator, then grown by the
guest's `memory.grow`, seen grown there. On v0.45.2 the GPU containers
run on Metal: the no-GPU and quota cases pass, and five of the eight
refusals behave as in Chromium; every other case, and the ONNX cases, stop
at the one missing piece, `createBindGroup` on a buffer sub-range.

Debugging the lane: `GPU_E2E_WEBENGINE_TRACE=1` prints every console line,
with its time, and the driver's phases to stderr as they happen (a native
crash kills the driver before its JSON result); `GPU_E2E_WEBENGINE_WRAP`
runs the driver under another program, split on spaces (for example
`lldb --batch -s run.txt -K crash.txt --`).

Per-submit round trip seen from the guest, 30 interleaved rounds each, Apple
M4 Max, 1-minute load 4 to 5, medians (read 4 KiB / read 1 MiB /
compute+copy+read 4 KiB / 1 MiB): Chromium 151 360 / 1020 / 825 / 1235 us;
Safari 27.2 340 / 1200 / 820 / 1340 us (340 / 840 / 860 / 940 us in the
2026-11-04 run below); Firefox 159 about 200 000 us for all four before
go-wasmcontainers/gpu#25 was fixed, and 2 560 / 3 060 / 2 580 / 3 100 us after
(go-gfx/webgpu v0.12.1). Timers are quantised (20 us in Chromium and Safari),
so differences under 100 us are noise.

Before / after the Firefox fix, same machine, 2026-11-04, 30 rounds, medians
(read 4 KiB / read 1 MiB / compute+copy+read 4 KiB / 1 MiB, us), 1-minute load
5.0 to 5.7 (the Firefox-before run ended at 8.2): Firefox 159 199 940 / 200 040
/ 199 940 / 200 040 -> 2 560 / 3 060 / 2 580 / 3 100; Chromium 345 / 720 / 830
/ 960 with the fix (the interleaved Chromium run without it was skipped by the
load guard; the figures in the previous paragraph are its baseline). LLM
decode (llm v0.3.1, SmolLM2-135M Q4_0, `TestLLMAB -p 128 -n 128`, load 7.1 to
7.9, 2 rounds x 3 interleaved runs): Firefox 159 about 290 tok/s (3.4 ms/tok)
with go-gfx/webgpu v0.12.1 and also with v0.12.0, Chromium about 550 tok/s
(1.8 ms/tok). The default decode does not read back per token, so the fix does
not move it; a one-read-per-token configuration (`-batch 1`) is not yet
compared before/after.

Why Firefox took 200 ms: it resolves `onSubmittedWorkDone` and `mapAsync` on
the tick of a 100 ms timer (`WebGPUParent.cpp` `POLL_TIME_MS = 100`; Bugzilla
[1870699](https://bugzilla.mozilla.org/show_bug.cgi?id=1870699) "Don't poll
WebGPU from a timer", assigned, a background-thread rewrite is in progress; our
measurements and the kick workaround are in
[comment 60](https://bugzilla.mozilla.org/show_bug.cgi?id=1870699#c60)), and the host waited twice (work-done, then map). The
control is `go run ./internal/cmd/ffwait` (a bare page, no engine, also in a
Worker): every await costs 100 ms, an empty submit as a kick takes it to
0.3 ms. go-gfx/webgpu v0.12.1 kicks the queue with an empty `submit([])`
while it waits ([docs/readback.md](https://github.com/go-gfx/webgpu/blob/main/docs/readback.md)).
When Firefox ships the fix the kick is a no-op.

Safari: `GPU_E2E_BROWSER=safari go test -tags e2e ./e2e/` drives the real
Safari through `safaridriver` (W3C WebDriver, `internal/webdriver`;
Playwright's WebKit is not Safari) after a one-time `sudo safaridriver
--enable`; without it the lane fails with that instruction. Sources opened:
MDN browser-compat-data `api/GPU.json` (Safari 26), the WebKit blog on Safari
26.0, safaridriver(1) of Safari 27.2, and WebKit
`Source/WebGPU/WebGPU/HardwareCapabilities.mm` (shader-f16 reported always,
subgroups on Metal 3, no WebGPU on virtual machines unless
`WebKitAllowWebGPUOnVMs`; setting that default on `macos-26` did not give a
working lane: the job failed after 15 minutes).

Sources opened: MDN browser-compat-data `api/GPU.json` (Firefox
`version_added` 141, partial; the per-platform notes above) and the Firefox
141 release notes (Windows). The Firefox lane drives the real Firefox binary
(`GPU_E2E_BROWSER=firefox go test -tags e2e ./e2e/`, no automation protocol:
the page posts its report to the test server). Playwright's own Firefox build
(Nightly 1538) does not start on macOS 27 in this setup (its Juggler handshake
times out, a helper process is killed). Firefox hides the adapter's vendor
and architecture, so the lane relies on `isFallbackAdapter`; it fails, never
skips, when `requestAdapter` returns null. Firefox 159 offers `shader-f16`;
the f32 path stays mandatory (older Firefox, other platforms). Known:
a submit with a read-back cost about 200 ms in Firefox 159 (its 100 ms
device-poll timer, twice), now about 2.6 ms with go-gfx/webgpu v0.12.1
(go-wasmcontainers/gpu#25); the longest
legitimate submit measured there is 226 ms, which is why the steady-state
budget is 500 ms. Destroying a device does not cancel work the GPU already
holds (go-wasmcontainers/gpu#26): a sanctioned container's remaining shader
time is still waited out, by the host before it serves anyone else, so an
innocent container is not sanctioned for it
([docs/gpu-scheduling.md](docs/gpu-scheduling.md), with the measurements).

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
