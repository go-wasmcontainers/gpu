# Small LLMs on the GPU for WASI containers: design

Status: **design** (bibliography and decisions). Steps 1 M1 and M2 are
implemented (see the table of missing pieces); nothing else is.
Every reference in the last section was opened on 2026-11-02 while writing
this page; figures marked *probe* were read from live HTTP responses that day.
No performance was measured for this page.

Goal: a go-wasmcontainers container (a `wasip1` core module) runs a small
language model (0.1 to 1.5 B parameters, 4 or 8 bits) on the GPU and streams
tokens — in the browser through WebGPU first, natively second.

## Decisions (user, 2026-11-02)

1. **Inference runs inside the container.** Running it in the engine would
   break the isolation wasm gives and put every new model into the engine.
   The `wasip1` guest holds the GGUF parser, the tokenizer, the weights'
   upload, the WGSL kernels and the KV cache. The host only **validates and
   replays a batched GPU command stream** — go-gfx/webgpu's wire format, the
   one gpudev's webgpu-batch capset already carries — with **one guest→host
   crossing per submit**, about one per token. The component model is not
   needed. A host-side wasi-nn `ggml` backend is deferred (later, or never);
   LlamaEdge compatibility is not a goal.
2. **Weights come from Hugging Face**: precise files pinned by revision and
   sha256, a narrow CSP `connect-src`, cached in OPFS.
3. **A separate GPU memory grant**, opt-in per container, ceiling 1.5 GiB.

## Why inside, and what it costs

| | Host-side backend (deferred) | **Guest-side inference over a replayed stream** (chosen) |
|---|---|---|
| Model code | in the engine: every architecture, tokenizer and quantisation is host code, reachable by every container | in the container image; the engine learns nothing per model |
| Trust | host parses GGUF and runs model logic on untrusted input | host parses only frames, descriptors and streams — formats it already fuzzes (below) |
| Crossings per token | one `compute_single` | one `submit` (stream + readback of the sampled token) |
| GPU work per token | one `queue.submit` | one `queue.submit` (`Device.Replay`: one JS call [gfx-readme]) |
| Weight upload | host → GPU | guest reads 16 MiB chunks of a read-only file, each sent in a frame; the frame is copied out of guest memory once, then written to the GPU — one more copy per chunk than the host path, paid once per load |
| ABI | wasi-nn + WasmEdge's extension [we-func] | webgpu-batch frames [vgpu-webgpu] through a small host module |

The per-call cost that ruled out guest-side GPU work before (one SAB round
trip, 38 µs, per `wasi:webgpu` call; DESIGN.md R3) does not apply: a frame
carries the whole token's work. Guest memory (wasm32, ≤ 4 GiB) never holds
the model: weights stream through one chunk buffer to the GPU, and only the
tokenizer tables, a few activations for readback and the frame being built
live in the guest.

## Host ABI: what exists, what is missing

### Exists

- **go-wasmcontainers/gpu** serves `wasi:webgpu@0.3.0-rc.2` lowered by hand
  to core-module imports (package `wit`, `webgpu.NewModule`), one host call
  per WebGPU call, with per-process quotas (4096 handles, 256 MiB, 1 MiB per
  WGSL module, 256 pipelines, 2 s per submit) (README). Usable for setup;
  too chatty per token.
- **go-gfx/webgpu** (v0.5.0 when this page was written, **v0.7.0 now** and
  pinned here since gpu#20; M1 and M2 below landed in v0.7.0) [gfx-readme]:
  - `wire`, no GOOS constraint: opcodes, `Decode` (the Go mirror of
    `replay.js`: argument counts, placement inside passes, object ids
    known, data ranges), `Remap` (refuses any id the mapping does not
    grant), `StreamVersion` 3 [gfx-wire];
  - id-based creation descriptors: `EncodeDescriptor`,
    `DecodeDescriptor` (refuses unknown fields, wrong reference counts and
    kinds; fuzzed), `wire.RemapDescriptor`, `wire.MaxRefs` 4096;
  - `Device.Replay(words, data, refs...)`: checks the stream with
    `wire.Decode` against the granted objects only, then one call to
    `replay.js` [gfx-replay];
  - `NewRemoteDevice(sink, label, quota)` (build `!js`, so it compiles for
    `wasip1`): the go-gfx/webgpu API recording into a `RemoteSink`
    (`Create(id, kind, desc)`, `Destroy(id)`, `Submit(words, data)`), with
    the browser path's checks and quota at the call. **Not carried:**
    `MapAsync` (readback), `WriteTexture`, command encoders, bundles —
    `ErrRemoteUnsupported` [gfx-remote];
  - `Quota` charged before the browser is called; optional features and
    limits refused unless the quota lists them.
- **go-virtio/gpu `wgpubatch`** (guest and host): the capset blob (magic
  `WGPB`, version, `stream_version`, `max_frame`, `max_objects`,
  `max_rings`, `max_memory`), frames of messages CREATE, DESTROY, STREAM
  (a Recorder submit verbatim), BIND_RESOURCE; `Decode` with limits;
  fuzzed [vgpu-webgpu].
- **go-virtio/gpudev** (host): the webgpu-batch context — guest ids private
  to the context, `wire.Remap`/`RemapDescriptor` before the backend sees
  them, a failed frame loses its context — and `webgpuback`, whose
  `Context` (`Create`, `Destroy`, `Replay`, `Bind`, `Close`) runs on
  go-gfx/webgpu; validated in Chromium on Apple Metal and SwiftShader
  [gpudev][gpudev-src]. The context logic (`batchContext`, `runFrame`) is
  unexported and tied to virtio's `SUBMIT_3D`.

### Missing (each in its owner, one implementation)

| # | Missing piece | Owner | Why |
|---|---|---|---|
| M1 | **Done** (go-gfx/webgpu v0.7.0, go-virtio/gpudev v0.2.0). Transport-free webgpu-batch host context: package **`go-gfx/webgpu/batch`** (`Context` with `Create`/`Destroy`/`Stream`/`Lookup`, `Backend`, `Reader`, and `DeviceBackend` under `GOOS=js`), not `gpudev/batch`: gpudev depends on go-gfx/webgpu, and the context needs only `wire`, so the lowest shared library hosts it and `wgpubatch` framing stays in go-virtio/gpu. gpudev is rewired onto it, no behaviour change | go-gfx/webgpu | one implementation of the frame semantics |
| M2 | **Done in go-gfx/webgpu v0.7.0; gpudev transport open** ([gpudev#10](https://github.com/go-virtio/gpudev/issues/10)). **Read-back through the stream, not `RemoteSink.Read`**: `wire.ReadBuffer` (**opcode 34**, `StreamVersion` **4**: id, offset(2), size, windowOffset), `Recorder.ReadBuffer`, `ReadSink.SubmitRead`, a remote `Buffer.MapAsync` (read mode), `Quota.ReadBytes`. Rules are `GPUBuffer.mapAsync`'s. Window **MaxRead 1 MiB by default, 64 MiB at most** (`wire.MaxReadWindow`), checked before replay. gpudev passes no window, so it refuses a reading stream and `Bind` stays `ErrBind` | go-gfx/webgpu, gpudev | the guest must read the sampled token (4 bytes) and, in tests, activations |
| M3 | Features and limits in the capset: `shader-f16`, `subgroups`, `maxStorageBufferBindingSize`, `maxComputeWorkgroupStorageSize`, `maxComputeInvocationsPerWorkgroup` (capset version 2) | go-virtio/gpu `wgpubatch` | the guest picks kernel variants; a remote device has no adapter to ask |
| M4 | Host-side range checks on replay: `WriteBuffer`, `CopyBufferToBuffer` ranges against buffer sizes, `Dispatch` counts against `maxComputeWorkgroupsPerDimension`, before `replay.js` | go-gfx/webgpu (`checkReplay`) | today these reach WebGPU validation (`wire.Decode` checks ids and data ranges, not buffer sizes) [gfx-replay]; a Go error at submit is the rule of the library |
| M5 | Host module `wasi_webgpu_batch` for `wasip1` guests (below) | go-wasmcontainers/gpu | the transport |
| M6 | Guest sink: a `RemoteSink` over M5, frames built with `wgpubatch.Encoder` | go-wasmcontainers/gpu (`wasigpu`) | guests write against the go-gfx/webgpu API unchanged |
| M7 | Read-only mount of a host-fetched model file into the container (`/models/<name>.gguf`) | go-wasmcontainers/engine, playground | the guest reads weights by `fd_read`; it never fetches |

### The host module (M5)

Import module `wasi_webgpu_batch` (core module, `i32`/`i64`, errno results,
like `wasi_ephemeral_nn`):

| Function | Does |
|---|---|
| `caps(buf, len) -> errno` | writes the `wgpubatch` capset blob (M3) |
| `context_create(out) -> errno` | needs `--gpu`; one go-gfx/webgpu device per container (DESIGN.md), a context per call, at most 4 |
| `submit(ctx, frame, frame_len, read, read_len) -> errno` | copies the frame out of guest memory, decodes and runs it (M1), waits for completion, then performs the frame's reads (M2) into `[read, read+read_len)` before returning |
| `context_destroy(ctx) -> errno` | releases everything the context holds |

The guest blocks in `submit` through the engine's existing
`wasi.Result.Wait`/`Resume` (DESIGN.md); the host writes guest memory only
inside that call. Natively, `submit` replays nowhere until a pure-Go WebGPU
exists (DESIGN.md: no adapter natively) — `context_create` returns
`unsupported`, and native inference is the guest's own CPU path.

### Security model of the replay

The guest is hostile; the replay is the trust boundary.

1. **Copy, then decode.** The frame (≤ `max_frame`) is copied out of guest
   memory before anything reads it (the virglrenderer rule in
   [vgpu-webgpu]).
2. **Framing** by `wgpubatch.Decode`: magic, version, sizes, alignment,
   message count. Fuzzed.
3. **Handles.** Guest ids are private to the context and below
   `max_objects`; `wire.Remap`/`RemapDescriptor` translate them and refuse
   any id not granted; the host's own objects are never in the table.
4. **Descriptors** pass `DecodeDescriptor` and the direct path's
   validation; each creation is charged to the context's `Quota` (memory,
   pipelines, WGSL ≤ 1 MiB per module) before the browser is called.
5. **Streams** pass `wire.Decode` (structure, ids, data ranges in u64) and
   M4 (buffer ranges, dispatch sizes).
6. **WebGPU validation is the second line.** Each replay runs in an error
   scope; an error loses the context, never the host, never another
   container. WebGPU zeroes new resources and bounds shader accesses
   [webgpu-sec in DESIGN.md].
7. **GPU time.** A submit not done within 2 s destroys the context's
   device (the existing quota). Prefill is chunked by the guest to stay
   well under it.
8. **Reads** copy only from buffers of the calling context, only into the
   `read` window the call passed, bounded in u64.
9. **Features.** `shader-f16` and `subgroups` are granted when the adapter
   has them; `timestamp-query` stays refused.
10. **Fuzzing.** Existing: `wgpubatch` decode, `FuzzDecodeDescriptor`,
    `wire.Decode` (and its browser agreement test with `replay.js`). New:
    `FuzzBatchCalls` drives arbitrary call sequences and frames over
    arbitrary guest memory, as `FuzzHostCalls` does for wasi-nn.
11. **32-bit.** The engine worker is wasm32: every guest u32/u64 is bounded
    in u64 before any `int` conversion.

### Cost of one read-back, and the per-token budget

Measured 2026-11-03 (go-gfx/webgpu `batch.TestReadLatency`, docs/readback.md
there): Apple M4 Max, Chromium, Metal, worker, host side only (replay, wait
for the GPU, `mapAsync`, copy, unmap), medians of 30 interleaved rounds at a
1-minute load of 5.0.

| window | read only | compute pass + buffer copy + read |
|---:|---:|---:|
| 4 KiB | 124 µs | 270 µs |
| 1 MiB | 155 µs | 320 µs |

The fixed cost of a read is about 120 µs and does not grow with the window up
to 1 MiB (1 MiB adds about 30 µs). A decode token reads 4 bytes (`argmax`) or
k pairs (`topk`), so it pays the 4 KiB figure at most: **about 0.12 ms for
the read, 0.27 ms for the trivial-compute submit as a floor** (a ceiling of
about 3 700 submits per second before any model work). At 20 tokens per
second (50 ms per token) the read is 0.25 % of the budget; at 100 tokens per
second (10 ms) it is 1.2 %. Reading the logits instead (600 KB for Qwen2.5)
would cost about 0.15 ms more than the sampled token but 600 KB of window per
token and the CPU argmax: still well below the 1 MiB window, but the GPU
`argmax` stays, for the window, not for latency. The guest's own crossing
(the 38 µs of DESIGN.md R3) comes on top. Not measured: a `wasip1` guest,
Firefox, a non-trivial kernel.

## The guest: an inference library in Go

A Go package (working name `llm`, in this repository) built for `wasip1`,
over the go-gfx/webgpu API with a remote device (M6). Model families first:
`llama` (SmolLM2, TinyLlama, Llama 3.2) and `qwen2` (Qwen2.5: Q/K/V biases).

- **GGUF**: magic, version 3, `tensor_count`/`metadata_kv_count` (u64),
  typed metadata, tensor infos (name, ≤ 4 dimensions, `ggml_type`, aligned
  offset), `general.alignment` default 32 [gguf]. Bounded and fuzzed even
  though it runs in the guest: a crash there is the guest's, but the library
  is reused natively.
- **Tokenizers** from `tokenizer.ggml.*`: `gpt2` BPE and `llama`
  SentencePiece [gguf]. Chat templating stays the application's
  (`tokenizer.chat_template` is Jinja).
- **Upload**: per tensor, read 16 MiB chunks, repack, `WriteBuffer` in a
  frame. GGUF blocks of Q4_0 (18 bytes), Q8_0 (34) and Q6_K (210) are not
  4-byte aligned [ggml-common], and WGSL reads storage as 32-bit words: each
  tensor becomes two aligned arrays (packed quants as `array<u32>`, scales
  and mins as f32 — exact from f16), split into buffers of at most the
  binding size, so no adapter limit has to be raised.
- **KV cache** in GPU buffers owned by the guest, one per conversation;
  prefix reuse across turns is the guest's choice.
- **Per token**: one `Recorder` with every dispatch of the forward pass,
  `argmax` (or `topk`) on the GPU, one `submit` with a 4-byte read (or
  k pairs).
- **CPU path**: the same forward pass in Go, the equality oracle for the
  WGSL and the native fallback.

### WGSL kernels

The reference list is llama.cpp's WebGPU shaders [lcpp-webgpu]; ours are
written from the ggml semantics, not copied.

| Kernel | Use | Variants |
|---|---|---|
| `get_rows` | embedding gather, dequantising one row | per weight type |
| `rms_norm_mul` | RMSNorm × weight (fused) | f32; f16 |
| `matvec_q` | decode (n = 1): every projection and the logits | Q4_0, Q8_0, Q4_K, Q6_K, F16, F32; subgroup reduction where `subgroups` exists |
| `matmul_q` | prefill (n > 1), tiled in ≤ 16 KiB workgroup memory | same types |
| `add_bias` | Qwen2 Q/K/V biases | f32 |
| `rope` | rotary positions, normal and NeoX ordering, `rope.freq_base` and scaling keys [gguf] | f32 |
| `kv_write` | append K and V at the position (llama.cpp `set_rows`) | f32 cache; f16 cache where `shader-f16` |
| `attn_decode_split` + `attn_decode_reduce` | one query against the cache, split along the sequence then reduced (llama.cpp `flash_attn_vec_split`/`_reduce`); GQA by head index | f32; f16 |
| `attn_prefill` | causal attention over a prompt chunk, tiled with online softmax | f32; f16 |
| `swiglu` | `silu(gate) · up` (llama.cpp `glu`) | f32 |
| `add` | residuals | f32 |
| `argmax` | greedy sampling on the GPU: 4 bytes read back instead of the logits (151 936 floats for Qwen2.5's vocabulary, ~600 KB per token) | f32 |
| `topk` | top-k (k ≤ 64) on the GPU; top-p, min-p, penalties and the draw on the CPU over k pairs | f32 |

Q4_K's 6-bit packed scales unpack as `get_scale_min_k4` does
[ggml-quants]. **An f32 path is mandatory**: `shader-f16` and `subgroups`
are absent from Firefox, `subgroups` from Safari [bcd-feat]; llama.cpp's
WebGPU backend requires `shader-f16` [lcpp-webgpu].

## Limits and quotas

WebGPU defaults: `maxBufferSize` 256 MiB, `maxStorageBufferBindingSize`
128 MiB, `maxComputeWorkgroupStorageSize` 16 384 B,
`maxComputeInvocationsPerWorkgroup` 256 (128 in compatibility mode),
`maxStorageBuffersPerShaderStage` 8, `maxComputeWorkgroupsPerDimension`
65 535; a required limit better than the adapter's rejects `requestDevice`
with `OperationError` [webgpu-spec]. WebLLM requests 32 KiB of workgroup
storage and 10 storage buffers per stage [tvm-webgpu]. **Our kernels stay
within the defaults**; the host requests no raised limit.

**The flag: `wctr run --gpu --gpu-memory=SIZE`.** It sets the GPU memory
quota (`webgpu.Quota.MemoryBytes`, advertised to the guest as the capset's
`max_memory`) of the container's webgpu-batch contexts, summed over them.
Without it the quota is the existing 256 MiB. `SIZE` above **1.5 GiB**
(1 610 612 736 bytes) is refused by `wctr` before the container starts; it
needs `--gpu`. API: `HostConfig.DeviceRequests[].Options["memory"]`, the
Docker Engine API's shape [moby-hc in DESIGN.md]. `wasi:webgpu` and wasi-nn
keep their own 256 MiB quotas.

What fits: Qwen2.5-0.5B-Instruct GGUF is 428 730 208 bytes in Q4_0 and
675 710 816 in Q8_0 [hf-tree]; WebLLM states 944.62 MB for it and
1 629.75 MB for Qwen2.5-1.5B (q4f16_1, 4 096-token context)
[webllm-config], so a 1.5 B model fits under 1.5 GiB only with a shorter
context. The KV cache is `2 × block_count × head_count_kv × head_dim ×
ctx × bytes`, from the GGUF keys [gguf]; the guest checks it against
`max_memory` before allocating.

| Quota (per container) | Value | Where |
|---|---|---|
| GPU memory, batch contexts | 256 MiB, or `--gpu-memory` ≤ 1.5 GiB | `Quota`, at CREATE |
| Contexts | 4 | `context_create` |
| Live objects per context | `max_objects` 4096 | frame decode |
| Frame | `max_frame` 32 MiB (a 16 MiB weight chunk plus its stream) | `submit` |
| Read window | ≤ 1 MiB per `submit` | `submit` |
| WGSL / pipelines | 1 MiB per module / 256 | `Quota` |
| GPU time | 2 s per submit | host timer |
| Features | `shader-f16`, `subgroups` if the adapter has them; no `timestamp-query` | device creation |

## Where the weights live

GitHub Pages refuses sites above 1 GB [gh-pages]: weights never go into the
playground or its `/v2` mirror.

*Probe* (2026-11-02, `Origin: https://go-wasmcontainers.github.io`,
`Range: bytes=0-15`): `huggingface.co/<repo>/resolve/<rev>/<file>` answers
`302` with `access-control-allow-origin` echoing our origin, `accept-ranges:
bytes`, `x-linked-size`, `x-linked-etag` (= the file's LFS sha256),
`cache-control: no-store`; the redirect target answered `206` with
`content-range` and `access-control-allow-origin: *`; the 16 bytes were
`GGUF`, version 3. Three files (two Qwen2.5-0.5B quantisations, one SmolLM2
in another repository) all redirected to **`us.aws.cdn.hf.co`**, path
`/xet-bridge-us/…`, with a signed, expiring query. A single `bytes=N-M`
range is a CORS-safelisted header [fetch]: no preflight.

- **Registry** (JSON, in the playground): `{name, repo, revision (commit),
  file, size, sha256, license}`. The resolve URL uses the commit, never
  `main`. A container names an entry (`wctr run --model NAME`); it never
  supplies a URL.
- **CSP**: `connect-src 'self' https://huggingface.co
  https://us.aws.cdn.hf.co`. Hugging Face may move the redirect to another
  host; the fetch then fails on CSP (a refusal, never a wider policy), and
  the registry test (plan step 6) names the new host.
- **Fetch** in the engine worker by ranges (16 MiB), resumable, into
  **OPFS** keyed by sha256; verified whole before it is marked complete;
  `navigator.storage.persist()` requested. The origin quota covers Cache
  API, IndexedDB and OPFS together (Chrome up to 60 % of the disk, Firefox
  best-effort min(10 %, 10 GiB per site), Safari ~60 %; LRU eviction of the
  whole origin) [mdn-quota]. WebLLM offers `cache`, `indexeddb`,
  `cross-origin`, `opfs` and defaults to the Cache API [webllm-config];
  OPFS is chosen for ranged, resumable writes.
- **Mount** (M7): the verified file appears read-only at
  `/models/NAME.gguf`. The guest reads it with `fd_read`; it has no network.
- **Native**: `os.UserCacheDir()/go-wasmcontainers/models/<sha256>.gguf`,
  same pinning and verification, mounted the same way.

## Alternatives considered

| | Verdict |
|---|---|
| Host-side wasi-nn `ggml` backend (WasmEdge's `compute_single`, `get_output_single`, `fini_single`, errnos 100–103 [we-func][we-types][we-compute][we-out]) | **deferred**: model code in the engine (decision 1). wasi-nn itself has no streaming and no KV cache [nn-witx][nn-wit] |
| Extend ONNX (MatMulNBits, GroupQueryAttention [ort-contrib][ort-kernels]) | rejected: the two largest ORT contrib operators, a KV cache wasi-nn's ONNX shape cannot hold, tokenizer outside the graph; onnx-community Qwen2.5-0.5B q4 is 786 MB [hf-tree] |
| WebLLM or llama.cpp ggml-webgpu in the engine worker | rejected: their own device outside our quotas, JS/C++; llama.cpp requires `shader-f16` (no Firefox) [lcpp-webgpu][bcd-feat] |
| llama.cpp built for `wasm32-wasip3` in the guest over `wasi:webgpu` [lcpp-build] | rejected for now: a component (the engine is a core-module host), C++, `shader-f16` required, one host call per WebGPU call. The guest-side principle is the one we keep |

## Performance reference

Machine: Apple M4 Max. Models: Qwen2.5-0.5B-Instruct (GGUF Q4_0 and Q8_0;
WebLLM `q4f32_1` and `q4f16_1`) and SmolLM2-360M-Instruct. Formats differ
(ggml blocks vs MLC group quantisation); each row says which.

| Row | Tool | Setting |
|---|---|---|
| CPU reference | llama.cpp, tagged release, CPU build | `llama-bench -t 1 -ngl 0 -p 128 -n 128`, pp and tg tokens/s |
| Browser reference | WebLLM, same Chromium as our e2e | decode and prefill tokens/s, same prompt |
| Native GPU (context only) | llama.cpp Metal | `-ngl 99` |
| Ours, CPU | the guest library's Go path, natively, one core | same prompt, greedy |
| Ours, browser | a container in the playground, Metal adapter, Chromium and Firefox | same; plus time per `submit` split into guest recording, crossing, replay, GPU |

Published anchor (not our machine): on an M3 Max in Chrome Canary 133,
WebLLM decoded Llama-3.1-8B at 41.1 tok/s and Phi-3.5-mini at 71.1 tok/s,
71.2 % and 79.6 % of native MLC-LLM [webllm-paper]. No single-core
llama.cpp figure for these models was opened; step 0 measures it.

Targets: CPU path ≥ llama.cpp `-t 1` (slower is an open defect); browser
decode ≥ 50 % of WebLLM on the same model at step 8, ≥ 80 % after step 9.
Interleaved rounds, never above a 1-min load of ~8, allocation counts
recorded.

## Plan

Each step: a PR in its owner repository, tests natively and in Playwright
(Chromium, Firefox), README and docs, a release.

| # | Step | Acceptance test |
|---|---|---|
| 0 | Baselines | The table above filled for both models on the M4 Max (llama.cpp one core; WebLLM in Chromium), with load average, versions and commands committed under `docs/` |
| 1 | Host ABI pieces: **M1 and M2 done** (go-gfx/webgpu v0.7.0, go-virtio/gpudev v0.2.0; acceptance passed in Chromium on Apple Metal, every element of a 4 KiB and a 1 MiB read-back equal; remaining: the gpudev read window), M3 capset v2 (go-virtio/gpu), M4 replay range checks (go-gfx/webgpu) | gpudev's existing e2e still pass through the exported context; a remote `MapAsync` reads back what a compute pass wrote, in-process and in Chromium; a stream writing past a buffer is refused at submit with a Go error; fuzz targets extended |
| 2 | `wasi_webgpu_batch` host module (M5), guest sink (M6), `--gpu-memory` | A `wasip1` container doubles a buffer through a remote device and reads it back, in Chromium and Firefox; one `submit` per frame counted; `FuzzBatchCalls` in CI; refusals tested: no `--gpu`, `--gpu-memory` above 1.5 GiB, an allocation past the quota, an id of another context, a frame above `max_frame`, a read outside the window, a 2 s overrun |
| 3 | `gguf` parser and dequantisation (F32, F16, Q4_0, Q8_0, Q4_K, Q6_K) | Headers of both models read over HTTP ranges; `FuzzGGUF` with a committed corpus and a refusal test per bound; dequantised values bit-identical to ggml's `dequantize_row_*` on every tensor |
| 4 | Tokenizers (`gpt2`, `llama`) | Token ids identical to llama.cpp's tokenizer on a committed multilingual corpus, both directions |
| 5 | CPU forward pass (`llama`, `qwen2`), KV cache, sampling — the oracle port | Logits within 1e-3 relative of llama.cpp's for a fixed prompt; 64 greedy tokens identical; tok/s vs llama.cpp `-t 1` in the README |
| 6 | Registry, fetcher, OPFS cache, read-only mount (M7), CSP | A container reads `/models/NAME.gguf`; a second run makes zero model requests; a corrupted cache file is detected and refetched; an interrupted download resumes; a size or sha256 mismatch is refused; a test checks that every registry entry still redirects to a host the CSP lists |
| 7 | WGSL kernels, f32, default limits; weight upload by chunks | Each kernel: emulated index test + browser test against step 5; peak guest memory during load below 64 MiB for the 0.5 B model; 32 greedy tokens identical to the CPU path on Chromium/Metal and Firefox; GPU memory charged equals the guest's computed sum |
| 8 | Streaming chat container (`examples/chat`), one `submit` per token | Tokens stream in the playground; one crossing per decode token counted; decode tok/s vs WebLLM recorded |
| 9 | f16 and subgroup variants, prefill tiling, tuning | Same token test with variants on; performance table updated against WebLLM |
| 10 | Playground demo, landing, docs | Published pages checked (200, new text) |

## Open questions for the user

- **Q1 — First models.** SmolLM2-135M/360M and Qwen2.5-0.5B (Apache-2.0,
  per their model-card tags), then Qwen2.5-1.5B with a shorter context;
  Llama 3.2 1B (licence not checked for this page) — include it?
- **Q2 — Native GPU.** Natively `context_create` returns unsupported until
  a pure-Go WebGPU (gogpu, DESIGN.md) is evaluated; inference natively is
  the guest's CPU path. Confirm.
- **Q3 — Where the guest library lives.** In this repository (`llm`), or
  its own repository, since it is guest code reusable outside containers?

## References (opened)

- [gfx-readme] go-gfx/webgpu `README.md`, v0.5.0 (c9bc019): batching, `wire`, remote device, descriptors, quotas
- [gfx-wire] go-gfx/webgpu `wire/wire.go` (opcodes, `StreamVersion` 3, `Decode`)
- [gfx-remote] go-gfx/webgpu `remote.go` (`//go:build !js`, `ErrRemoteUnsupported`), `remotesink.go` (`RemoteSink`)
- [gfx-replay] go-gfx/webgpu `replaycheck.go` (`checkReplay`), `record_js.go` (`Device.Replay`)
- [vgpu-webgpu] go-virtio/gpu `docs/WEBGPU.md` (capset 30 blob, frames, quotas, security model)
- [gpudev] go-virtio/gpudev `README.md`, main (73421c9)
- [gpudev-src] go-virtio/gpudev `batch.go` (`runFrame`), `backend.go` (`Context`, `Window`)
- [nn-witx] WebAssembly/wasi-nn `wasi-nn.witx`, main (71320d95b8): https://github.com/WebAssembly/wasi-nn/blob/main/wasi-nn.witx
- [nn-wit] WebAssembly/wasi-nn `wit/wasi-nn.wit`, `wasi:nn@0.2.0-rc-2024-10-28`
- [we-func] WasmEdge `plugins/wasi_nn/wasinnfunc.cpp`, master (673031ab7e)
- [we-types] WasmEdge `plugins/wasi_nn/wasinntypes.h`
- [we-compute] WasmEdge `plugins/wasi_nn/GGML/compute/compute_engine.cpp`
- [we-out] WasmEdge `plugins/wasi_nn/GGML/core/output_generator.cpp`
- Also opened for the deferred option: WasmEdge `plugins/wasi_nn/DESIGN.md`, `GGML/metadata/metadata_parser.cpp`, `GGML/compute/inference_manager.cpp`
- [gguf] ggml `docs/gguf.md`, master (facdc44a67)
- [ggml-common] ggml `src/ggml-common.h` (`block_q4_0`, `block_q8_0`, `block_q4_K`, `block_q6_K`)
- [ggml-quants] ggml `src/ggml-quants.c` (`dequantize_row_*`, `get_scale_min_k4`)
- [lcpp-webgpu] llama.cpp `ggml/src/ggml-webgpu/` (`wgsl-shaders/`, `ggml-webgpu.cpp`), master (1b6935b227); `docs/ops/WebGPU.csv`
- [lcpp-build] llama.cpp `docs/build.md` (WebGPU, WASI, wasi:webgpu)
- [webllm-paper] Ruan et al., arXiv:2412.15803v2, Table 1: https://arxiv.org/html/2412.15803v2
- [webllm-config] mlc-ai/web-llm `src/config.ts`, main (f1ecbaa916); also `src/engine.ts`
- [tvm-webgpu] apache/tvm `web/src/webgpu.ts` (`detectGPUDevice`)
- [ort-contrib] onnxruntime `docs/ContribOperators.md`, main (c7bf7939b5)
- [ort-kernels] onnxruntime `docs/OperatorKernels.md`
- transformers.js, "Running models on WebGPU": https://huggingface.co/docs/transformers.js/guides/webgpu
- [webgpu-spec] gpuweb `spec/index.bs` (limits, `"shader-f16"`, `"subgroups"`, `requestDevice`); W3C CR Draft 15 September 2026
- [bcd-feat] mdn/browser-compat-data `api/GPUSupportedFeatures.json`
- [fetch] WHATWG Fetch, CORS-safelisted request-header (`range`)
- [mdn-quota] MDN, Storage quotas and eviction criteria
- [gh-pages] GitHub Docs, GitHub Pages limits
- [hf-tree] Hugging Face API, file trees of `Qwen/Qwen2.5-0.5B-Instruct-GGUF` and `onnx-community/Qwen2.5-0.5B-Instruct`, model-card tags, 2026-11-02
- go-wasmcontainers/gpu `README.md`, `docs/DESIGN.md`, `nn/nn.go`, `go.mod`
