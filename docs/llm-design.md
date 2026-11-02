# Small LLMs on the GPU for WASI containers: design

Status: **design** (bibliography and decisions). Nothing here is implemented.
Every reference in the last section was opened on 2026-11-02 while writing
this page; figures marked *probe* were read from live HTTP responses that day.
No performance was measured for this page.

Goal: a go-wasmcontainers container (a `wasip1` core module) sends a prompt
and receives a stream of tokens from a small language model (0.1 to 1.5 B
parameters, 4 or 8 bits) running on the GPU — in the browser through WebGPU
first, natively second. It builds on what exists (README, docs/DESIGN.md):
the `wasi_ephemeral_nn` host module, its ONNX CPU backend, the `nn/webgpu`
backend (one WGSL dispatch per ONNX node on go-gfx/webgpu, one JS call per
submit), the `--gpu` capability, 256 MiB of GPU memory per graph, 64 MiB of
model bytes per `load`.

## Verdict

**A dedicated GGUF backend on the host, reached through wasi-nn, confirmed**
with four corrections to the proposal:

1. **Adopt the de-facto LLM extension of wasi-nn, don't invent one.** Neither
   wasi-nn text (witx nor WIT) has streaming, a KV cache or session state
   [nn-witx][nn-wit]. WasmEdge's GGML backend (LlamaEdge) adds exactly what
   is missing to the *same witx module* our guests already import:
   `compute_single`, `get_output_single`, `fini_single`,
   `load_by_name_with_config`, metadata as JSON on input/output index 1, and
   errnos 100–103 [we-func][we-types][we-compute][we-out]. We serve that shape.
2. **The 256 MiB per-graph GPU quota and the 128 MiB default binding size do
   not fit even the smallest target.** Qwen2.5-0.5B-Instruct in GGUF Q4_0 is
   428 730 208 bytes [hf-tree]; WebLLM states 944.62 MB of VRAM for the same
   model at q4f16_1 with a 4096-token context [webllm-config]. The LLM backend
   needs its own, explicitly granted, memory quota, and kernels that split any
   tensor over several ≤ 128 MiB bindings so no adapter limit has to be raised.
3. **An f32 path is mandatory.** `shader-f16` and `subgroups` are absent from
   Firefox; `subgroups` is absent from Safari [bcd-feat]. llama.cpp's WebGPU
   backend *requires* `shader-f16` [lcpp-webgpu], so it cannot be our browser
   engine on Firefox. f16 and subgroup kernels are optional variants.
4. **Weights are repacked at upload.** GGUF blocks of Q4_0 (18 bytes), Q8_0
   (34) and Q6_K (210) are not 4-byte aligned [ggml-common], and WGSL storage
   buffers are read as 32-bit words. The host splits each tensor into an
   aligned structure of arrays (packed quants, scales) while streaming it to
   the GPU; the dequantised values stay bit-identical to ggml's.

## The three options

| | ONNX extended | **Dedicated GGUF backend** (chosen) | Wrap an existing runtime |
|---|---|---|---|
| What | Add `com.microsoft` MatMulNBits, GroupQueryAttention, RotaryEmbedding, SkipSimplifiedLayerNormalization… to `onnx` and `nn/webgpu` | New encoding `ggml` in `nn`: GGUF parser, tokenizer, a llama/qwen2 forward pass in Go (CPU) and WGSL (GPU) | (a) WebLLM in the engine worker; (b) llama.cpp ggml-webgpu via Emscripten in the engine worker; (c) llama.cpp built for `wasm32-wasip3` *inside the guest*, over our `wasi:webgpu` |
| Model files | onnx-community Qwen2.5-0.5B-Instruct: `model_q4.onnx` 786 MB, `model_q4f16.onnx` 483 MB, `model.onnx` 1.99 GB [hf-tree]; tokenizer and chat template outside the graph | One file holds weights, hyper-parameters, tokenizer (`tokenizer.ggml.*`) and `tokenizer.chat_template` [gguf]; Q4_0 428.7 MB, Q4_K_M 491.4 MB, Q8_0 675.7 MB for the same model [hf-tree] | (a) MLC weights + one TVM-compiled wasm per model [webllm-paper]; (b)(c) GGUF |
| Operators to write | MatMulNBits: 2–8 bits, block ≥ 16 along K, zero points, `accuracy_level` [ort-contrib]; GQA with 16 inputs incl. `past_key`/`past_value` in BNSH, sliding-window eviction formula, KV quantisation [ort-contrib][ort-kernels] — a large surface, specified for ORT's EPs | ~15 kernels (below), one model family at a time, layouts fixed by ggml [ggml-common][ggml-quants] | none (a)(b); (c) needs our `wasi:webgpu` complete for Dawn's C API |
| KV cache, streaming | wasi-nn ONNX has none: past/present KV would cross guest memory every token, or the host keeps hidden state the ONNX graph does not describe | Host-side, per execution context; one token per `compute_single` | (a)(b) inside the runtime; (c) inside the guest |
| Quotas | ours | ours | (a)(b) **their own device**, outside our quotas (the same objection DESIGN.md made to onnxruntime-web); (c) ours, through `wasi:webgpu` |
| Browsers | ours (f32 everywhere) | ours (f32 everywhere, f16 where present) | (a) WebLLM requests 32 KiB workgroup storage and 10 storage buffers per stage, above the defaults 16 KiB and 8 [tvm-webgpu][webgpu-spec]; (b)(c) require `shader-f16`: no Firefox [lcpp-webgpu][bcd-feat] |
| Fit with the engine | good | good | (c) `wasm32-wasip3` is a component [lcpp-build]; the engine is a core-module host (DESIGN.md). Every WebGPU call is one SAB round trip (38 µs, DESIGN.md R3): assuming ~15 dispatches per layer × ≥ 4 calls each × 24 layers ≈ 1 440 calls ≈ 55 ms per token of overhead alone, before any GPU work |
| Language | Go | Go | (a) JS + TVM wasm; (b)(c) C++ |

**Verdict.** The ONNX route buys nothing the GGUF route lacks and costs the
two largest contrib operators of ORT, a tokenizer and a chat template that
live outside the model, and a KV cache that wasi-nn's ONNX shape cannot hold.
Wrapping (a) or (b) gives up our quotas, Firefox and the pure-Go rule. (c) is
worth keeping as a *guest-side* possibility — it costs the host nothing more
than a complete `wasi:webgpu` — but not as our engine: it needs the component
model and pays a round trip per WebGPU call. The host-side GGUF backend keeps
everything on our side of the quota: **one `queue.submit` per token** through
the go-gfx/webgpu Recorder, and a 4-byte readback of the sampled token.

## The wasi-nn interface

### What the standards give

- witx (`wasi_ephemeral_nn`): `load`, `load_by_name`, `init_execution_context`,
  `set_input`, `compute`, `get_output`; encodings `openvino, onnx, tensorflow,
  pytorch, tensorflowlite, autodetect`; errnos 0–8; no streaming, no state
  beyond the context handle [nn-witx]. The witx file says it is kept for
  consistency and points to the WIT.
- WIT `wasi:nn@0.2.0-rc-2024-10-28`: adds **`ggml`** to `graph-encoding`
  (order `openvino, onnx, tensorflow, pytorch, tensorflowlite, ggml,
  autodetect`), `compute(list<named-tensor>) -> result<list<named-tensor>>`;
  "An inference 'session' is encapsulated by a `graph-execution-context`"; no
  streaming, no KV cache [nn-wit]. A whole generation per `compute` is the
  most it can express.

### What WasmEdge adds (the shape we serve)

Host functions in the same module [we-func]: `load_by_name_with_config(name,
config)`, `compute_single(ctx)`, `get_output_single(ctx, index, buf, max,
written)`, `fini_single(ctx)`, `unload(graph)`, `finalize_execution_context(ctx)`.
Numbering [we-types]: encoding `GGML = 6` (after `Autodetect = 5`); errnos
`EndOfSequence = 100`, `ContextFull = 101`, `PromptTooLong = 102`,
`ModelNotFound = 103`; targets `CPU 0, GPU 1, TPU 2, AUTO 3`.

Semantics, from the source:

- `compute` clears the context and the KV cache (`llama_memory_clear`), runs
  the prompt, then samples until `n-predict` or end of generation;
  `EndOfSequence` becomes success [we-compute][we-infer].
- `compute_single`: the first call clears and evaluates the prompt; every
  call samples **one** token [we-compute].
- `get_output_single(…, 0, …)` returns the text piece of the latest token;
  index 1 returns metadata JSON `input_tokens`, `output_tokens`,
  `llama_build_number`, `llama_commit` [we-out].
- `PromptTooLong` when the prompt exceeds the context minus 4;
  `ContextFull` when position + new tokens exceed the context;
  `EndOfSequence` on an end-of-generation token (unless `ignore-eos`) or a
  reverse prompt [we-infer].
- `fini_single` "resets context-owned streaming state" [we-design].
- WasmEdge keeps the `llama_context` (hence the KV cache) **on the graph**
  and serialises all operations on a graph [we-design][we-compute].

### Our module

The `wasi_ephemeral_nn` module gains the six functions above, with
WasmEdge's names, parameters and errno values, so guests written against
LlamaEdge's bindings find what they import. Differences, each deliberate:

| Point | WasmEdge | Ours | Why |
|---|---|---|---|
| KV cache | per graph; contexts of one graph serialised | **per execution context** (one model's weights shared by several contexts, each with its own KV cache, counted in the quota) | two conversations on one model must not clear each other |
| Prompt reuse | first `compute_single` clears the cache and re-evaluates the whole prompt | keep the context's token sequence; re-evaluate only after the longest common prefix with the new prompt | multi-turn chat is O(new tokens), not O(history); observable only as speed |
| Metadata keys | ~190 keys, incl. `model-url`, `hf-repo`, `hf-token`, `image`, `mmproj`, `prompt-file`, `slot-save-path`, `out-file`, `tts-*` [we-meta] | **allow-list**: `ctx-size`, `n-predict`, `temp`, `top-k`, `top-p`, `min-p`, `repeat-penalty`, `presence-penalty`, `frequency-penalty`, `seed`, `ignore-eos`, `reverse-prompt`, `n-gpu-layers`; any other key → `invalid_argument` naming it | a key that names a URL or a file is a guest-chosen network or filesystem access the host must not grant |
| Target with `load_by_name` | metadata `n-gpu-layers` | `n-gpu-layers` absent or 0 → CPU; ≥ the model's `block_count` → GPU (needs `--gpu`); in between → `unsupported_operation` | partial offload is not planned; a GPU request is never quietly run on the CPU (DESIGN.md) |
| `load` with builders | GGUF bytes | GGUF bytes, bounded by `Limits.ModelBytes` (64 MiB): test models only | weights of a real model must not transit through guest memory (4 GiB, and a copy) |
| Encoding number | `GGML = 6` | 6, as WasmEdge; 5 refused for ggml | open question Q3 |

Today `nn.loadByName` forces the CPU target and `Named` holds the builders
in memory (`nn/nn.go`); both change: a named entry becomes a *source*
(below), and the target comes from the config.

### A guest's conversation

```
load_by_name_with_config("qwen2.5-0.5b-instruct-q4_0", '{"n-gpu-layers":99,"ctx-size":2048}') -> graph
init_execution_context(graph) -> ctx
set_input(ctx, 0, u8 tensor = UTF-8 prompt, chat template already applied)
set_input(ctx, 1, u8 tensor = '{"n-predict":256,"temp":0.7,"seed":1}')
loop:
    compute_single(ctx)            -> 0 | EndOfSequence | ContextFull | PromptTooLong
    get_output_single(ctx, 0, buf) -> UTF-8 bytes of this token (may be 0 bytes: an incomplete UTF-8 sequence is held back)
get_output(ctx, 1, buf)            -> {"input_tokens":…, "output_tokens":…}
set_input(ctx, 0, next prompt)     # same ctx: the KV prefix is reused
fini_single(ctx)                   # forget the conversation, keep the context
```

The host tokenises (GGUF vocabulary). Templating stays the guest's at first:
`tokenizer.chat_template` is a Jinja template [gguf], and a Jinja subset in
the host is open question Q6. Go guests get `wasinn.Chat` (`Load`, `Prompt`,
`Next() (string, error)`); `examples/chat` streams to stdout.

## WGSL kernels

Model families first: `llama` (SmolLM2, TinyLlama, Llama 3.2) and `qwen2`
(Qwen2.5: Q/K/V biases, the `.bias` suffix of [gguf]). Tensor names
`token_embd`, `blk.N.attn_{norm,q,k,v,output}`, `blk.N.ffn_{norm,gate,up,down}`,
`output_norm`, `output` [gguf]. The reference list is llama.cpp's own WebGPU
shaders [lcpp-webgpu]; ours are written from the ggml semantics, not copied.

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
| `argmax` | greedy sampling on the GPU: 4 bytes read back, not the logits (151 936 floats for Qwen2.5's vocabulary would be ~600 KB per token) | f32 |
| `topk` | top-k (k ≤ 64) on the GPU; top-p, min-p, penalties and the draw on the CPU over k pairs | f32 |

Weights are stored per tensor as two aligned arrays (packed quants as
`array<u32>`, scales and mins as f32 — exact from f16), split into chunks of
at most `maxStorageBufferBindingSize`; a kernel over a split tensor runs once
per chunk (row ranges). Q4_K's 6-bit packed scales are unpacked as
`get_scale_min_k4` does [ggml-quants]. Every kernel gets the two checks
`nn/webgpu` already has: index arithmetic emulated in Go on every platform,
and the real WGSL in Chromium and Firefox through Playwright, against the Go
CPU implementation.

## Limits and quotas

From the WebGPU spec: defaults `maxBufferSize` 256 MiB,
`maxStorageBufferBindingSize` 128 MiB, `maxComputeWorkgroupStorageSize`
16 384 B, `maxComputeInvocationsPerWorkgroup` 256 (128 in compatibility
mode), `maxStorageBuffersPerShaderStage` 8, `maxComputeWorkgroupsPerDimension`
65 535; asking for a limit better than the adapter's rejects `requestDevice`
with `OperationError`; user agents may bin limits into at most 32 buckets
[webgpu-spec]. `shader-f16` enables WGSL `f16`; `subgroups` enables subgroup
operations and exposes `subgroupMinSize`/`subgroupMaxSize` [webgpu-spec].
Browsers: `shader-f16` Chrome 144 (Linux: Intel Gen12+ only), Safari 26, no
Firefox; `subgroups` Chrome 144, no Firefox, no Safari [bcd-feat].

**Design rule: the LLM backend runs within the default limits.** No limit is
raised; `shader-f16` and `subgroups` are requested only when the adapter has
them, and select kernel variants. The fallback adapter stays refused.

| Quota (per container unless said) | Default | Where |
|---|---|---|
| LLM GPU memory: weights + KV caches + scratch, host-computed | **separate from the 256 MiB graph quota; granted explicitly** (`wctr run --gpu --gpu-memory 1GiB`); without it, `load_by_name` of a model larger than the graph quota fails with `too_large` | `load_by_name_with_config`, `init_execution_context` |
| KV cache per context | `2 × block_count × head_count_kv × head_dim × ctx-size × bytes`, from the GGUF keys [gguf]; charged at `init_execution_context` / first `set_input` | context |
| `ctx-size` | ≤ model `context_length` and ≤ 4096 | metadata |
| `n-predict` | ≤ 4096 per prompt | metadata |
| Prompt | ≤ 64 KiB of UTF-8; `PromptTooLong` past the context | `set_input` |
| Prefill chunk | sized so a submit stays well under the 2 s watchdog (DESIGN.md); measured per adapter | backend |
| Contexts per model | 4 | `init_execution_context` |
| Model download | only names in the host's registry; ≤ 2 GiB per file; sha256 verified | fetcher |
| GGUF parser | `tensor_count`, `metadata_kv_count`, string and array lengths (u64 in the file), `n_dimensions` ≤ 4, products of dimensions without overflow, offsets inside the file and aligned [gguf]; fuzzed, corpus committed | `gguf` |
| Metadata JSON | ≤ 4 KiB, allow-listed keys, numeric ranges | `set_input(1)`, config |

Native: the same quotas, the KV cache in Go memory, counted against the
container's memory limit.

## Where the weights live

GitHub Pages refuses sites above 1 GB [gh-pages]: weights never go into the
playground or its `/v2` mirror.

*Probe* (2026-11-02, `Origin: https://go-wasmcontainers.github.io`,
`Range: bytes=0-15`) of
`huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf`:
`302` with `access-control-allow-origin` echoing our origin, `accept-ranges:
bytes`, `x-linked-size: 428730208`, `x-linked-etag` = the file's LFS sha256,
`cache-control: no-store`; the redirect goes to `us.aws.cdn.hf.co`, which
answered `206`, `content-range: bytes 0-15/428730208`,
`access-control-allow-origin: *`; the 16 bytes were `GGUF`, version 3. A
single `bytes=N-M` range is a CORS-safelisted header [fetch], so ranged
fetches need no preflight. The model repository is Apache-2.0 [hf-tree].

Design:

- **Registry, host-side.** `nn.Named` gains a source: `{name, url, size,
  sha256, license}`. The playground ships a small registry (JSON, in Pages);
  the guest can only name an entry. No guest-supplied URL, ever.
- **Fetch in the engine worker** by ranges (16 MiB), resumable, each chunk
  repacked and written straight to its GPU buffer; the whole file is never
  held in the Go heap of the worker (itself wasm32). go-gfx/webgpu needs a
  write from a JS `ArrayBuffer` that bypasses Go memory — to be checked
  against its API, else added there.
- **Cache in OPFS**, keyed by sha256, written as it streams, verified whole
  before it is marked complete; `navigator.storage.persist()` requested. The
  quota counts Cache API, IndexedDB and OPFS together: Chrome up to 60 % of
  the disk per origin, Firefox best-effort min(10 %, 10 GiB per site),
  Safari ~60 %; eviction is LRU under pressure and takes the whole origin
  [mdn-quota]. WebLLM offers the same four choices (`cache`, `indexeddb`,
  `cross-origin`, `opfs`) and defaults to the Cache API [webllm-config].
  OPFS is chosen because a ranged, resumable, partial write is natural there
  and not in the Cache API.
- **Native**: `os.UserCacheDir()/go-wasmcontainers/models/<sha256>.gguf`,
  same verification; `wctr` can pre-fetch.
- **CSP**: the playground's `connect-src` must add `https://huggingface.co`
  and the CDN host the redirect names; DESIGN.md's "no network fetch for
  models" is reversed. See Q1.

## Performance reference

Machine: Apple M4 Max. Model pairs: Qwen2.5-0.5B-Instruct (GGUF Q4_0 and
Q8_0; WebLLM `q4f32_1` and `q4f16_1`) and SmolLM2-360M-Instruct (WebLLM
`q4f32_1`/`q4f16_1`, GGUF Q8_0). Formats differ (ggml blocks vs MLC group
quantisation): the table says so per row.

| Row | Tool | Setting |
|---|---|---|
| CPU reference | llama.cpp, tagged release, CPU build | `llama-bench -t 1 -ngl 0 -p 128 -n 128`, pp and tg tokens/s |
| Browser reference | WebLLM, same Chromium as our e2e | decode tokens/s, prefill tokens/s, same prompt |
| Native reference (context only) | llama.cpp Metal | `-ngl 99` |
| Ours, native | `nn` ggml CPU backend, one core | same prompt, greedy |
| Ours, browser | `nn/webgpu` ggml backend, Metal adapter, Chromium and Firefox | same |

Published anchor (not our machine): on an M3 Max in Chrome Canary 133,
WebLLM decoded Llama-3.1-8B at 41.1 tok/s and Phi-3.5-mini at 71.1 tok/s,
71.2 % and 79.6 % of native MLC-LLM [webllm-paper]. No published single-core
llama.cpp figure for these models was opened; it is measured in step 0.

Targets: native single core ≥ llama.cpp `-t 1` (slower is an open defect);
browser decode ≥ 50 % of WebLLM on the same model at step 7, ≥ 80 % after
step 8. Measured in interleaved rounds, never above a 1-min load of ~8,
allocation counts recorded alongside.

## Plan

Each step: a PR, tests natively and in Playwright (Chromium, Firefox), README
and docs, a release.

| # | Step | Acceptance test |
|---|---|---|
| 0 | Baselines | The table above filled for both models on the M4 Max, with load average, tool versions and commands committed under `docs/` |
| 1 | `gguf`: bounded parser over an `io.ReaderAt` (file or HTTP ranges) | Reads the header, all metadata and tensor infos of the two models from their first MiBs over HTTP; `FuzzGGUF` with a committed corpus; each bound has a refusal test (huge counts, overflowing dimensions, offset past the end, misalignment) |
| 2 | Dequantisation in Go: F32, F16, Q4_0, Q8_0, Q4_K, Q6_K (and the types step 1 finds in the chosen files) | Bit-identical f32 to ggml's `dequantize_row_*` on every tensor of both models (reference dumps produced from the tagged ggml) |
| 3 | Tokenizers: `gpt2` BPE (Qwen2.5, SmolLM2) and `llama` SentencePiece (TinyLlama) from `tokenizer.ggml.*` | Token ids identical to llama.cpp's tokenizer on a committed multilingual corpus, both directions |
| 4 | CPU forward pass, `llama` and `qwen2` (the oracle port), KV cache, sampling | Logits within 1e-3 relative of llama.cpp's for a fixed prompt; 64 greedy tokens identical; tok/s vs llama.cpp `-t 1` in the README |
| 5 | wasi-nn: encoding 6, the six functions, errnos 100–103, metadata allow-list, per-context KV cache, prefix reuse; `wasinn.Chat`, `examples/chat` | A Go guest streams a greedy answer identical to step 4; `FuzzHostCalls` covers the new functions; refusals tested (unknown key, `model-url`, partial `n-gpu-layers`, `ContextFull`) |
| 6 | Registry, fetcher, cache (OPFS, native dir) | Second load makes zero model requests; a corrupted cache file is detected and refetched; an interrupted download resumes; a size or sha256 mismatch is refused |
| 7 | WGSL kernels, f32 only, within default limits | Each kernel: emulated test + browser test vs step 4; end to end, 32 greedy tokens identical to the CPU on Chromium/Metal and Firefox; GPU memory charged equals the host's computed sum |
| 8 | f16 and subgroup variants, prefill tiling, tuning | Same token test with variants on; the performance table updated against WebLLM |
| 9 | Playground demo, landing, docs | A chat container in the playground streams tokens; published pages checked (200, new text) |

## Open questions for the user

- **Q1 — Network.** May the playground fetch weights from Hugging Face
  (CSP change, a third-party CDN whose redirect host may change)? The
  alternative is a mirror of our own — where (GitHub Release assets were not
  evaluated for CORS and ranges)?
- **Q2 — GPU memory.** Is a separate, opt-in LLM memory grant
  (`--gpu-memory`) acceptable, and what ceiling? Order of magnitude from
  WebLLM at a 4 096-token context: 944.62 MB for Qwen2.5-0.5B and
  1 629.75 MB for Qwen2.5-1.5B at q4f16_1 [webllm-config]; ours is computed
  from the GGUF keys in step 1.
- **Q3 — Encoding number.** `ggml` is 6 in WasmEdge [we-types] and 5 in the
  WIT's enum order [nn-wit]. We propose 6 (the witx world is WasmEdge's).
- **Q4 — Compatibility target.** Must LlamaEdge's own guests (Rust, built
  for WasmEdge) run unchanged, or only the ABI shape? The former means
  testing against their binaries and accepting their metadata subset.
- **Q5 — First models.** SmolLM2-135M/360M (Apache-2.0), Qwen2.5-0.5B
  (Apache-2.0, per their model-card tags), then Qwen2.5-1.5B; Llama 3.2 1B
  (licence not checked for this page) — include it?
- **Q6 — Chat template.** Guest-side (as proposed), or a Jinja subset in the
  host driven by `tokenizer.chat_template`?
- **Q7 — Native GPU.** CPU only natively until a pure-Go WebGPU (gogpu,
  DESIGN.md) is evaluated — confirm.

## References (opened)

- [nn-witx] WebAssembly/wasi-nn `wasi-nn.witx`, main (71320d95b8): https://github.com/WebAssembly/wasi-nn/blob/main/wasi-nn.witx
- [nn-wit] WebAssembly/wasi-nn `wit/wasi-nn.wit`, `wasi:nn@0.2.0-rc-2024-10-28`: https://github.com/WebAssembly/wasi-nn/blob/main/wit/wasi-nn.wit
- [we-func] WasmEdge `plugins/wasi_nn/wasinnfunc.cpp`, master (673031ab7e): https://github.com/WasmEdge/WasmEdge/blob/master/plugins/wasi_nn/wasinnfunc.cpp
- [we-types] WasmEdge `plugins/wasi_nn/wasinntypes.h`
- [we-design] WasmEdge `plugins/wasi_nn/DESIGN.md`
- [we-meta] WasmEdge `plugins/wasi_nn/GGML/metadata/metadata_parser.cpp`
- [we-compute] WasmEdge `plugins/wasi_nn/GGML/compute/compute_engine.cpp`
- [we-infer] WasmEdge `plugins/wasi_nn/GGML/compute/inference_manager.cpp`
- [we-out] WasmEdge `plugins/wasi_nn/GGML/core/output_generator.cpp`
- [gguf] ggml `docs/gguf.md`, master (facdc44a67): https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
- [ggml-common] ggml `src/ggml-common.h` (`block_q4_0`, `block_q8_0`, `block_q4_K`, `block_q6_K`, `QK_K` 256)
- [ggml-quants] ggml `src/ggml-quants.c` (`dequantize_row_q4_0`, `_q8_0`, `_q4_K`, `get_scale_min_k4`)
- [lcpp-webgpu] llama.cpp `ggml/src/ggml-webgpu/` (directory, `wgsl-shaders/`, `ggml-webgpu.cpp`: required features, binding-size checks), master (1b6935b227)
- [lcpp-ops] llama.cpp `docs/ops/WebGPU.csv` (MUL_MAT q4_0/q4_K/q8_0, FLASH_ATTN_EXT supported)
- [lcpp-build] llama.cpp `docs/build.md`, sections WebGPU, WASI, wasi:webgpu
- [webllm-paper] Ruan et al., "WebLLM: A High-Performance In-Browser LLM Inference Engine", arXiv:2412.15803v2 (13 April 2026), Table 1: https://arxiv.org/html/2412.15803v2
- [webllm-config] mlc-ai/web-llm `src/config.ts`, main (f1ecbaa916): `prebuiltAppConfig`, `CacheBackend`
- [webllm-engine] mlc-ai/web-llm `src/engine.ts` (`ShaderF16SupportError`, the 1 GB binding-size warning)
- [tvm-webgpu] apache/tvm `web/src/webgpu.ts` (`detectGPUDevice` required limits and features)
- [ort-contrib] onnxruntime `docs/ContribOperators.md` (MatMulNBits, GroupQueryAttention), main (c7bf7939b5)
- [ort-kernels] onnxruntime `docs/OperatorKernels.md` (WebGpuExecutionProvider rows)
- [tjs-webgpu] transformers.js, "Running models on WebGPU": https://huggingface.co/docs/transformers.js/guides/webgpu
- [webgpu-spec] gpuweb `spec/index.bs` (limits, `"shader-f16"`, `"subgroups"`, `requestDevice`), and W3C CR Draft 15 September 2026: https://www.w3.org/TR/webgpu/
- [bcd-feat] mdn/browser-compat-data `api/GPUSupportedFeatures.json` (`feature_shader-f16`, `feature_subgroups`)
- [fetch] WHATWG Fetch, CORS-safelisted request-header (`range`): https://fetch.spec.whatwg.org/#cors-safelisted-request-header
- [mdn-quota] MDN, Storage quotas and eviction criteria: https://developer.mozilla.org/en-US/docs/Web/API/Storage_API/Storage_quotas_and_eviction_criteria
- [gh-pages] GitHub Docs, GitHub Pages limits: https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits
- [hf-tree] Hugging Face API, file trees of `Qwen/Qwen2.5-0.5B-Instruct-GGUF` and `onnx-community/Qwen2.5-0.5B-Instruct` (sizes), model card tags (licence), 2026-11-02
- go-wasmcontainers/gpu `README.md`, `docs/DESIGN.md`, `nn/nn.go`
