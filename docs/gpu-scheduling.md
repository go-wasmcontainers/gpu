# GPU time: scheduling, budget, quota (go-wasmcontainers/gpu#26)

A container's GPU work is not under the host's control once WebGPU has it.
This note says what the host does about that, why, and what it does not do.

## Threat model

- **In the browser** all the containers of a tab belong to one student on their
  own machine. The aim is robustness: a runaway shader in one container must
  not make the student's other containers fail, and must not hang the tab. It
  is not a defence against another user.
- **Natively** (`wctrd` on a shared GPU server, deferred) the same containers
  would belong to different users and the problem is cross-user denial of
  service. The design does not assume the browser case: nothing below depends
  on the containers sharing an owner, and the sanction is per container.

## What the platform offers (bibliography)

Opened for this note:

- **WebGPU** (W3C, <https://www.w3.org/TR/webgpu/>): `GPUDevice.destroy()` is
  "unmap() all GPUBuffers from this device", then lose the device with reason
  "destroyed"; "outstanding asynchronous operations will fail", and a note says
  implementations *can* abort outstanding asynchronous operations immediately.
  Nothing there says submitted GPU work is cancelled. §3.5.2 (devices): a lost
  device and every object on it become unusable. §2.1.9 (denial of service):
  an implementation "may set up a watchdog timer" so that an application does
  not make the GPU unresponsive for more than a few seconds, "similar to
  WebGL". There is **no API to cancel or preempt submitted work**.
  `onSubmittedWorkDone()` is the only completion signal (its own definition was
  not in the part of the page read; its ordering rule, "p1 must settle before
  p2" for one queue, was, in §3.2.2).
- **Chromium's GPU watchdog** (`gpu/ipc/service/gpu_watchdog_thread.h`): it
  "deliberately crashes" the GPU process when a watched thread does not answer
  within its timeout (`DeliberatelyTerminateToRecoverFromHang`; the header's
  `kUnreasonableTimeoutDelay` is 5 s). So a hang is answered by the browser
  killing the GPU process, which loses *every* device of the page, the
  student's other containers' included.
- **wgpu's Metal backend** (`wgpu-hal/src/metal/command.rs`, the part
  returned): no timeout, cancel or abort for a command buffer after submit is
  visible in it. The page was truncated; this is "none seen", not a proof.

Not opened, so not cited: Apple's `MTLCommandBufferError.timeout` page (the
fetch returned no body) and Firefox's own long-work handling beyond what its
behaviour shows below. Measured, not read: on Metal, `destroy()` leaves the
work running (go-wasmcontainers/gpu#26: Chromium 151 started the next
container 3.7 s late; Firefox 159's next container hit its own 2 s deadline).

**Conclusion: no way to cancel submitted work exists, so no design can bound
the first stall by cancellation.** Layers 1 to 3 bound everything else; layer 4
bounds the first stall by not letting a loop run long enough to need cancelling.

## Layers

### 1. Fair scheduling

One submit is in flight in the whole module, not per device: work of two
devices queues behind each other on one GPU, so the GPU is what is
serialised. A container's guest is blocked in `submit` until its work is done
(the host waits for `onSubmittedWorkDone`), so a context has one submit in
flight by construction. The waiting submits of different containers are served
round-robin (`gpubatch/sched.go`), so no container can fill the queue ahead
of another. A submit's clock starts when it gets its turn, never when it was
queued: a container does not pay for the turns of others.

### 2. A measured budget and a sanction

Each submit is timed from the start of its turn to the completion of its work
(read-back included). Two limits, both from the start of the turn:

| limit | default | applies to |
|---|---|---|
| `Limits.Budget` | **500 ms** | a frame that only streams commands (steady state) |
| `Limits.GPUTime` | **2 s** (the hard cap) | a frame that creates objects: shader and pipeline compilation have no steady-state size |

On overrun the host: destroys the container's device (every context is lost),
revokes its GPU grant for the rest of its life, answers the submit *timeout*
(errno 8), then *revoked* (errno 12) to every later call except
`context_destroy` (so a new `context_create` or `caps` is refused), reports
`gpu-sanction` on the engine's event bus (`wctr events`; the attributes carry
`submit_ms`, `budget_ms` and the kind of frame) and keeps it for `wctr inspect`
(`State.Sanction`). The revocation is kept by container ID in the module, so a
restart of the same container is still revoked, until the engine itself
restarts. **A container can stall the GPU at most once.**

Because destroy() does not cancel, the GPU still holds the destroyed device's
work. The module keeps the turn until a trivial submit of its own, on a device
the host owns (`Browser.Drain`), has completed behind it (at most
`Limits.DrainTime`, 30 s), and only then serves the next container. That is
what removes the false positive of #26: an innocent container's budget starts
on an idle GPU.

Budget, chosen from measurements (Apple M4 Max, `GPU_E2E` lanes):

| workload | Chromium 151 | Firefox 159 |
|---|---|---|
| batchcompute (read-back, refusals, quota, grant, shaders), longest submit | 8.4 to 14.8 ms | 198.3 to 198.6 ms |
| 120-submit round-trip bench (steady stream frames), longest submit | 7.7 to 9.9 ms | 202.0 ms; **226.4 ms** in a run at load 8 |
| victim container (a 4 KiB compute and read-back every 20 ms for 25 s), longest stream-only submit | 13 to 66 ms | 200.3 to 205.3 ms |
| LLM (go-wasmcontainers/llm v0.1.0, `examples/generate`, SmolLM2-135M Q4_0, 183 submits), longest submit | 30.7 to 43.4 ms | not measured (the LLM e2e is a Chromium lane); a submit costs at least the 200 ms floor of #25 |

The longest legitimate submit measured is **226 ms (Firefox 159: the 200 ms
floor of #25, plus load)**. 250 ms would leave 24 ms (10 %): a false positive
waiting for a busy machine. The default is therefore **500 ms**: 2.2 times the
Firefox maximum and 11 times the Chromium LLM maximum. The hard cap stays 2 s,
4 times the budget and 9 times the slowest legitimate submit. Both are
configurable (`Limits.Budget`, `Limits.GPUTime`).

### 3. Static admission bounds

Next to `wire.CheckRanges`, `wire.CheckInvocations` (go-gfx/webgpu v0.11.0)
refuses, before anything is replayed, a dispatch or a stream asking for more
invocations than the quota. A dispatch counts as its three workgroup counts
times the device's `maxComputeInvocationsPerWorkgroup` (256 here: the capset
reports the smaller of the adapter's limit and WebGPU's default), which bounds
the WGSL's `@workgroup_size` without parsing it. Defaults: **2^26 invocations
per dispatch, 2^30 per stream**. Measured on the LLM: largest dispatch
12,582,912 (the quota is 5.3 times that), largest stream 321,212,416 (3.3
times). A refused stream answers *refused* (errno 6), loses its context and is
not a sanction. `FuzzCheckInvocations` covers the check. A
`DispatchIndirect` takes its counts from GPU memory and is not counted; only
the time budget bounds it.

### 4. Loop budget: the first stall, bounded (layer 4)

Nothing above bounds the first stall: a hostile shader that fits the quota can
loop for seconds, the host sees it only when the budget expires, and the GPU
runs the rest of it anyway. Layer 4 does not cancel; it makes the loop stop by
itself. Every guest WGSL module with a compute entry point is rewritten before
`createShaderModule` with go-gfx/wgsl `transform.LoopBudget` (v0.2.0, applied
in `go-gfx/webgpu/batch`, `DeviceBackend.SetLoopBudget`): each outermost loop
nest of each function activation counts its iterations and, past the budget,
stores 1 into a host-owned flag and leaves the loop.

- **Parse only.** The host calls `wgsl.Parse` and the transform, never
  `Validate` (its checker still has false rejects). WGSL that does not parse
  reaches the browser unchanged and the browser reports its own error: nothing
  is refused by the host. Such a shader is bounded by layer 2 only. If the
  browser refuses the rewritten text but accepts the original, the original is
  kept and counted (`Stats.LoopFallbacks`, zero in every run below).
- **Budget per dispatch.** `B = Work / invocations` of the dispatch (workgroup
  counts times 256), rounded down to a power of 4, at least 1024 and at most
  2^24 (`Browser.LoopWork`, default 2^36, `LoopMin`). A big dispatch gets short
  loops, a small one long loops; the cap exists because a one-workgroup
  dispatch would otherwise get billions of serial iterations (the first
  Firefox run of the hostile test found it). A `DispatchIndirect` is budgeted
  as 2^26 invocations. The budget is a `var<uniform> u32` bound with a dynamic
  offset into a host buffer; no state is written per dispatch.
- **Reserved slot.** Bind group 3 (`maxBindGroups - 1`; the device asks for no
  raised limit) belongs to the host: binding 0 is the flag, binding 1 the
  budget. A pipeline layout with more than 3 bind group layouts, a WGSL module
  with a binding in `@group(3)`, or a `SetBindGroup` of index 3 or more is
  refused with errno **13, *reserved*** (the context is lost, the grant is not
  revoked). The host pads the layouts of rewritten pipelines with empty groups
  and sets groups 1 to 3 itself, once per pass, and the flag's copy rides the
  stream's own submit.
- **Truncation.** After a stream that ran a rewritten pipeline the host reads
  the flag. If it is set the submit answers errno **14, *truncated*** (a
  truncated result is never returned), the context is lost, and the container is
  sanctioned exactly like a timeout: grant revoked, `gpu-sanction` event with
  `cause=loop-budget` (`cause=time-budget` otherwise). There is nothing to
  drain: the work is already off the GPU.
- **Work is per GPU.** 2^36 was measured on an Apple M4 Max. On a software
  adapter (SwiftShader in CI) the same loop is about a thousand times slower,
  so a hostile shader fits under the budget and is cut by the 2 s cap (layer 2)
  instead; set `Browser.LoopWork` lower for such an adapter.
- **Not covered.** Straight-line code (bounded already) and the dispatch size
  (layer 3). A module that also has a `@vertex` entry is not rewritten (a
  vertex stage cannot write storage). The older `wasi_webgpu` module (no
  scheduler, no budget) and the host's own wasi-nn kernels do not go through
  `gpubatch` and are not rewritten.

**Choosing `Work` from measurement.** `internal/cmd/loopsweep`
runs the LLM tests (`TestLLMOracle`, `TestLLMTrace`, `TestLLMShaders`:
SmolLM2-135M Q4_0, 50 shader modules, largest dispatch 12,582,912 invocations,
largest stream 321,212,416) with `Work = 2^k` and the floor at 1, so nothing masks
a truncation. They pass with zero truncations down to **2^28** and fail at
2^27: the most expensive legitimate dispatch (invocations times loop
iterations) is between 1.3 and 2.7 x 10^8. The default **2^36 is 256 times the
smallest passing work**; every other e2e (batch, engine, quota, the wasi-webgpu
backend) has zero truncations (the batchcompute shaders have no loops). The hostile shader at the largest dispatch the
quota allows (about 6.7 x 10^7 invocations, B = 1024) is cut after 26 to 37 ms
in Chromium 151 and 100 ms in Firefox 159 (Apple M4 Max).

**Cost.** Rewriting (parse, transform, print, in the browser's WebAssembly):
**0.6 ms per module on average** (LLM: 50 modules, 30 ms in all, 35 of them
rewritten; batch suite: 26 modules, 10 ms in Chromium, 13 to 15 ms in
Firefox). Per dispatch, the counters and the extra `SetBindGroup`s cost LLM
decode **about 10 % of its tokens per second** (Chromium 151, SmolLM2-135M Q4_0, 32
tokens, interleaved off/on, 1-minute load 5 to 6: 248 tok/s without layer 4
against 223 with, 12 samples each; an earlier 16-sample series gave 260 against
229). Layer 4 can be turned off (`Browser.NoLoopBudget`, `?noloop=1` in the
e2e page) for a measurement or a trusted workload; that is a trade, not a
default.

## Acceptance measurements (layers 1 to 3; layer 4 at the end)

The e2e tests are `TestBatchSanction` (a hostile container whose shader is
aimed at 3 s, against a victim container submitting every 20 ms for 25 s, in
three runs: `control` with the budget and cap raised to 60 s so nobody is
sanctioned, then `combined` and `split`), `TestBatchDeadline`,
`TestBatchComputeQuota`, and the zero-sanction assertion of every other batch
and LLM test. Apple M4 Max, Chromium 151 and Firefox 159 (headed). The
victim's **worst submit**:

| run | Chromium 151 | Firefox 159 |
|---|---|---|
| control (no sanction: the victim waits the shader out; shader aimed at 4.5 s) | 4,516 ms | 2,076 ms |
| combined (creation and dispatch in one frame; sanction at the 2 s cap; aimed at 4.5 s) | 4,526 ms | 3,876 ms |
| split (dispatch in a frame of its own; sanction at 500 ms; aimed at 3 s) | 3,031 ms | 3,176 ms |

These are runs at a 1-minute load of 3 to 4. The victim's median submit was
1.2 to 1.3 ms in Chromium and 175 to 274 ms in Firefox (its 200 ms floor).
The shader's real length varies with the GPU's other users and, in Firefox,
with a calibration that sees work only in 200 ms steps (a "3 s" shader ran
1.3 s once, 9 s at load 8): the e2e therefore logs the victim's worst
latency and only asserts that it is below the run's length, and the combined
run aims at 4.5 s so that it passes the 2 s cap.

What the numbers say: the victim's latency is the hostile shader's remaining
length. Sanctioning does not shorten it, and does not lengthen it; what it changes is
that the victim is **never sanctioned itself** (0 errors in every run, one
sanction in the whole module), that the hostile container gets no GPU after
its first overrun (the next `context_create` answers *revoked*, asserted),
and that the stall happens once. Every other batch test, the LLM tests
(`TestLLMOracle`, `TestLLMTrace`, `TestLLMShaders`) and the existing e2e pass
with **zero sanctions** on Chromium; the batch tests pass with zero sanctions
on Firefox.

**Safari 27.2 (safaridriver lane, #29): partly validated.** The legitimate
workloads are never sanctioned (batchcompute, refusals, quota, grant, the
compute-quota test: longest submit 11.7 ms, zero sanctions). The hostile run
is not validated, and does not pass `TestBatchDeadline`: WebKit has its own
watchdog. With the host's budget raised to 60 s, Safari itself destroyed the
hostile device after 3.25 s of shader ("object destroyed", the host did
nothing), and every later `requestAdapter` answered null (the next containers
got *unsupported*). With the default budget, the sanction at 2 s was followed
by the same refusals, the drain probe was refused too (it retries until
`DrainTime`), and the next container that did get a device was itself
sanctioned. The load average was 12 to 19 during those runs, so the timing is
not clean, but the adapter refusal is. On 2026-11-04 safaridriver could not
start a Safari session at all ("timed out while connecting to a Safari
instance"), so the run could not be repeated at a low load. It is tracked as an open defect
(Safari's behaviour after a hung shader), not claimed as working; the Safari
victim-latency figure is therefore **not measured**.

### Layer 4: the same hostile/victim test, before and after

The same `TestBatchSanction`, same machine (Apple M4 Max, 1-minute load 5 to 6),
the victim's **worst submit**. *Before* is the control run (no layer 4, budget
and cap at 60 s: nobody is sanctioned and the victim waits the shader out; shader
aimed at 4.5 s); *after* is the default (layer 4 on) with the hostile cut and
revoked:

| run | Chromium 151 before | Chromium 151 after | Firefox 159 before | Firefox 159 after |
|---|---|---|---|---|
| combined (creation and dispatch in one frame) | 4,574 ms | **40.8 ms** | 2,176 ms | **277.5 ms** |
| split (dispatch in a frame of its own) | 3,031 ms (v0.10.0) | **13.2 ms** | 3,176 ms (v0.10.0) | **280.1 ms** |

Firefox's victim median is 177 ms (its 200 ms floor, #25), so its added latency
is about 100 ms; Chromium's median is 1.2 to 1.4 ms. Both hostile containers
answer *truncated* (errno 14) once, are revoked (the next `context_create` answers
*revoked*), the victim is never sanctioned (0 errors), and the engine event
carries `cause=loop-budget`. Zero truncations and zero fallbacks in every other
test: on Chromium the whole e2e (`TestBatch*`, `TestContainersOnWebGPU`, `TestLLMOracle`, `TestLLMTrace`, `TestLLMShaders`, `TestLLMDecodeBreakdown`, `TestBrowserCapabilities`, `TestWebGPUBackend`), on
Firefox the batch tests, `TestBrowserCapabilities` and `TestWebGPUBackend`.
**Safari is not measured**: automation is disabled on this Mac (safaridriver
cannot start a session, #32); the Safari acceptance stays open.
