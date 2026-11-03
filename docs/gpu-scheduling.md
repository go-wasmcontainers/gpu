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
the first stall by cancellation.** Layers 1 to 3 below bound everything else.

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

## What this does not do: the first stall (layer 4, open)

Nothing above bounds the first stall. A hostile shader that fits the quota can
loop for seconds; the host sees it only when the budget expires, destroys the
device, and the GPU runs the rest of it anyway. Every other container waits
for the whole shader: **the victim's worst-case added latency is the hostile
shader's remaining length, not the budget** (measured below). The only way to
bound it is to rewrite the WGSL before it is compiled, so that every loop
carries an iteration counter and exits at a bound (loop instrumentation). That
is layer 4, out of scope here, and open: it needs a WGSL parser and a rewriter
that preserve semantics, and it still cannot bound a loop-free but huge
dispatch (layer 3 does that).

## Acceptance measurements

The e2e tests are `TestBatchSanction` (a hostile container whose shader is
aimed at 3 s, against a victim container submitting every 20 ms for 25 s, in
three runs: `control` with the budget and cap raised to 60 s so nobody is
sanctioned, then `combined` and `split`), `TestBatchDeadline`,
`TestBatchComputeQuota`, and the zero-sanction assertion of every other batch
and LLM test. Apple M4 Max, Chromium 151 and Firefox 159 (headed). The
victim's **worst submit**:

| run | Chromium 151 | Firefox 159 |
|---|---|---|
| control (no sanction: the victim waits the shader out) | 7,462 ms (load 6 to 8) | 3,189 and 7,109 ms |
| combined (creation and dispatch in one frame; sanction at the 2 s cap) | 3,052 ms (load 4 to 5, first build); 4,175 ms (load 6 to 8) | 3,475 and 3,161 ms |
| split (dispatch in a frame of its own; sanction at the budget) | 2,949 ms (load 4 to 5, budget then 250 ms); 5,066 ms (load 6 to 8) | 9,301 and 8,906 ms: not comparable, the shader itself ran about 9 s (see below) |

What the numbers say: the victim's latency is the hostile shader's remaining
length, and the shader's length changes with whatever else uses the GPU (the
same "3 s" shader ran 3.0 s or 7.5 s in Chromium, on this shared machine).
On Firefox a submit resolves in 200 ms steps, so the hostile's calibration
(which stops at 40 ms of work to stay far under the budget) cannot see work
below 200 ms and overshoots: its "3 s" split shader ran about 9 s. The e2e
therefore logs the victim's worst latency and only asserts that it is below
the run's length.
Sanctioning does not shorten it, and does not lengthen it; what it changes is
that the victim is **never sanctioned itself** (0 errors in every run, one
sanction in the whole module), that the hostile container gets no GPU after
its first overrun (the next `context_create` answers *revoked*, asserted),
and that the stall happens once. Every other batch test, the LLM tests
(`TestLLMOracle`, `TestLLMTrace`, `TestLLMShaders`) and the existing e2e pass
with **zero sanctions** on Chromium; the batch tests pass with zero sanctions
on Firefox. Not measured: Safari (its e2e lane, go-wasmcontainers/gpu#29, has
not landed).
