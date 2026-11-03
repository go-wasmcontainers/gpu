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
| batchcompute (read-back, refusals, quota, grant), longest submit | 8.4 ms (13.1 ms stream-only, victim run) | 198.6 ms |
| 120-submit round-trip bench, longest submit | 7.7 ms | 202.0 ms |
| LLM (go-wasmcontainers/llm v0.1.0, `examples/generate`, SmolLM2-135M Q4_0, 183 submits), longest submit | 30.7 to 39.8 ms | not measured (the LLM e2e is a Chromium lane); a submit costs at least the 200 ms floor of #25 |

The longest legitimate submit measured is **202 ms (Firefox, the floor of
#25)**. 250 ms would leave 48 ms (24 %), too little for a loaded machine, so
the default is **500 ms**: 2.5 times the Firefox figure and 12 times the
Chromium LLM figure. The hard cap stays 2 s, 4 times the budget and 10 times
the slowest legitimate submit.

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

See the table in the pull request that introduced this note and the README's
browser table; the e2e tests are `TestBatchSanction` (hostile shader aimed at
3 s against a victim container submitting every 20 ms for 25 s, in the
`combined` and `split` shapes), `TestBatchDeadline`, `TestBatchComputeQuota`
and the zero-sanction assertion of every other batch and LLM test.
