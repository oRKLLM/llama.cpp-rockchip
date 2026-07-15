# Async cross-stream submit path in ggml-ork — WIP recovery doc

**Branch:** `feat/ork-async-stream` (fork, off `feat/attn-npu`). ork-driver submodule at v0.7.0 (`22b7480`).
**Goal:** let one stream's NPU graph run while another stream's CPU work proceeds — the measured-free
CPU‖NPU overlap (`overlap_prof`, hidden~100%), lifted to the ggml-ork backend level so a pipelined
speculative/dflash decode loop can use it.

## What the reconnaissance established (2026-07-15)
- **The async submit primitive ALREADY EXISTS** in ork-driver (`ork_mm_*_async` + `ork_async_wait`,
  worker-thread launch) and PATH-b already overlaps NPU `run_stream_i8` with a CPU MUL_MAT_ID subgraph via
  `std::thread` inside one `graph_compute`. So submit-async is NOT the gap.
- **Two ork backends already coexist** (draft+target each get their own `ggml_backend_ork_context` + `ork_npu`
  fd). Caches are per-context (`backend->context`, not `g_ork_ctx`).
- **True CONCURRENT cross-stream execution is unsafe today** due to shared mutable global state:
  1. `g_attnp` (attention densify/QK^T/softmax scratch) + softmax function-`static`s — clobber under concurrent
     attention. (ATTENTION ONLY.)
  2. `g_ork_ctx` read in `supports_op` (last-init-wins) — PLANNING-time; benign if draft/target share qbits/config.
  3. **Single hardware queue** — NPU submits from two streams MUST serialize (CPU halves overlap; NPU halves cannot).
- `graph_compute` is fully synchronous; sched relies on it (synchronize=NULL, no events). Overlap must be BETWEEN
  two contexts' graph_computes driven by an application loop, NOT inside sched.
- **No concurrent consumer exists** — dflash/spec loop is strictly sequential (draft gen K → target verify batch).
  The batched-verify forward is `llama_decode(ctx_tgt, vb)` (examples/dflash/dflash.cpp:180).

## Design (slices)
**Slice 1 (THIS): backend async graph-compute + synchronize + single-NPU-queue mutex, gated `ORK_ASYNC`.**
- `g_npu_queue_mu` (process-global std::mutex): the async worker holds it around `graph_compute` so cross-context
  NPU work serializes on the one hardware queue. Sync path (make test) does NOT take it → zero behavior change.
- `ggml_backend_ork_graph_compute_async(backend, cgraph)`: spawns a per-context `std::thread` running the existing
  `graph_compute` body (reuses ALL per-node logic), returns immediately. One in-flight job per backend.
- `ggml_backend_ork_synchronize(backend)`: joins the worker (full memory barrier).
- Default OFF (functions are opt-in; nothing calls them unless the consumer/harness does) → `make test` unaffected.
- Validate with a standalone harness: stream A = NPU matmul graph run async; stream B = CPU work on main thread;
  assert wall ~= max(A_npu, B_cpu) (not sum) AND A's output correct. On RK3588.

**Slice 2+ (DEFERRED, scoped): per-context attention scratch (move `g_attnp` + softmax statics into the ctx struct)
for real transformer graphs; then wire into a PIPELINED dflash decode loop (target verify on a worker thread ‖
draft generate on main) for end-to-end tok/s.** Slice 1 uses matmul-only graphs, avoiding the attention scratch.

## State
- [x] Recon complete. Branch + WIP doc created.
- [x] Slice 1 backend code: `ggml_backend_ork_graph_compute_async` + `ggml_backend_ork_synchronize` +
      `g_npu_queue_mu` (single-queue serialize) + worker big-core pin (ORK_ASYNC_BIG). ggml-ork.{cpp,h}.
- [x] Slice 1 harness `tools/ork_xstream.cpp` + board validation (RK3588). Builds clean, NPU output bit-correct.
- [ ] Commit on feat/ork-async-stream.

## RESULTS (2026-07-15, board .236, ork_xstream, M=64 K=4096 N=4096)
The async cross-stream path WORKS and the NPU output is bit-correct through it (maxerr 0.000). Overlap is
free — but ONLY with core partitioning; the naive shared-core config contends:

| config | npu_solo | cpu_solo | overlap | hidden | note |
|---|---|---|---|---|---|
| both streams on big cores (default) | 2364 | 1774 | 3464 | 38% | big-core OVERSUBSCRIPTION: the NPU matmul's own quant/dequant + multi-core dispatch helpers already want the 4 A76 cores; a concurrent CPU stream fights them |
| DISJOINT: worker→big(4-7), draft→little(0-3), reps=8 | 3222 | 6965 | 7112 | 95.5% | free — draft (little) is the long pole, NPU fully hidden |
| DISJOINT, reps=2 (router-sized draft) | 3321 | 1746 | 3395 | 95.8% | **zero-time router confirmed: overlap ≈ npu_solo, draft fully hidden** |

**KEY FINDING:** the bare-submit `overlap_prof` (~100% free) was optimistic — the REAL ggml matmul is
CPU-core-hungry (quant/dequant + multi-core NPU dispatch, big-core-pinned + spin-polling). On a 4-big-core
SoC a concurrent CPU stream OVERSUBSCRIBES the big cores → overlap not free (hidden 38%). Fix = CORE
PARTITIONING: NPU worker on the A76 big cores, draft/router on the A55 little cores (disjoint) → overlap
free again (hidden ~96%). Tradeoff: the draft on little cores is ~4× slower, so this is ideal for a TINY
draft/router (fully hidden regardless) and needs a smarter split (e.g. 2 big + 2 big, or big+little mix) for
a heavy draft-model forward. The MECHANISM is proven; the core-allocation policy is the tuning knob.

## Design implication for the consumer (pipelined dflash/spec loop, Slice 3)
Pin the target-verify NPU worker to big cores, the draft/routing to little cores. Then target-verify(NPU) ‖
draft-generate(CPU-on-little) overlaps free. Serialize any NPU submits from BOTH streams on g_npu_queue_mu
(single hardware queue).

## Board/ops
- ork-driver primitives on `feat/npu-doorbell` (doorbell/NONBLOCK); NOT needed for Slice 1 (thread-based reuses
  existing blocking run funcs). Submodule stays at v0.7.0.
- Standard board safety: timeout every NPU cmd; SIGTERM not kill -9; wedge -> `ssh board 'sudo reboot'`.
