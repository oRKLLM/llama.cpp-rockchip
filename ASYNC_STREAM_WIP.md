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

## SLICE 2 (per-context attention scratch) — done + a reframing board finding (2026-07-15)
Moved the fused-attention scratch off the process globals into the per-context struct:
- `g_attnp` (attention pool: NPU scratch weights + QK^T/scores host buffers) -> `ctx->attnp`.
- the 13 softmax function-`static`s -> `ctx->attn_sm` (aliased as C++ references so the 42 downstream `sm_*`
  uses stay byte-identical + warm-reused). `bmm_fp16` was already safe (stack-local + per-ctx `ctx->npu`).
- freed per-context in `ggml_backend_ork_free` (+ join any in-flight async worker first).
- Compiles; matmul output still bit-correct (`ork_xstream` maxerr 0.000) => behavior-preserving.

**REFRAMING FINDING (board):** the first concurrency test drove TWO ork contexts' attention graphs on two
threads with NO serialization (raw `ggml_backend_graph_compute`). Result: **level-0 IOMMU translation fault +
soft reset + wedge** (needed a reboot). So the binding constraint for concurrent real graphs is NOT the CPU
scratch — it's the **single NPU hardware queue**: two contexts' submits cannot hit it concurrently. So:
- Per-context scratch (this slice) removes the CPU-side shared-state hazard and is the PREREQUISITE for a
  future fine-grained design, but by itself does NOT make two ork graphs safe to run concurrently.
- Real cross-stream concurrency REQUIRES serialized NPU submits. The Slice-1 coarse `g_npu_queue_mu` provides
  this (whole graph_compute serialized) — but then two ork graphs don't overlap at all (single queue anyway).
- Under the coarse mutex the scratch is never touched concurrently (mutex serializes), so per-context scratch
  only *matters* for a FINE-GRAINED design (lock just the submit ioctls, overlap the CPU/quant/softmax parts).
- The practically useful + safe overlap is unchanged: ONE ork stream (NPU verify) ‖ another stream's NON-ork
  CPU work (draft routing/sampling) — no concurrent ork graph_computes, so no queue collision, no scratch race.

Corrected concurrency test drives the two workers through the async+mutex path (submits serialize). BUT the
synthetic flash-attn harness (`tools/ork_xattn.cpp`) has a SETUP BUG: its CPU reference is itself garbage
(max|.|=2400, ork=inf; the attention output for this data should be ~0.15), so it does NOT yet give a clean
attention-correctness number on EITHER backend — the flash-attn graph/mask/layout construction is wrong, not
the backend. Single-stream run did NOT wedge (safe); the earlier wedge was purely the raw-concurrent-submit path.

**Validation status:** matmul path bit-correct post-refactor (`ork_xstream`, maxerr 0.000) => the refactor is
behavior-preserving for the core path, and the attention change is a mechanical per-context storage move
(references, identical warm-reuse). The attention path itself is best validated by a REAL MODEL run with
ORK_ATTN=1 (single-stream, safe) comparing perplexity vs ORK_ATTN=0 — NOT the synthetic harness (which needs
its flash-attn setup debugged first). Board was recovered via HA power-cycle after the concurrent-submit wedge.

## SLICE 2 VALIDATION (real model, 2026-07-15) — refactor CONFIRMED behavior-preserving
llama-perplexity, qwen3-1.7b-q8_0, wiki_tiny, c=512, 2 chunks:
- ORK_ATTN=0 (CPU attention, baseline): [1]222.3305 [2]58.3531
- ORK_ATTN=1 (NPU attention), Slice-2 (per-context): [1]69867.6979 [2]159722.0777
- ORK_ATTN=1 (NPU attention), pre-Slice-2 (old global g_attnp): [1]69867.6979 [2]159722.0777  <- BYTE-IDENTICAL
=> Slice-2 refactor is EXACTLY behavior-preserving (identical PPL pre/post; matmul path also bit-correct).
=> NPU attention (ORK_ATTN) is INCOHERENT (PPL ~70k vs ~222) but PRE-EXISTING — not the refactor. Matches
   attention-on-npu-2a (WIP, off by default). So a dflash consumer's TARGET VERIFY must keep attention on CPU
   (ORK_ATTN=0) and use the NPU only for the coherent MATMULS. Slice 3 does NOT depend on ORK_ATTN.

## SLICE 3 STRUCTURAL FINDING — dflash loop is STRICTLY SEQUENTIAL (no free overlap)
examples/dflash/dflash.cpp loop (per round): draft `llama_decode(ctx_dft)` -> verify `llama_decode(ctx_tgt)`
(NPU matmuls) -> accept-prefix -> commit `llama_decode(ctx_tgt)`. Every step consumes the previous step's
output, and draft(N+1) needs commit(N)'s accepted tokens. So the standard speculative loop has NO independent
work for the async cross-stream primitive to overlap. Realizing overlap needs ASYNC-SPECULATIVE decode: the
draft speculates block N+1 optimistically (assuming block N is accepted) DURING verify(N), rolling back the
speculated draft on rejection. That is a research-grade decode-loop redesign (new state management + acceptance
semantics), NOT a wiring change. => Slice 3 = that redesign, or a different consumer with genuine independent
NPU+CPU work (batch serving of independent sequences; overlap next-request prefill with current decode).

## SLICE 3 — DECISIVE OBSTACLE: speculative verify is BELOW the NPU threshold (runs on CPU)
Building the async-speculative engine surfaced a hardware-rule blocker. ork MUL_MAT offload gate
(ggml-ork.cpp:5510/5534/5552): `min_m=32`, `pass_m_threshold = M >= 32` for a big MULTI-DOMAIN int8 target
(single-domain small targets get threshold=1). Speculative/dflash VERIFY batches are M = block size (~5–16)
→ BELOW 32 → the target verify runs on CPU, not the NPU. So there is NO NPU work in the verify to overlap
with the CPU draft. This stacks on the earlier obstacles:
  1. dflash loop strictly sequential (draft→verify→commit chain).
  2. dflash drafter target-hidden-coupled (draft(N+1) needs verify(N)'s hidden) → can't ahead-speculate.
  3. NPU attention incoherent (verify attention must stay CPU).
  4. **verify M<32 for a big target → verify matmuls on CPU too** ← the decisive one.
=> Speculative/dflash decode is NOT a viable consumer for the async cross-stream primitive on RK3588: the
   target verify simply isn't NPU work at speculative block sizes.

**Natural consumer instead:** a workload with M>=32 NPU work ‖ independent CPU work —
  - BATCH SERVING: one request's PREFILL (M=prompt>=32 → NPU) ‖ another request's decode/CPU work.
  - PREFILL pipelines: overlap the next request's prefill (NPU) with current post-processing (CPU).
The primitive (Slices 1+2) is built, correct, and characterized; it's waiting on a consumer whose NPU half
is prefill-scale. Recommend pivoting the consumer, or parking until batch-serving is prioritized.

## TREE ATTENTION assessment (does it rescue Slice 3?) — clears the gate, but premise is shaky
Tree/multi-candidate speculation (verify a 32-64+ node tree in one forward w/ a tree attention mask) makes the
verify M>=32 -> clears the ork NPU gate. That's real. BUT the core payoff (NPU tree-verify > CPU verify) is
UNPROVEN and the evidence leans against it at practical model sizes:
- CPU prefill (qwen3-1.7b-q8, llama-bench, RELIABLE): ~84 t/s FLAT across pp32/64/128/256.
- NPU: llama-bench pp32=37.9 t/s but UNRELIABLE (short-prompt + lazy-pack false-low pitfall — the reason the
  project uses ork_bench). Clean ork_bench measurement BLOCKED: ork_bench DMA-faults on this model ("CREATE:
  Bad address" -> compute status -1, its GDN setup path). NPU itself HEALTHY (ork_xstream matmul bit-correct).
- Established truth (decode-is-cpu-path; NPU wins only LARGE-prefill/BIG-models) => for small-to-mid targets
  (where spec decode is practical on 32GB), CPU is competitive-or-better even at tree scale.
Plus tree spec is a LARGE research-grade build (tree construct + tree mask + tree KV + longest-path accept +
rollback; llama.cpp lacks it), and the async primitive is only a SECONDARY win needing an EAGLE decoupled draft.
=> NOT recommended speculatively. Gating fact to resolve FIRST: a clean NPU-vs-CPU verify-batch number (needs
   ork_bench fixed on the target, or another warmed harness). Without it, tree spec is a big bet on an unproven premise.

CONSISTENT THREAD (whole session): the NPU's strength is PREFILL of long prompts (big matmuls), NOT decode.
Every decode-acceleration path (speculative/dflash/tree) fights that decode-scale work is CPU-competitive. The
async cross-stream primitive is best where the NPU half is genuinely prefill-scale: BATCH SERVING (one request's
prefill on NPU ‖ another request's CPU work). That is the clean consumer.

## DECODE ASYNC-OVERLAP — the viable consumer (MEASURED 2026-07-15, tools/async_decode_probe.c)
The one decode-NPU idea with a measured positive. Async-overlap NPU decode matmuls (M=1, multi-core = now the
default, +40% single-domain) with between-matmul CPU-prep. Qwen2.5-7B projections, bit-exact (async C==sync C):
  op-point cpu/npu | stock per-call async | PERSISTENT big-core-pinned worker
  0.27             | 1.00x                | 1.25x (97% of ceiling)
  0.61             | 1.05x                | 1.36x (85%)
  1.36             | 1.12x                | 1.28x (73%)
  2.37             | 1.10x                | 1.22x (85%)
=> at decode's ~1.9 op-point: ~1.22-1.28x, bit-exact. Persistent PINNED worker is the key (stock per-call
   pthread_create spawn overhead capped it at 1.10x). Reaches only 73-85% of sum->max ceiling; residual =
   worker kernel SLEEP/WAKE (~50us/matmul). LITTLE-CORE DRAM-DOORBELL (spin not sleep; big cores keep prepping)
   targets exactly that -> ~1.35-1.4x.
BOUNDS: single-domain models only (7B multi-domain decode stays CPU — baseline 3.32 CPU vs 3.30 "MC", knob
gone/no-op). Matmul-level CEILING (real decode has dependency-chain edges that can't overlap; single-queue NPU
means the overlap is NPU-matmul ‖ CPU-prep, NOT NPU ‖ NPU). Realistic end-to-end ~1.1-1.25x.
INTEGRATION (next focused build): wire the persistent-pinned-worker async into the ggml-ork decode path as
dataflow-aware intra-graph matmul pipelining (launch NPU matmul async, overlap next INDEPENDENT op's CPU-prep on
big cores; PATH-b MoE std::thread is the in-backend precedent). Doorbell = follow-on (needs NONBLOCK+multi-submit
poll — a decode matmul is 3 core-submits). Intricate but the mechanism is proven.

## Design implication for the consumer (pipelined dflash/spec loop, Slice 3)
Pin the target-verify NPU worker to big cores, the draft/routing to little cores. Then target-verify(NPU) ‖
draft-generate(CPU-on-little) overlaps free. Serialize any NPU submits from BOTH streams on g_npu_queue_mu
(single hardware queue).

## Board/ops
- ork-driver primitives on `feat/npu-doorbell` (doorbell/NONBLOCK); NOT needed for Slice 1 (thread-based reuses
  existing blocking run funcs). Submodule stays at v0.7.0.
- Standard board safety: timeout every NPU cmd; SIGTERM not kill -9; wedge -> `ssh board 'sudo reboot'`.
