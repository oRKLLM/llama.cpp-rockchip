# Plan: SDK op shape-adapter ("slice-and-dice") — every op → NPU-valid, doorbell-eligible tiles (task #33)

## Goal / why (the doorbell-unified-submission rationale)

Today the backend has **two** NPU submit paths — blocking `ork_mm_run_i8` (kernel ioctl completion-wait) and the
**doorbell** `ork_dyn` (nonblock, userspace `civac` spin-poll). On the **single-stream** NPU, mixing them risks a
**submit race**: a blocking ioctl issued while a doorbell chain is in flight can wedge the IOMMU/NPU
(uninterruptibly — needs a reboot). The fix is **one controlled submission path: the doorbell owns every submit**
→ no race, no wedge, and interruptible (timeout/SIGTERM recoverable, unlike a wedged blocking ioctl). Prefill perf
is neutral either way (measured 213≈213); this is a **robustness/consistency** change, not a speed one.

The blocker to "everything on the doorbell": the doorbell only handles **verified** shapes — `c_base`
(Sn==1 & K≤4096 & Bf, any M). Wide-K (K>4096: `ffn_down` K=6144), wide-N (N>nmax=8192), unaligned (K%512, N%16),
and transposed ops are NOT doorbell-eligible. Today they fall to blocking (reintroducing the race) or hit the
**unverified M>1 wide-K colsplit → wedge** (observed: `ORK_PREFILL_DB` all-matmul hung on the down). A per-shape
`if` gate is a hack — it keeps the second submit path alive.

**The proper fix (this plan):** a compile-time (pack-time) **decomposition pass** that slices EVERY op into a
sequence of `c_base` doorbell-eligible tiles, so all submits ride the one doorbell path. No blocking fallback,
no unverified shape ever reaches the driver, no race.

## The decomposer (the "chef") — table-driven, data-not-branches

Input: op-kind + shape (K/N/M, dtype) + SoC caps (from `soc.c`: K≤4096, nmax=8192, `mg_max*64` M-cap, K%512,
N%16, core count, Bf availability). Output: an ordered **tile schedule** — a list of leaf sub-ops each a valid
`c_base` tile, plus the host glue (K-slice int32 accumulate, N-slice scatter, M-tile, transpose/pad).

A rule table maps each constraint violation to a decomposition primitive (recurse until every leaf is `c_base`):

| violation | primitive | note |
|---|---|---|
| K > 4096 | **K-slice** into ⌈K/4096⌉ pieces (each K%512==0), host-accumulate int32 partials | the exact path `run()` already uses for ffn_down K=18944 — proven bit-exact |
| N > nmax(8192) | **N-tile** into ⌈N/nmax⌉ slices, scatter into the output columns | matches `Sn>1` slicing |
| M > mg_max·64 | **M-tile** | the weight-DMA-amortization cap |
| K%512 / N%16 ≠ 0 | **pad** to alignment (or hard error w/ a clear message) | |
| needs transpose (Kᵀ, e.g. attn QKᵀ) | **reshape/transpose** op (host bridge, or on-device once RESHAPE decode lands) | see RESHAPE_WIP |

Everything reuses ork's **existing, proven** K-slice / N-tile / M-tile primitives — the chef just *composes* them
per shape instead of the run-path deciding ad-hoc.

## Where it runs (pack-time "compile", stored in the orkpack)

The SDK "compiles" a model into an orkpack; the slice plan per op-shape is computed **there** (SoC + shapes both
known) and stored in the orkpack (a new section, alongside `Bf` and the prescribed schedule from
`PRESCRIBED_TUNING_PLAN.md`). Runtime just **replays** the stored tile list through the doorbell — no runtime
decomposition, no branching, no blocking. Portability guard: store the pack-time SoC id; re-derive at load if the
detected SoC differs (the decomposition is SoC-cap-dependent).

## Composition with the rest of the architecture

- **Slice-and-dice (this, #33)** decides **WHAT** the doorbell-eligible tiles are (decomposition).
- **Prescribed planner (#35)** decides **HOW** to schedule/place those tiles (mechanism, core-map).
- **Doorbell** is the **single runtime substrate** that executes them — uniformly, no race.

Together: pack-time = decompose (chef) + schedule (planner) → bake in orkpack; runtime = replay on the doorbell.

## What it solves

- **The wide-K wedge, properly:** `ffn_down` K=6144 → K-slice into e.g. [4096, 2048] (or 2×3072, K%512) → each a
  `c_base` doorbell tile → the down runs on the doorbell (no blocking exception, no unverified colsplit, no wedge).
- **Unified submission:** every op → doorbell tiles → one submit path → the blocking/doorbell submit race is gone
  by construction → wedge-free + interruptible.

## Increments (each bit-exact-gated)

1. Constraint table + decomposer core (K-slice / N-tile / M-tile / pad) for int8 matmul. Unit-test each leaf
   decomposition **bit-exact** vs the un-sliced CPU reference (fixed-seed, like `test_matmul`'s golden).
2. Route the wide-K `ffn_down` through it → doorbell-eligible. A/B vs blocking: perf-neutral, **no wedge**, PPL
   identical. This retires the `ORK_PREFILL_DB` K-gate hack.
3. Extend to all op-kinds (attn QKV/O; QKᵀ/AV need the transpose primitive — host bridge first, RESHAPE later),
   norms, activations. Each behind a bit-exact gate.
4. Store the slice plan in the orkpack (format bump); runtime replays through the doorbell; drop all blocking
   submit sites → the doorbell owns every submit.
5. Compose with the prescribed planner (#35): one pack-time "compile" emits decomposition + schedule.

## Risks / notes

- **Bit-exactness is the gate** — K-slice int32 accumulate + N-slice scatter must match the un-sliced reference
  (the `run()` K-slice path is the proven template; reuse it, don't reinvent).
- **Transposes** (Kᵀ for attention) aren't a cheap on-device op yet (RESHAPE_WIP blocked) — those ops stay
  host-bridged until that lands; they're the last to move fully onto the doorbell.
- Pack-time cost is negligible (analytical decomposition; small stored plan).
- Board rules for the validation runs (single-stream, reboot-on-wedge, `make test` bit-exact + attest refresh).

## Files

- `ork-driver/src/` — the decomposer + constraint table (reuse `run()`'s K-slice/N-tile/M-tile), the orkpack
  slice-plan section in `npu.c` (format bump), `soc.c` caps as the input.
- `ggml/src/ggml-ork/ggml-ork.cpp` — route ops through the sliced doorbell plan; retire blocking submit sites.
- Ties into: the doorbell (`ork_dyn`), the prescribed planner (#35), RESHAPE (transposes), OPS_REGISTRY.
