# Plan: prescribed model-based tuning — pack-time calibration + orkpack-baked schedules

**Goal.** Make chain/op scheduling *adaptive across model + chipset* with *zero per-op runtime cost*, by moving
the adaptivity to a one-time PLANNER at **orkpack creation** (on-target), baking the winning **prescribed
schedule** into the orkpack, and executing it statically on `ork_spine_unit` primitives at run time.

**Why this shape (measured, 2026-07-26, qwen3-1.7b, RK3588).** The dynamic `ork_spine_run` DAG scheduler is
correct (PPL-identical) but **55–90 t/s vs 141 naive** — a 20µs poll-tick + cross-thread per-op dispatch tax.
Hand-prescribed schedules hit **~151 (+6.5%)** but are hardcoded to this model+chip. The core-map alone swung
**112 ↔ 151** (silu-on-littles-vs-big, driver core, `nc`, sleep-vs-spin). So: adaptivity belongs in a one-time
plan, not a per-op loop; and the plan MUST be chosen per model+chip because the optimum moves.

Start with the FFN SwiGLU chain; the framework generalizes to every chain/op (attention FRONT/CORE, RMSNorm,
standalone activations).

---

## 1. The schedule descriptor (the baked artifact)

A compact, serializable, versioned struct — the "prescribed plan" for one op-shape. "Data, not branches" (cf.
`XSPEC`, the domain auto-sizer, `ork_op_cap`).

```c
// one entry per UNIQUE (chain-type, shape) in the model. All FFN layers share a shape => 1 entry.
typedef struct {
    uint16_t chain_type;     // ORK_CHAIN_FFN_SWIGLU / ORK_CHAIN_ATTN_FRONT / ATTN_CORE / RMSNORM / ...
    uint16_t version;        // descriptor schema version
    int32_t  K, N, Kd;       // shape signature (the key). M is runtime -> not keyed.
    uint8_t  mechanism;      // NPU submit: 0=blocking/kernel-IRQ(sleep), 1=doorbell(spin)
    uint8_t  npu_nc;         // doorbell colsplit core count (0=all); N/A for blocking
    uint8_t  driver_core;    // NPU spinner core (doorbell only), 0xff=none/blocking
    uint8_t  ncpu;           // # CPU units in the plan
    uint8_t  cpu_cores[8];   // CPU-unit core ids (silu/glu), ncpu valid
    uint16_t tile_rows;      // M-tile size formula param (0 = full-M / nt=1); executor derives nt=ceil(M/tile)
    uint8_t  op_place[8];    // per-sub-op placement: ORK_PL_NPU/CPU/EITHER (indexed by the chain's op template)
    uint8_t  reserved[6];
    float    calib_tok_s;    // pack-time measured throughput of this plan (provenance/telemetry)
} ork_sched_plan;            // 64 bytes, fixed
```

The **op template** (which sub-ops exist + their deps) is chain_type-intrinsic, defined in code (not stored) —
e.g. FFN = {gate,up,silu,glu,down} with the fixed dep DAG. The plan only stores the *tunable* choices
(placement, core-map, mechanism, tile). Small + forward-compatible (version + reserved).

---

## 2. orkpack format extension

The orkpack already bakes NPU-tiled weights + `Bf` + IOVA domain layout and is created on-target (`ORK_PERSIST`
during a board run). Add a **`TUNE` section** the same way `Bf` was added (format bump + regen):

```
[existing orkpack sections ...]
TUNE section:
  magic "ORKTUNE", section_version u16
  soc_id u32                 // pack-time detected SoC (soc.c) — portability guard
  calib_meta { corpus_hash, timestamp(passed-in), calibrator_version, n_reps, ORK_GIT_HASH }
  n_plans u16
  ork_sched_plan[ n_plans ]  // one per unique (chain_type, shape) tuned
```

- **Lookup at load:** key = (chain_type, K, N, Kd); miss => analytical fallback (§5) or naive path.
- **Portability guard:** if detected SoC != `soc_id`, ignore the baked core-map and re-plan analytically at load
  (the baked map is silicon-specific; weights/domains already made the orkpack board-specific, so this is a
  safety net, not the common path).
- **Back-compat:** absent TUNE section => today's behavior (naive/hand-gated paths). Never a hard dependency.

---

## 3. The calibration microbench (the planner)

Runs ONCE at orkpack creation, on-target, after the weights are packed (shapes + real silicon both known).

**Driver:** for each unique (chain_type, shape) in the model, enumerate a *bounded* candidate set, run each
through the **prescribed executor** (§4) on real packed weights + a representative activation, time ≥`n_reps`
warm reps, verify correctness, pick the fastest correct plan, store it.

```
for each unique (chain_type, shape):
    ref = run naive/serial path once  -> golden output (for correctness)
    cands = candidate_plans(chain_type, shape, soc_caps)      // §3.1, bounded
    best = null
    for c in cands:
        out, t = timed_run(prescribed_exec(chain_type, shape, c, weights, act), reps=n_reps)
        if !bit_equal(out, ref) within tol: continue          // correctness gate (§3.2)
        if best==null || t < best.t: best = {c, t}
    plans.push(best.c with calib_tok_s)
```

### 3.1 Candidate generation (bounded search, measurement-informed)

The search space is small and pruned by SoC caps + the measured op-cost table (`ork_op_cap_for`):
- **mechanism** ∈ {blocking, doorbell} (2)
- **core-map** — from the SoC's big/little layout. Measured winners: for *blocking*, silu on all cores incl.
  big (apipe: 0–6); for *doorbell*, silu on big + a little-core spinner + reserved cores (nc=2). So the
  candidate maps are a curated handful per mechanism, not a full permutation:
  - blocking: {silu=all-but-1-big, silu=all}, driver=none
  - doorbell: {silu=big-only + spinner=little, nc∈{2,3}}, {silu=littles-only, nc=all}
- **tile_rows** ∈ {full-M, M/2} — full-M usually wins (weight-DMA-bound; per-tile re-stream tax). Include M/2
  only if the chain has a real cross-tile pipeline opportunity.
- **Prune with the analytical model first:** rank candidates by the cost-model (§5), only *measure* the top-K
  (e.g. K=6). Keeps pack-time cost bounded (seconds per shape × few unique shapes).

Total: ~6–12 measured candidates × ~3 reps × few unique shapes ⇒ pack-time overhead on the order of seconds,
paid once. Log what was tried + the winner (never silently cap the search).

### 3.2 Correctness gate

Every candidate's output is compared to the naive/serial reference for that shape (bit-identical for
integer-deterministic paths; fp tolerance where applicable). An incorrect candidate is **rejected**, never
timed-into-the-winner. This is the calibration-time analog of the PPL gate (which already confirmed the
overlap paths are numerically identical to baseline).

---

## 4. The prescribed executor

One executor, parameterized by an `ork_sched_plan`, drives the persistent `ork_spine_unit`s via
`ork_spine_unit_dispatch`/`_wait` — **not** `ork_spine_run` (whose dynamic poll-loop is the measured tax). Per
chain_type, a small hand-written dispatch routine walks the fixed op template with the plan's placement/core-map:

- FFN: `gate(NPU)` → (`up(NPU)` dispatched ‖ `silu(CPU units)` dispatched) → join → `glu(CPU)` → `down(NPU)`.
  Coarse (≈ apipe's shape) so the 20µs `_wait` is paid a few times per layer, not per-op.
- Units are persistent (started at backend init from the plan's core-map, stopped at free), reused across calls.
- Mechanism switch: blocking `ork_mm_run_i8` on the NPU unit (sleeps) vs doorbell chain (spins) per the plan.

This is a generalization of the validated bespoke paths (apipe+pool / sched+driver, both ~151) onto the shared
`ork_spine_unit` type, with the choices read from the plan instead of env vars.

---

## 5. Analytical cost-model (the fallback + the pruner)

A closed-form estimator used to (a) prune the candidate set before measuring, and (b) plan at load when the
baked SoC != detected SoC. Inputs: SoC caps (NPU GMAC/s, NPU core count, per-core-class CPU rates, DRAM BW) +
op costs (`ork_op_cap_for`) + shape. Estimates each candidate's critical-path from lane models
(NPU matmul time, CPU silu time per core-set, overlap). Not as accurate as measurement (hence pruning, not
deciding), but portable-by-construction and needs no board run.

---

## 6. Generalization: the chain/op registry

FFN is the first entry; the framework is a registry so every chain/op plugs in:

```c
typedef struct {
    uint16_t chain_type;
    void  (*op_template)(...);         // sub-ops + dep DAG (intrinsic, not stored)
    int   (*candidates)(shape, soc, ork_sched_plan* out, int max);   // §3.1 bounded set
    long  (*prescribed_exec)(plan, shape, weights, act, out);        // §4 executor
    int   (*reference)(shape, weights, act, ref_out);                // §3.2 golden
    double(*cost_model)(plan, shape, soc);                           // §5 estimate
} ork_chain_tuner;
```

The calibrator iterates registered tuners over the model's unique shapes. New chains (attention FRONT `[RMSNorm→
QKV]`, CORE `[QK^T→exp→reduce, e·V]`, standalone activations) each register these five hooks; the calibrator,
orkpack section, load path, and executor are all chain-agnostic. This is the op-level sibling of task #33
(the shape-adapter) and consumes the `ork_op_cap` placement table.

---

## 7. Build increments (each measurable, gated)

1. **Prescribed executor for FFN on `ork_spine_unit`** (env-param'd first, no orkpack yet). A/B vs apipe+pool
   (151) — must match. This ports the validated schedule onto the shared unit type.
2. **Calibrator for FFN** (candidate gen + timed_run + correctness gate), emits an `ork_sched_plan` to stdout;
   validate the picked plan == the hand-tuned winner (~151).
3. **orkpack TUNE section** (format bump + regen): write the plan at pack creation, read at load, feed the
   executor. Portability guard + analytical fallback.
4. **Analytical cost-model** (pruner + SoC-mismatch fallback).
5. **Registry + 2nd chain** (attention FRONT or a standalone activation) to prove chain-agnosticism.

## 8. Gates / risks
- Correctness gate in the calibrator is mandatory (reject wrong candidates); ship a plan only if it beats the
  naive path for that shape (else store naive / leave the entry out).
- Pack-time cost must stay bounded (prune to top-K; log drops).
- `ork_spine_unit_wait` is a 20µs sleep-poll — keep schedules coarse (few joins/layer).
- Portability: baked core-map is silicon-specific; the SoC-id guard + analytical fallback prevent a moved
  orkpack from running a wrong-chip plan.
- Board rules (single-stream, reboot between heavy runs, one clean run per boot) apply to the calibrator's
  timed runs; the calibrator must self-serialize (it's the pack-time owner of the NPU).

## Files
- `ork-driver/tools/` — the calibrator harness (C; per §3) + candidate/cost-model helpers.
- `ork-driver/src/npu.c` + `include/ork_npu.h` — orkpack TUNE section read/write (format bump), `ork_sched_plan`.
- `ggml/src/ggml-ork/ggml-ork.cpp` — the prescribed executor + chain registry (FFN first), load-path plan lookup.
- `OPS_REGISTRY.md` / memory — record the framework + per-chain tuned status.
