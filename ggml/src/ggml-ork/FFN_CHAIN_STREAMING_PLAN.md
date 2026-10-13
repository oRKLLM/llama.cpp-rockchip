# Plan: layer-windowed streaming FFN chain (make `ORK_FFN_CHAIN` viable on >4 GiB multi-domain models)

## Context — the measured problem

`ORK_FFN_CHAIN` fuses the SwiGLU inner (gate·x → SiLU → up·x → GLU → down·glu) to keep int8 intermediates
on-NPU. It is off-by-default and, per `OPS_REGISTRY.md`, **PARTIAL/WIP**. Getting it to run on the dense
multi-domain **qwen2.5-7b** (2026-07-25) required fixing two real bugs (both landed, see below), after which
the chain runs end-to-end — but the measurement shows it is **not viable** on a >4 GiB model, for an
*architectural* reason, not a bug:

| Config | Result on 7B (5–8 domains) |
|---|---|
| `NO_BF` (auto-set by the chain), 3 domains | warmup decode fails — "no verified doorbell path, Bf=0" |
| `KEEP_BF`, 8 domains | warmup decode fails — IOVA OOM (`bcreate Bb[0] 8 MB`: resident set fills all domains) |
| `ALLOW_JIT` (either) | runs, but **1.02 t/s prefill, churn = 1960** — a **~44× regression** vs the 44 t/s non-chain baseline |

**Root cause:** the chain (via the wcache's keep-everything-resident LRU policy) tries to hold *all* FFN
weights NPU-resident simultaneously. On the 7B that is ~11 GiB (gate/up/down × 28 layers, + a full-K `Bf`
rebuild per K≤4096 weight under `KEEP_BF`). That exceeds the per-domain ~4 GiB IOVA budget, so the runtime
either can't allocate a runtime scratch pack (OOM) or evicts-then-re-packs weights every op (churn → 1 t/s).
The chain's *math* does not require co-residence — ggml-ork.cpp already notes "gate/up/down do NOT need
co-residence; each is a SEPARATE `ork_mm_run_*` call, `dom_activate` switching between." Only **one weight is
needed resident at a time.** The co-residence is an artifact of the LRU cache policy, not the algorithm.

## Goal (REVISED — domain-ordered full residence, ZERO reload; restores prior behavior)

The correct model is **not** per-layer reload/streaming — it is **domain-ordered full residence**, which the
non-chain multi-domain path ALREADY achieves (11.45 GiB across 5 domains, **churn=0**, 44 t/s): every weight
stays NPU-resident across ALL domains the whole time (the domains' combined IOVA, N × ~3.9 GiB, holds the
full set), and because domains are **layer-aligned** (`ork_weight_domain` advances only at layer boundaries →
contiguous layer ranges), a forward naturally processes domain 0's layers, then domain 1's, … — `dom_activate`
just switches the active domain at each domain boundary (~4 swaps/forward × 0.78 µs). **No reload, no
eviction, churn=0.** The chain REGRESSED this (churn=1960 / OOM); the goal is to restore it for the chain path.

## Approach (REVISED)

1. **Hold the chain's full resident set, process domain-ordered — don't evict mid-forward.** The chain must
   NOT re-pack at runtime. Concretely:
   - **Size for the chain's larger set.** NOTE (measured, corrected): the baseline ALSO runs the wide
     `ffn_up`/`ffn_down` on the NPU at prefill (M≥32; `run_multicore` per-layer down-proj) — an earlier claim
     that it CPUs them was wrong. The chain's footprint is bigger only because of the extra per-tensor `fc.wg`
     gate (a 3rd gate copy). The auto-sizer must size `n_domains`/budget for that + `fc.wg` + **per-domain
     headroom for runtime scratch** (the KEEP_BF OOM was an 8 MB `Bb[0]` with all domains packed to the brim).
     **DONE** — commit `9bd0e5962` added the `fc.wg` inflation term → churn 1960→0, no OOM.
   - **Load everything in the LOAD phase, never JIT at runtime.** churn (`mem_create_runtime`) counts
     runtime packs; ORK_ALLOW_JIT builds weights during the first forward → those count as churn (the 1960 was
     largely JIT build packs). A pre-built complete pack (all chain weights) + full residence = churn 0.
   - With the set fully resident and budget ≥ footprint, `ork_wcache_evict` never fires within a forward →
     domain-ordered processing falls out naturally (dom_activate at boundaries, no eviction). The landed
     `wcache_pin` covers the within-op case; this covers the across-domain case by simply not over-committing.

2. **Prefill-only (already true).** The chain's `M<32` skip keeps decode (M=1) off the chain → decode stays
   CPU. So there is NO decode NPU path to satisfy — the `NO_BF` "no verified doorbell path" failure and the
   warmup-decode OOM both come from decode wrongly trying the NPU; ensure M=1 `ffn_up`/`down` **decline to CPU**
   (don't pack at runtime). That removes the last two failure modes without touching the driver decode path.

## Why it should win (residence, not a reload tradeoff)

Because it's zero-reload, the chain runs at the **same residence efficiency as the non-chain 44 t/s path**,
PLUS the fusion benefit (gate/SiLU/up/GLU collapsed, int8 intermediates never round-trip to fp32, fewer
submits). So the expected win is the **fusion delta over 44 t/s**, not a reload gamble. The only real work is
restoring full residence (sizing + no runtime JIT + decode-declines-to-CPU) so churn returns to 0. Still
**gated on a board A/B beating 44 t/s** (+ PPL for the SmoothQuant static scales), but the framing is now
"restore the residence the chain regressed," not "pay a per-forward reload."

## Measurement gate

- A/B on the board (governors=performance, single-stream, warm, ≥2 reps): non-chain baseline (44 t/s) vs the
  windowed chain, on qwen2.5-7b-q8_0, **prefill P≥128**, `churn` must be ≈ `28×3` (one pass, no thrash),
  byte-plausible output (SmoothQuant accuracy caveat still applies — validate PPL, not just t/s).
- Ship the windowed policy **only on a clear prefill win (≳10%)** over 44 t/s. Otherwise record it in
  `OPS_REGISTRY.md` + memory as measured-not-worth-it and leave the chain single-domain-only.

## Files

- `ggml/src/ggml-ork/ggml-ork.cpp` — `ork_wcache_evict` windowed-eviction mode; a `wcache_active_layer` hint on
  the context; `ggml_backend_ork_ffn_swiglu_chain` sets the layer + drives the window. Reuses the landed
  `wcache_pin`.
- `OPS_REGISTRY.md` — update the `ORK_FFN_CHAIN` row with the multi-domain status + the windowed-streaming
  result once measured.
- `docs`/memory — record the A/B outcome.

## Risks / notes

- **Uncertain payoff** — the honest expectation is "maybe," gated on the A/B. The whole point is to *decide*,
  not to assume.
- **Prefill of a dense >4 GiB model is a narrow target** — MoE models already fit one domain (small active set);
  ≤4 GiB dense already fits one domain with the chain. So this helps essentially only dense 7B-class models.
- **SmoothQuant accuracy** — `ORK_FFN_CHAIN` needs per-layer static-scale calibration on a representative
  batch (the `M<32` skip guards this); validate perplexity, not just throughput.
- Board rules (single-stream, `sudo reboot` on wedge, reboot between heavy runs — cross-run IOVA is not freed).

## Landed fixes this builds on (2026-07-25)

1. **`Bb[k]` group-pack domain place+spill** (committed `66fa997cb`) — multi-domain loads work; non-chain 7B
   loads at 44 t/s.
2. **SiLU-probe domain fix** (`ork_npu_probe_i8_silu_cfg` used `c->dom_active`, not hardcoded dom 0) — killed a
   per-layer multi-domain wedge (0 self-heal resets after).
3. **wcache-pin use-after-free fix** — a multi-weight op held `ork_weight&` refs across an evicting resolve;
   ASAN-confirmed fixed. The window policy in this plan is the *cross-layer* generalization of that pin.
