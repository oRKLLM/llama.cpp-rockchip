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

## Goal

A **layer-windowed streaming** residence policy for the chain so the resident working set is bounded to the
**active layer** (≈400–800 MB, fits one domain), with each layer's weights (re)loaded **once per prefill
forward** instead of thrashed per-op. Turns the IOVA-budget wall into a bounded, amortizable per-forward DMA
cost. **Gated on beating the 44 t/s non-chain baseline** — a null/negative result is acceptable and closes
the thread.

## Approach

1. **Windowed wcache eviction policy (the core change).** Today `ork_wcache_evict` is demand-driven LRU
   (evicts the globally-least-recently-used when over budget). Add a **layer-window mode** (behind the chain):
   evict the *previous* layer's weights when the chain advances to a new layer, keeping only the active layer
   (and optionally layer+1 for prefetch/pipelining) resident. Deterministic eviction (last-layer-out,
   next-layer-in) means a weight is never evicted-then-immediately-re-needed — which is exactly the churn
   signature (`churn = 1960 ≫ 84 = 28 layers × 3 weights`).
   - Mechanism: a `layer` hint from `ggml_backend_ork_ffn_swiglu_chain` (it already knows the layer via the
     weight name / `ork_layer_of`) into the wcache; eviction prefers non-active-layer entries.
   - This composes with the existing **`wcache_pin`** (already landed): pin protects the *active op's* weights
     within a layer; the window bounds *across* layers.

2. **Prefill-only engagement (already true).** The chain has an `M < 32` skip, so decode (M=1) never uses it.
   Streaming is correct only for prefill (one forward over M rows amortizes the reload); decode stays CPU.
   So the redesign needs **no** decode path — the `NO_BF` "no verified doorbell path" failure disappears
   because the chain simply isn't on the decode path. (This also sidesteps task #19.)

3. **Bf under a window fits.** With only ~1 layer resident, `KEEP_BF` (Bf ≈ another tile per K≤4096 weight)
   costs ~1 extra layer's worth (~few hundred MB) — well inside one domain. So `KEEP_BF` can stay on (keeps
   the fused-SiLU envelope) without the 8-domain blowup, and the OOM disappears.

## Why it *might* win (and why it must be measured)

- Streaming reload = ~11 GiB @ ~11 GB/s ≈ 1–2 s per prefill forward, **amortized over the M-row batch**
  (M=128 → ~0.01 s/row) — cheap per token.
- The chain moves the *whole* FFN onto the NPU (int8 GEMM ~17× the CPU). The **baseline already declines the
  wide `ffn_up`/`ffn_down` (N/K = 18944) to CPU** — so the win is `CPU_FFN_time_saved − reload_DMA_time`.
  Plausibly positive at prefill M, but **not assumed** — the baseline's CPU FFN + NPU-for-fitting-weights is
  already 44 t/s. The redesign clears the bar only if the reload DMA is cheaper than the CPU FFN it removes.

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
