# Static regcmd graph — autonomous build WIP

Goal: a precompiled/chained static layer graph that amortizes per-op round-trips (the rkllm prefill
lever). Established direction: on-NPU layer, HW-chain independent matmul groups, precompile regcmd,
escape ggml per-node dispatch. Baseline int16 prefill ~97 tok/s (Qwen3-1.7B-Q8_0, 256, orkpack) vs
rkllm ~184. Board 10.3.0.236 (non-prod, wedge-OK). ork_bench requires ORK_PERSIST=.orkpack.

Foundation already landed (fork feat/ork-static-graph):
- ORK_CLAIM_OPS: whole layer = one 24-node ork subgraph (01ef7db2b).
- ORK_ATTN + SOFT_MAX handler: attention QKᵀ/AV/softmax on NPU (coherent, per-node).
- run_chain_i8 keepwarm + bsync-dedup + DMA-scratch (ork-driver feat/static-graph).
- ORK_STATIC_GRAPH planner (7a726dcc4): emits the layer's static plan.

## Steps (execute in order)
1. [ ] Scan-ahead grouping: reorder independent same-input matmuls (q/k/v) so they HW-chain (they're
       split in graph order by RoPE/norm). Extend the group-fuse with a done-set + lookahead-past-movable.
2. [ ] Executor: run the plan directly (run_chain_i8 groups / run_i8 singles / CPU-delegate between).
3. [ ] Regcmd precompile + replay: extend the pcrc cache to fixed-M prefill (build regcmd once, replay).
4. [ ] Measure: coherence (bit-identical / PPL) + prefill vs 97 baseline.

## Claimed-ops -> NPU primitives (2026-07-12, all gated: ORK_OPS_NPU)
- **RoPE**: NEW primitive `ork_npu_rope_neox_f16` (ork-driver 7579461) — NEOX as x⊙COS+rot(x)⊙SIN via 2
  ewmul+1 add; validated vs CPU (err 0.0025); wired (1a191581), coherent in-model ('Toby Steele').
- **RMSNorm** (ork_npu_rmsnorm_f16 norm-only) + **MUL** (ork_npu_ewmul_f16, broadcast norm-weight): wired
  on NPU (4ee02e98), coherent.
- **residual ADD**: fp16 BREAKS coherence (accumulates over 28 layers -> garbage). Kept fp32 (CPU); opt-in
  ORK_OPS_NPU_ADD. It's a boundary op (no chaining benefit) so fp32 is fine.
- RESULT: RoPE+RMSNorm+MUL on NPU = coherent; the attention/FFN data path is now largely on-NPU (residual
  the one fp32/CPU boundary). Per-node still pays round-trips; the heterogeneous-chain assembler amortizes.
- NEXT: the chain assembler (chain the now-on-NPU segments into single submits) + the mul_mat_group_i8
  variable-N fix (q/k/v scan-ahead) + the 3-graph/gmax selection design.

## Rules
- Everything gated (ORK_STATIC_GRAPH etc.) — never break the default path.
- Coherence-gate every step (bit-identical output vs baseline, or ork_ppl). No coherence = back off + gate off.
- Commit each step with the result. errno=110 = real wedge (NOT dmesg soft-reset). Board self-heals.
- Hard stop -> web-search for ideas before giving up.

## Log — autonomous run (all 4 steps executed; everything gated OFF, default path unchanged @ ~97 tok/s)

- **Step 1 (scan-ahead grouping)** — DONE, gated off (ORK_SCAN_AHEAD, fc303a80c). Groups independent
  same-input matmuls across movable ops (q/k/v). Works for gate+up (consecutive, coherent) but grouping
  q/k/v exposes a LATENT bug in mul_mat_group_i8 for the variable-N/GQA case (q N=2048, k/v N=1024 — never
  grouped before). Output garbage; dst-stride fix (nb[1], kept) + pack/bscale inspection didn't resolve.
  FOLLOW-UP: mul_mat_group_i8 variable-N correctness.
- **Step 2 (executor)** — SATISFIED by ORK_CLAIM_OPS (01ef7db2b): the whole layer runs as one 24-node
  ork subgraph (planner 7a726dcc4 emits the plan). Coherent. No separate rewrite needed.
- **Step 3 (precompile prefill regcmd)** — ASSESSED, deferred. Lands in the intricate per-N-slice-chaining
  M>1 mcworker path (high-risk) AND the profile shows regcmd-synth is a small slice (prefill is CPU-quant +
  submit-round-trip bound), so precompiling regcmd is low-impact. Not worth the surgery vs the real lever.
- **Step 4 (MEASURE)** — DONE, the verdict. Full static stack (CLAIM_OPS + ATTN + SOFTMAX_NPU, whole layer
  ork-owned): **76.3 tok/s vs 97.4 baseline = -22%, coherent.** Profile: submits went UP (9807 vs ~6460) —
  per-node on-NPU attention/softmax + delegated ops each ADD a round-trip. **Per-node on-NPU REGRESSES.**

## VERDICT + confirmed path forward
The static-graph FOUNDATION is built + coherent (whole-layer contiguity, attention/softmax/silu/norm all
runnable on NPU, all primitives validated), but running it PER-NODE regresses because each op pays a
round-trip. The win requires REDUCING submits, not moving ops on-NPU — i.e. the **heterogeneous PC-chain
assembler**: chain a whole NPU-only run into ONE submit (task_number=N), like the vendor's decoded 27-task
softmax chain. Web-confirmed (jas-hacks RKNN RE + NVDLA double-buffering + CUDA-Graph): vendors run whole
models as ONE precompiled command buffer. Chainable units (CPU ops interrupt the data path, so not one
all-layer chain): the ATTENTION BLOCK (QKᵀ+softmax+A·V — all on NPU now) as one chain; the FFN inner as one
chain. This is a fresh large build (heterogeneous multi-op-type PC-chain across QKᵀ/SDP/AV + per-head batch),
beyond the 4-step plan. Phase-0 (matmul→silu 2-task chain) validated the mechanism; extending to the
attention block is the next milestone. Everything from this run is GATED OFF — main/default path is the
unchanged 97 tok/s baseline.
