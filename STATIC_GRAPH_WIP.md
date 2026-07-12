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

## Rules
- Everything gated (ORK_STATIC_GRAPH etc.) — never break the default path.
- Coherence-gate every step (bit-identical output vs baseline, or ork_ppl). No coherence = back off + gate off.
- Commit each step with the result. errno=110 = real wedge (NOT dmesg soft-reset). Board self-heals.
- Hard stop -> web-search for ideas before giving up.

## Log
(append per step)
