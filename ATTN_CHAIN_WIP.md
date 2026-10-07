# Fused attention chain on NPU — WIP (feat/attn-npu)
Goal: intercept FLASH_ATTN_EXT -> QK^T(fp16 chain) -> on-NPU softmax -> A·V(fp16 chain), one/few
submits, keep-warm for matmul<->SDP swaps, persistent scratch pool (no per-op bcreate OOM).
Linchpin DONE: all-on-NPU softmax = 9-task chain, 485us, bit-exact (softmax_replay).
Phases: 2=general on-NPU softmax(reduce on NPU); 3=assemble attn chain; 4=wire FLASH_ATTN + A/B vs CPU.
Board 10.3.0.236: qwen3-1.7b-q8_0.gguf (dense attn), ork_bench (FA off by default -> need FA on).

## Phase 3 built (2026-07-13)
ggml_backend_ork_flash_attn_ext: batched QK^T + A·V via stream_f16_chain (1 submit each, all heads),
CPU softmax(scale+mask), persistent attn_pool (wqk/wav scratch reused), GQA (h/rk2), nkv->%32 pad.
Wired: graph_compute case + supports_op (ORK_ATTN, no alibi/softcap/sinks, N>=64) + weightless-offload += FA.
ork_bench needs FA enabled (env ORK_BENCH_FA). NEXT: board build, coherence vs CPU-FA, A/B @P512/2048.
Softmax still CPU (round-trip) -> if binds, move to on-NPU chain (linchpin proven).
