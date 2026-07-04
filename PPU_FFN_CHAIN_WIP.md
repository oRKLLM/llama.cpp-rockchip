# PPU FFN-Chain WIP (round-trip-free int8 SwiGLU FFN inner on NPU)

## Goal
Route the whole SwiGLU FFN inner (gate+SiLU → up → ewmul → down) as ONE on-NPU int8 chain, int8
intermediates never touching fp32. ORK_FFN_CHAIN=1 gated; default path untouched.

## State (2026-07-04)
- **Keystone SOLVED (ork-driver):** per-scale-correct fused SiLU exists + validated ~1 int8
  (`ork_mm_silu_build_lut` public API + tools/silu_native.c: mean|err| 0.8–1.06, max 3–4). The
  fixed-LUT register search (tools/silu_calibrate.c) tops out at mean 14.6 — DON'T use it; use LUT regen.
- **All ork-driver primitives exist + validated:** ork_mm_run_i8_silu, ork_mm_run_i8_out8,
  ork_npu_ewmul_i8, ork_mm_run_i8, ork_mm_silu_build_lut.
- **Detection DONE + validated on-board:** graph_compute matches [gate,up,GLU,down] as ONE 4-node ork
  subgraph at prefill **M≥32** (GLU supports_op needs M≥ork_ppu_minm()=32; at M<32 GLU→CPU and the FFN
  fragments into [gate,up]|[down] so NO match). Confirmed at M=128: 4/4 layers matched.
- Board build: `~/llama-ppu` (branch feature/ggml-ork-ppu-ops @ 59d224ec9, same as local fork).
  Sync: `rsync local ggml-ork.cpp -> board:llama-ppu/...`; build `cmake --build build --target llama-cli -j4`.
  Local fork: /Users/michael/Dev/llama.cpp feature/ggml-ork-ppu-ops (FFN-chain edits UNCOMMITTED).

## Structural constraint (the real one, deeper than the LUT)
Model quantizes activations PER-ROW (asr[m]) and weights PER-CHANNEL (bs[n]); fused SiLU output stage
applies a SINGLE scalar R → cannot express per-row/per-channel. So the chain REQUIRES PER-TENSOR scales
(scalar s_x, s_Wg/s_Wu/s_Wd) = accuracy concession (the GLU foothold degraded for this reason).

## Scale plan (per-tensor)
- x (ffn_norm out) → s_x = max|x|/127, x_i8.
- Wg/Wu/Wd requantized per-tensor (scalar) → new ptcache {ork_w*, scale}.
- gate: ork_mm_run_i8_silu, in_scale=s_x*s_Wg, out_scale=s_silu(chosen), LUT via ork_mm_silu_build_lut (cache/layer).
- up:   ork_mm_run_i8_out8, R_up = s_x*s_Wu/s_up.
- glu:  ork_npu_ewmul_i8, s_glu = s_silu*s_up*128 (gain 1/128).
- down: ork_mm_run_i8 (K=Nff=6144>4096 → high-level K-split path), dst = down_i32 * s_glu * s_Wd.

## RESULT (2026-07-04) — NET LOSS, not yet viable
Built end-to-end: detection (M≥32) + ggml_backend_ork_ffn_swiglu_chain (per-tensor ptcache + per-layer
lutcache + all 4 ops on NPU int8) + wired into graph_compute (ORK_FFN_CHAIN=1, off by default).
Measured in-model on Qwen3-1.7B-Q8_0 (board 10.3.0.236, `sudo timeout ... llama-bench -p 128 -r 1`):
- **CHAIN pp128 = 140.14 t/s  vs  BASELINE (per-node) = 147.43 t/s  → ~5% SLOWER.**
- Intermittent NPU **soft-reset** during the chain (7 resets in ~8s; kernel recovers, run completes) —
  one of the fused ops (silu/out8/ewmul) intermittently wedges. Stability defect.
- Per-tensor quant (lossy); coherence not separately confirmed — moot given the speed loss.

### Why it loses (structural, matches committed findings)
The fused chain needs MORE NPU submits than baseline: per FFN per M-tile(≤64) = LUT-load + silu-matmul +
up-out8 + ewmul + down(K-split) ≈ 5-6 submits, and the fused SiLU path is M-CAPPED at 64/submit. Baseline
= 3 larger matmuls (M-scheduler up to mg_max*64) + a cheap CPU GLU. The fp32 round-trip the chain eliminates
is NOT the prefill bottleneck (prefill is submit/compute-bound, not round-trip-bound). Also double IOVA
(per-tensor weights packed SEPARATELY from the per-channel wcache).

### To even reach parity would need a breakthrough
1. Fused-path large-M-tiles (mg_max*64, like the int8 matmul fix) so silu/out8 aren't capped at 64/submit.
2. Load the silu LUT ONCE (not per M-tile) — currently re-streamed every tile.
3. Fix the intermittent soft-reset (fused-op submit robustness at Nff=6144, K=6144 down).
Even then the ceiling is bounded (prefill isn't round-trip-bound). Decode can't use it (M=1 → GLU on CPU →
no match, and the serialization wall makes M=1 lose anyway). Per-channel-R RE (SDP CVT) would fix accuracy
but not the submit-count loss. **Verdict: not yet viable; needs the large-M fused-path + single-LUT-load first.**

Code kept, gated OFF (ORK_FFN_CHAIN unset = default per-node path, untouched, baseline 147). Committed WIP.

## Board ops
- ALWAYS `sudo timeout N env ... llama-bench` (timeout as root kills the root child; plain `timeout sudo` can't).
- Kill stragglers with SIGTERM (never -9 in-flight submit). Watch for stale prior-session llama procs
  contending the single-stream NPU (found a 46-min zombie this session).
- llama-bench (not llama-cli) for clean prefill + no spinner spam.
