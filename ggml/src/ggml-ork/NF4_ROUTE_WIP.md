# NF4/int4 CPU-prefill route — WIP recovery

## Goal (task #54, this session)
Run the 35B MoE prefill using the **int4/NF4 experts from the orkpack on the CPU** (batched NEON
GEMM), as the "resident int4 for both NPU and CPU" A/B. Then benchmark prefill t/s + accuracy vs the
NPU path and vs ggml Q4_K.

## What's built (all gated, uncommitted)
- `ork-driver/include/ork_native_cpu.h`: `ork_gemm_i4_4x4`+`ork_cpu_gemm_i4` and
  `ork_gemm_nf4_4x4`+`ork_cpu_gemm_nf4` — 4x4 register-blocked batched GEMM (weight/unpack amortized
  across M rows). NF4 unpacks via `vqtbl1q_s8(lut,...)`; I4 via sign-extend.
- `ork-driver/examples/test_i4_gemm.c` (+Makefile EXAMPLES): validates batched vs M=1 gemv,
  bit-exact + speedup sweep. **Must run on the board (NEON dotprod).**
- `ggml-ork.cpp` run_cold batched cold path (~4234): `if (ork_native && batch_fmt && cold_batch)`
  gathers each cold expert's M_e rows → one `ork_cpu_gemm_i4/nf4`. `[ork COLD-GATE]` diag under
  ORK_VERBOSE. ofmt from orkpack O4N1 header qk (0=I4, 1=NF4).
- `ggml-ork.cpp` NEW: **`ORK_MOE_CPU`** flag (~4174 decl, ~4203 gate) — forces ALL experts to the CPU
  cold path (skips the NPU resolve). This is the switch that makes the experts reach run_cold. Without
  it, orkpack-resident experts route to hot_e (NPU) and run_cold/COLD-GATE never fires.

## Root cause found (why COLD-GATE never printed before)
`run_cold()` only runs for experts NOT admitted to the NPU. orkpack experts are int4-resident on the
NPU IOVA → `ork_resolve_weight_i8` succeeds → they route to `hot_e` (NPU), so S≠0 and the batched CPU
path is never reached. `ORK_MOE_HOT=0` does NOT force cold (that caps the old hot-slot cache, not the
footprint-resident resolve). Fix = the new `ORK_MOE_CPU` gate.

## Board facts
- board reachable (aarch64), governors=performance, DMC=performance.
- orkpack: `~/qwen3.6-35b-a3b-w4a4.orkpack` (16.9 GB, has all 30720 routed experts — verified).
- Single-stream: ONE run at a time, SIGTERM not SIGKILL, `sudo reboot` on wedge.
- Board runs the llama.cpp fork; ork sources live under `~/llama.cpp/ggml/src/ggml-ork/` (submodule
  path). Must sync BOTH ggml-ork.cpp AND ork-driver/include/ork_native_cpu.h before building llama-bench.

## Next steps
1. Sync sources to board (ggml-ork.cpp + ork_native_cpu.h + test_i4_gemm.c + Makefile).
2. Fast proxy: build+run `test_i4_gemm` on the board → confirm batched I4+NF4 bit-exact on real NEON.
3. Build llama-bench with ork backend.
4. A/B prefill: `ORK_PERSIST=~/qwen3.6-35b-a3b-w4a4.orkpack ORK_MOE_CPU=1 ORK_VERBOSE=1 -t4 -p512`
   - confirm `[ork COLD-GATE] ...batched=1` prints (path executes) BEFORE trusting numbers.
   - batched vs `ORK_MOE_COLD_BATCH=0` (per-row) A/B.
   - vs NPU path (no ORK_MOE_CPU) and vs ggml Q4_K mixed.gguf.
5. Accuracy: perplexity NF4-CPU vs Q4_K.
6. Consolidate/commit (gated), update memory + OPS_REGISTRY, board cleanup.

## Gotchas
- fast-validation: confirm path EXECUTES (COLD-GATE) before measuring; don't trust t/s that's ≈native.
- test_i4_gemm earlier "bit-exact" may have been pre-power-cycle; re-run on board to be sure. [DONE: bit-exact on board]

## UPDATE 2026-08-17 — full root-cause chain + DATA LOSS incident

### DATA LOSS (surface to user)
The first probe run (ORK_PERSIST=<w4a4 file> ORK_QUANT=4 ORK_MOE_CPU=1) had `quant_sig 52` which
mismatched the stored `quant_sig 564` on the existing 16.9 GB `qwen3.6-35b-a3b-w4a4.orkpack`, so the
loader auto-**regenerated** it — and clobbered it down to a **673 MB dense-only** file. The original
16.9 GB expert-bearing orkpack is GONE (no backup). It is a *derived* artifact (rebuildable from the
GGUF), not source — recoverable, but the user should know the file was overwritten.
- To rebuild the ORIGINAL (NPU path, task #54): `ORK_QUANT=4 ORK_HADAMARD=1 ORK_MOE_NPU=1` write mode
  (sig 564 = 52 + hd<<9). Its experts are `ORKPACK_DT_I4_NATIVE` (dtype 5, Hadamard-rotated).

### ork_build_sig() decode (line 194)
`sig = qb | (hy<<8) | (hd<<9) | (gq<<10)`, qb = first char of $ORK_QUANT ('4'=52). So:
- 52  = ORK_QUANT=4 (plain)        → O4N1 experts (dtype 4, CPU-readable)  ← the CPU route wants THIS
- 564 = ORK_QUANT=4 + ORK_HADAMARD → I4_NATIVE experts (dtype 5, NPU only) ← what the old file was

### The format gap (why the CPU route never had a compatible pack)
- CPU batched cold path detects `ork_native` only when expert `dtype == ORKPACK_DT_I4` (O4N1, line 3756).
- The old orkpack's experts were `I4_NATIVE` (dtype 5) → CPU path would NOT have engaged even before
  the overwrite. The "30720 experts present" was true but in the wrong format for CPU.

### Two code fixes applied (gated, uncommitted)
1. `ORK_MOE_CPU` flag (routing ~4174/4203): forces all experts to the CPU cold path.
2. `ORK_MOE_CPU` in supports_op (~7510): CLAIM MUL_MAT_ID (else the handler never runs → the routing
   flag was dead code AND write-mode never packs experts). Guarded the ORK_MOE_NPU warning too.

### persist expert-packing (ork_persist_write_experts, line 1233)
- Only runs in `persist_mode==2` (WRITE) AND only if supports_op claims MUL_MAT_ID (needs
  ORK_MOE_NPU / ORK_NO_EXPERT_REPACK / ORK_MOE_BIGLITTLE / now ORK_MOE_CPU).
- tier==4 (int4) O4N1 path (line 1319) is PURE CPU (ork_pack_i4a8_cpu_blob, no bcreate), parallel.
- NF4 codebook is auto: `nf4 = ork_src_type_bits(type) >= 16` (line 1321). So **NF4 requires an
  f16/f32 source** (qwen3.6-35b-a3b-f16.gguf, 69 GB). Q4_K source (mixed.gguf) → uniform int4 only.

### Current build in flight
Building `qwen3.6-35b-a3b-o4n1.orkpack` (NEW file) from mixed.gguf: `ORK_MOE_CPU=1 ORK_QUANT=4` write
mode → O4N1 UNIFORM int4 experts (dtype 4). This validates the CPU batched route end-to-end + gives a
perf number. NF4 accuracy path = same build from the f16 gguf (documented next step).

### Benchmark plan (once o4n1 pack exists)
- confirm: `ORK_MOE_CPU=1 ORK_VERBOSE=1` load → `[ork COLD-GATE] ork_native=1 ofmt=0 ...batched=1`.
- A/B: batched vs `ORK_MOE_COLD_BATCH=0` (per-row); vs ggml Q4_K native (no ork MoE).
- accuracy: perplexity (needs f16-NF4 pack for the real NF4 accuracy claim).

## RESULTS (2026-08-18) — o4n1 pack built (16.9 GB, 30930 weights)
- CPU int4 route EXECUTES: `[ork COLD-GATE] ork_native=1 ofmt=0(I4) cold_experts=223 cold_batch=1 -> batched=1`.
- **CONTAMINATION**: building with `ORK_QUANT=4` forced the DENSE/attn weights to int4-on-NPU too;
  at prefill they hit `bcreate FAIL errno=14 (Bad address)` (multi-domain IOVA mapping wall) → dense
  falls back slow. So these numbers are NOT a clean CPU-MoE measure — the dense path is degraded.
  At M=512 the dense NPU path crashes outright (`res=-3`); only M≤64 completes.
- pp64 CPU int4 **batched**  (ORK_MOE_CPU=1):                 **6.17 t/s**
- pp64 CPU int4 **per-row**  (ORK_MOE_CPU=1 COLD_BATCH=0):    **5.83 t/s**  → batched = **1.06× end-to-end**
  (vs 2.3× at the KERNEL level — the expert GEMM is a small slice of total prefill; dense/quant/scatter dominate).
- Native ggml mixed.gguf ceiling (no orkpack): **pp64 = 25.48 t/s** (pp512 hits the same M=512
  dense-NPU multi-domain wall → run errors after pp64; 25.48 is the valid M=64 ceiling).

### VERDICT (M=64, apples-to-apples M)
| config | pp64 t/s |
|---|---|
| native ggml mixed (dense f16/Q4_K, experts ggml Q4_K) | **25.48** |
| CPU int4 batched (ORK_MOE_CPU=1) | 6.17 |
| CPU int4 per-row (COLD_BATCH=0) | 5.83 |

CPU int4 route is ~4× slower than native — mostly the broken forced-int4 dense path, but even a clean
dense wouldn't flip it: the batched kernel's 2.3× is only 1.06× end-to-end, and native's ggml Q4_K
repack GEMM already batches the experts well. **The CPU int4/NF4 route does not beat native prefill.**
Route is BUILT + PROVEN CORRECT (bit-exact, executes); it just isn't a prefill win — exactly what
[[moe-w4a4-prefill-wall]] predicted (compute-bound on dequant; ~28 t/s native ceiling > 22 target).

### Interpretation
Consistent with [[moe-w4a4-prefill-wall]] + [[moe-ork-dense-only-experts-cpu-floored]]: CPU MoE prefill
is compute-bound on dequant; the batched int4 kernel (real 2.3× kernel win, bit-exact) barely moves
end-to-end because the expert GEMM isn't the dominant term and the forced-int4 dense path is broken on
NPU. The native mixed Q4_K path (~28 t/s, dense stays f16/Q4_K) remains the better prefill.
A clean CPU-int4 measure needs a pack that keeps DENSE at int8/f16 and only EXPERTS int4 — the current
ORK_QUANT=4 global forces dense int4 too. That, plus the f16-NF4 accuracy build, are the open items.

## NF4 REBUILD (2026-08-18, user: "redo the nf4 orkpack according to this need")
Clean recipe found via `ork_orkpack_tier` (line 988):
- `ORK_QUANT=4` forces int4 for EVERY tensor (→ dense int4 → crash). AVOID it.
- `ORK_ORKPACK_I4_FFN=1` forces int4 ONLY for ffn/exps/shexp tensors; attn/dense stay int8
  (source-driven: f16 → int8). No dense-int4 crash, better dense accuracy.
- NF4 codebook auto: `nf4 = ork_src_type_bits >= 16` → build from the **f16 gguf**.
- `ORK_MOE_CPU=1` to claim the MoE op in write mode (else experts don't persist).
- NO ORK_QUANT/HADAMARD → `ork_build_sig()==0`; LOAD must also omit them (sig must match).
BUILD: `ORK_PERSIST=qwen3.6-35b-a3b-nf4.orkpack ORK_ORKPACK_I4_FFN=1 ORK_MOE_CPU=1` on f16.gguf,
`-p 8 -n 0` (persist_write_experts packs ALL experts per tensor, so one short pass suffices).
NEW filename (learned from the w4a4 clobber): `qwen3.6-35b-a3b-nf4.orkpack`.
Rebooted board first (clean CMA) — prior runs left bcreate/CMA fragmentation.
LOAD/bench: `ORK_PERSIST=...nf4.orkpack ORK_ORKPACK_I4_FFN=1 ORK_MOE_CPU=1` (no ORK_QUANT) → expect
`[ork COLD-GATE] ork_native=1 ofmt=1(NF4) ...batched=1`. Compare vs o4n1 (uniform int4) + native (PPL).

## NF4 RESULT — PREFILL WIN (2026-08-18), reverses the earlier contaminated negative
Build OK: `qwen3.6-35b-a3b-nf4.orkpack` 18 GB, 31010 weights; tier log confirms `attn -> int8`,
experts -> NF4-int4. Load confirms `COLD-GATE ork_native=1 ofmt=1(NF4) batched=1`.
Same-fresh-board llama-bench (mixed.gguf base, -t4):
| config | pp64 | pp512 |
|---|---|---|
| native ggml mixed | 24.04 | 28.65 |
| **NF4 route (experts NF4-CPU + attn int8-NPU)** | **32.86** | **38.64** |
→ **~1.35× FASTER than native** at prefill. The earlier 6.17 "not a win" was the BROKEN dense-int4-NPU
path (ORK_QUANT=4 global) — a contaminated negative. With attn kept int8 + NF4 experts, the batched
NF4 CPU kernel (1-op vqtbl unpack) + int8 attn on NPU beats native's Q4_K-CPU experts.
CAVEAT — DECODE (M=1) under ORK_MOE_CPU is PATHOLOGICAL (~0.1 t/s): the batched kernel has no rows to
amortize. So ORK_MOE_CPU is a PREFILL lever only; decode must NOT use it (llama-cli hung 7 min on it).
VALIDATION: use `ork_bench <gguf> <promptfile> P G` (not llama-cli) — prefill P (fast NF4 path) + read
G generated tokens for coherence.

### VALIDATED (2026-08-18)
- Correctness: `ork_bench` "What color is the sky? Answer:" → generated **"1. Blue 2. Green 3. Red"**
  (coherent, correct). NF4 batched CPU expert path computes correctly.
- Decode fine: 3.99 t/s (~native). The llama-cli 7-min hang was CONVERSATION MODE (`-no-cnv` didn't
  take), NOT the MoE path. Use ork_bench or `-no-cnv`/`-st` for one-shot.
- CLEAN A/B (performance governor pinned — reboot had reset DDR to dmc_ondemand; -t4, r=3, same board):
| config | pp64 | pp512 |
|---|---|---|
| native ggml mixed | 24.18 | 27.20 |
| **NF4 route** | **31.18** | **36.19** |
| speedup | **1.29×** | **1.33×** |
→ **VALIDATED WIN: 1.33× over native at pp512.** Reverses the earlier contaminated 6.17 negative.

### Board tool note (user pointer): use `ork_bench <gguf> <prompt> [P] [G]`, not llama-cli, for ork
validation. Also ALWAYS re-pin governors after a reboot (dmc reverts to ondemand).

### OPEN
- Perplexity (NF4 accuracy vs native) — not yet measured (needs a corpus on-board; llama-perplexity or
  ork's own path). This is the remaining "how much accuracy does NF4 cost" number.
- Commit path: ork-driver changes (ork_native_cpu.h + test_i4_gemm + Makefile) need board `make test` +
  refreshed tests/sbc_attest.txt + OPS_REGISTRY row before commit. ggml-ork.cpp (ORK_MOE_CPU) is
  gated default-off. Decode-path optimization if ORK_MOE_CPU decode ever matters.
