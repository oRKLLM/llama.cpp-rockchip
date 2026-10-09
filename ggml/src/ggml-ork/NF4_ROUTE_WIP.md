# WIP — MoE NF4 auto-profile: DONE + MEASURED, blocked on `make test` (board hard-wedged 2026-08-19)

## STATE: nothing committed this session's decode/no-knob work. Board needs recovery first.

### Board (resume here)
HARD WEDGED. Sequence: perplexity run at `n_batch=2048` (llama-perplexity default) drove M=2048 wide
colsplit → `RKNPU_SUBMIT` timeouts (errno 110) + self-heal thrash → kernel NPU fault
(`rknpu_iommu_dma_map_sg` / `rknpu_gem_object_create` in dmesg) + D-state kworkers → a mid-submit SIGTERM
left the driver faulted → EVERY invocation segfaulted (even `ORK_OFF=1`) → `sudo reboot` came back once,
then a second forced reboot (to clear a `chain_xition_probe` zombie blocking `make test`) never brought
sshd back (ping OK, port 22 closed) → HA plug power-cycle (off/12s/on) → board did not return.
Per AGENTS.md, a plug cycle that does NOT restore it means likely **SPI bootloader corruption →
PHYSICAL SPI reflash, then boot from the Belkin supply** (SSD/rootfs was intact last time this happened).
**User is fixing the board manually tomorrow.** After recovery: **re-pin governors** (they reset every
boot): dmc + all cpu policies → `performance` (verify `/sys/class/devfreq/dmc/cur_freq` = 2112000000,
cpu6 = 2352000).

### THE ONE REMAINING TASK
Run `cd ~/llama.cpp/ggml/src/ggml-ork/ork-driver && sudo make test` on a CLEAN board, then commit.
- Last attempt hung on **`chain_xition_probe`** (~20 min, `timeout 360` didn't kill it because `timeout`
  SIGTERMs `sudo`, which doesn't forward to the child). That probe exercises `npu.c` chain transitions,
  which this session NEVER touched, and it hung on an already-degraded NPU — but that is UNPROVEN until
  it passes on a clean boot. Do not commit ork-driver until it does (or until the hang is shown pre-existing,
  e.g. by `git stash`-ing this session's changes and reproducing it).
- ATTEST note: `ATTEST_SRCS` = npu.c CORE + test_matmul/quant/test_sn3/model. This session changed NONE of
  them (header + new examples + tools + docs only), so `tests/sbc_attest.txt` is unchanged and CI
  `check-attest` will not fail regardless.

### WHAT'S UNCOMMITTED (all validated except the full-suite gate)
**ork-driver** (branch `feat/nf4-cpu-batched-gemm`, == main @ 2db74b9):
- `include/ork_native_cpu.h` — PRFM prefetch (`ORK_PRF_DIST` default **64** = one cache line, `ORK_PRF_LOC`)
  in `ork_dot_i4`/`ork_dot_nf4`/`ork_dot_nf4_x4` + the batched `ork_gemm_nf4_4x4`. Bit-exact (`test_i4_gemm`).
- `examples/test_nf4_decode.c` (NEW) — DRAM-realistic M=1 decode probe (NEXP distinct weights, 32 MB > L3)
  + achieved GB/s. THE instrument for memory work (`test_i4_gemm`'s single 0.5 MB weight stays L3-resident).
- `examples/test_moe_dispatch.c` (NEW) — dispatch-tax amortization probe (separate vs chain vs grouped vs
  one-program).
- `Makefile` — both probes in EXAMPLES. `OPS_REGISTRY.md` — CPU-kernel row updated + new probe-anchored
  prefetch row (`check-registry` PASSES). `AGENTS.md` — new "MoE profile is AUTOMATIC" section (measured
  table + the trade + the `--tensor-type attn=q8_0` model recipe + `ork_ppl` guidance). `README.md` — env
  table: derived path default, `ORK_ORKPACK_PATH` override, `ORK_PERSIST` retired.
- `tools/re/ork_bench.cpp` — reads `ORK_ORKPACK_PATH`; no longer sets `ORK_PERSIST`; sets the override only
  for the legacy non-default `.q<N>` variant (so the default path is knob-free end-to-end).
- `tools/re/ork_ppl.cpp` — docs: pin `ubatch` (the 2048 default is what wedged the board).

**parent llama.cpp-rockchip** (branch `feat/moe-decode-chain`, off `diverged-history` @ 2e7e75638):
- `ggml/src/ggml-ork/ggml-ork.cpp`:
  * **DECODE-FAST** fused M=1 MoE kernel (`ORK_MOE_DECODE_FAST`, default on): one activation quant per
    token (bcast-cached) + `ork_cpu_gemv_m1` DIRECT to dst, **persistent OpenMP pool** (`#pragma omp
    parallel for`, NOT per-op std::thread — that was the hidden overhead).
  * **MoE AUTO-PROFILE (no knobs)**: `g_ork_is_moe` latches on the first `GGML_OP_MUL_MAT_ID` at load-time
    graph planning → 4 sites: (a) tier expert/ffn→int4, (b) NF4 codebook by MODEL TYPE (not source bits, so
    one model = one pack scheme), (c) experts→batched CPU cold path + claim MUL_MAT_ID, (d) decline dense
    at M==1 (all-CPU decode) — **guarded off in WRITE mode** (`persist_mode==2`), else the pack build packs
    zero dense weights. Escape hatches: `ORK_MOE_AUTO=0`, `ORK_MOE_NPU`.
  * orkpack path DERIVED from the model as a fallback (`/proc/self/cmdline`) + `ORK_PERSIST` → `GGML_ABORT`
    with guidance. Header docs rewritten (auto profile + model recipe).
  * decode-chain dead end REMOVED (see below).
- `src/llama.cpp` — the fork ALREADY derived `<model>.orkpack`; renamed its published var
  `ORK_PERSIST` → `ORK_ORKPACK_PATH` (still `overwrite=0`, so an explicit override wins).
- Submodule bump to the new ork-driver commit is still TO DO (after `make test`).

### MEASURED (all on-board, governors pinned; q8attn base unless noted)
| config | prefill | decode | PPL (1024-tok window) |
|---|---|---|---|
| native ggml (ORK_OFF) | 23.3 | 7.5 | 10.2058 |
| **ork auto-profile (zero env)** | **36.9 (1.59×)** | 6.3–6.6 | 10.7718 (+5.5%) |
| " with auto-BUILT pack (NF4←Q4_K) | — | 6.33 | 10.8332 (+6.1%) |
Decode ladder: 3.60 (NPU attn) → 4.41 (all-CPU, std::thread) → 5.19 (OpenMP pool) → 5.30 (prefetch) on the
**mixed** base; then the base-file swap (f16→q8_0 attn) took it to **6.59**. Attn-precision ladder
(prefill/decode): f16 38.4/5.26 | q8_0 36.9/6.59 | Q4_K 26.8/7.42 (<5 bits ⇒ NPU declines the source).
Zero-env verified: `orkpack derived from model` + `MoE model detected -> auto profile` + coherent
"1. Blue 2. Green 3. Red". Thermals clean (57.3 °C sustained, no throttle). Flat/dead: KV-quant at short
ctx, 2-core single-cluster (4.33 < 5.26), `OMP_WAIT_POLICY=ACTIVE`, in-tree ggml integration (≤1 t/s).

### DEAD ENDS (measured, do not retry)
- NPU decode-chain (`run_chain_i4` over M=1 experts): 0.70 t/s (5× slower) — per-token resolve/quant/
  residence for the varying active set dominates. Code stripped; probe kept.
- Experts on NPU: IOVA-walled (int8-inflate ~35 GB ≫ 4 GB; int4-native fits but the kernel is ~54× slower).
- Dense all-256-experts (XAMBA-style): 32× compute at prefill, read-all-16 GB at decode.

### NEXT (after `make test` + commits)
1. Commit ork-driver (kernel+probes+tools+docs+registry) → ff into main; commit parent (DECODE-FAST +
   auto-profile + llama.cpp rename + submodule bump) → ff into `diverged-history`.
2. Then: `scratchpad/PLAN_gdn_chunkwise_npu.md` — GDN chunk-GEMM offload. KEY FINDINGS ALREADY MADE:
   ork's `SSM_SCAN` gate is **Mamba-2-only** (`A->ne[0]==1`) so it declines GDN entirely; llama.cpp
   ALREADY emits GDN chunkwise (`k_cumdecay`/`v_t_new` intermediates) as dynamic-src0 matmuls that ork
   declines by default; `ORK_ATTN=1` is the existing opt-in to claim them. Stages 0-1 are ENV-ONLY
   measurements (op breakdown, then `ORK_ATTN` A/B) — do those before any code.
   NOTE `task_number>1` chaining DOES work (AGENTS §"Multi-task hardware chaining"; `ork_dyn_begin_mc`,
   measured 3.4×/3.67× amortization) — the earlier "kernel rejects task_number>1" claim was WRONG.

### BOARD FILES
`qwen3.6-35b-a3b-q8attn.gguf` (20.2 GiB, 5.00 BPW — the recommended base) + `…-q8attn.orkpack` (18.4 GB,
auto-built, NF4 experts) | `…-nf4.orkpack` (18.9 GB, f16-sourced NF4) | `…-w4a4.orkpack` (16.9 GB
I4_NATIVE, rebuilt after the earlier clobber) | `…-mixed/-q4km/-f16.gguf` | `/tmp/ppl_corpus.txt` is
REGENERATE-ON-BOOT (repo docs, 12879 words) — /tmp is cleared each boot, as are the prompt files.
