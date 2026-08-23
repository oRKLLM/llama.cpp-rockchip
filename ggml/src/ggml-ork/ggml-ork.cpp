// ork-driver NPU matmul backend for ggml (Rockchip RK35xx).
//
// Modeled on the BLAS backend: a mul-mat-only accelerator that offloads GGML_OP_MUL_MAT to the
// Rockchip NPU via ork-driver and leaves all other ops to the CPU backend. Uses the int8 (W8A8)
// path: weights are dequantized then per-channel int8-quantized and packed once (cached, NPU-
// resident); activations are per-row int8-quantized each call; the NPU computes the int32 product
// which is dequantized (aScale[m]*bScale[n]) into the fp32 dst. ~1% vs fp32 on real weights, half
// the weight bytes of fp16. (int4/W4A4 + per-group scales is the next step down.)
//
// =================== MoE AUTO-PROFILE (no env knobs — the MODEL TYPE decides) ===================
// A routed-MoE model (any GGML_OP_MUL_MAT_ID op, detected at load-time graph planning) automatically
// selects the MEASURED-OPTIMAL scheme for this hardware. NOTHING needs to be set:
//   * orkpack tier   : expert/ffn tensors -> int4 (NF4 codebook when the source is f16+), attn/dense int8
//   * prefill (M>1)  : experts on CPU via the batched 4x4 NF4/int4 NEON GEMM (ork claims MUL_MAT_ID);
//                      attn/dense int8 on the NPU
//   * decode  (M==1) : ALL-CPU — the dense/attn backbone is declined (the NPU per-submit floor loses at
//                      M=1), experts use the fused M=1 gemv fast-path (one quant/token, direct-to-dst,
//                      persistent OpenMP pool) with one-cache-line PRFM prefetch
// Measured, qwen3.6-35B-A3B, RK3588 -t4, governors=performance, q8_0-attn base (see MODEL RECIPE below):
//   prefill 36.9 t/s (1.59x native ggml 23.3) | decode 6.59 t/s | coherent | no thermal throttle.
// MODEL RECIPE (a FILE property, not a runtime knob — the single biggest decode lever measured):
//   llama-quantize --tensor-type attn=q8_0 --tensor-type shexp=q8_0 <f16.gguf> <out.gguf> Q4_K_M
//   Attention precision trades prefill for decode: f16 38.4/5.26, q8_0 36.9/6.59, Q4_K 26.8/7.42 — and
//   BELOW 5 bits the NPU declines the source entirely (sbits<5.0 gate), losing the prefill path. q8_0 is
//   the balanced optimum. Expert precision in the GGUF is irrelevant (NF4 comes from the .orkpack).
// The .orkpack path is DERIVED from the loaded model (<model-dir>/<model-basename>.orkpack) — no env var
// needed; an absent pack is built once, and a MoE model always gets the SAME scheme (NF4 experts) whatever
// the GGUF's own precision, so one model = one pack. ORK_ORKPACK_PATH is a DEVELOPMENT override only.
// ORK_PERSIST is REMOVED and now aborts with guidance.
// Research escape hatches: ORK_MOE_AUTO=0 disables the profile; ORK_MOE_NPU forces experts to the NPU.
//
// ============================ ENVIRONMENT FEATURE FLAGS ============================
// All experimental paths are OFF by default; the default build is the validated stable baseline
// (dense MUL_MAT offload to NPU; everything else on CPU). Set a flag on the runtime command line.
//
//   ORK_MOE_NPU=1        EXPERIMENTAL, default OFF, NOT recommended on RK3588. Offload MoE experts
//                        (MUL_MAT_ID) to the NPU via the hot-expert partition (conforming-K experts go
//                        NPU-resident on the async stream; the rest run the threaded CPU GEMV). M2
//                        verdict on RK3588: decode ~6.56 vs CPU ~19.09 t/s (~2.9x SLOWER) — walled by
//                        ~1.2 GiB usable IOVA (32-bit IOMMU; ~17% hit-rate) and the LPDDR4X-bound cold
//                        GEMV. Revisit on wider-IOVA / DDR5 devices or the M>1 batched-verify path.
//                        Value-checked (truthy 1/true/yes/on; UNSET or 0/false/off => experts on CPU).
//                        Requires the repack-buffer exclusion in ggml-cpu/repack.cpp (presence-gated on
//                        the same flag). Legacy alias: ORK_NO_EXPERT_REPACK.
//   ORK_MOE_CACHE=<n>    Resident expert-pool slots PER SHAPE (default 384); reused round-robin via
//                        ork_i8_mm_repack (no IOMMU churn). Only relevant when ORK_MOE_NPU is on.
//   ORK_MOE_ALL_ACTIVE   STAGE 1 (default ON when ORK_MOE_NPU). At batch M>=ORK_MOE_BATCH_MINM admit
//                        ALL active experts to the NPU (the M-sweep probe's optimum), bounded only by
//                        the IOVA budget (ORK_MOE_HOT_GIB), not the hot_K count cap. =0 reverts to the
//                        pure hot_K LRU policy. STAGE-1 IN-MODEL VERDICT (LFM2.5-8B-A1B prefill, board
//                        10.3.0.236): the probe's M>=8 win does NOT survive — pp128 12.8 vs native-fused
//                        36.3 t/s (0 EFAULT, coherent). ggml's CPU MoE is a FUSED batched kernel; our
//                        per-expert split loses ~30% before any NPU (all-cold 25.4), and the serial NPU
//                        submit makes it worse. Default OFF via ORK_MOE_NPU; see STAGE1_MOE_BATCHED_WIP.
//   ORK_MOE_BATCH_MINM=<n>  Per-expert routed-row threshold (M_e) for the all-active regime (default 8,
//                        matching the probe crossover). Below it (decode) the hot_K LRU cap applies.
//   ORK_MOE_PATHB=1      EXPERIMENTAL, default OFF. PATH (b): at the batched regime (max M_e>=BATCH_MINM,
//                        conforming K) split this tensor's active experts into an NPU share (run_stream_i8
//                        on a DEDICATED thread, overlapped) ‖ a CPU share computed through ggml's REAL
//                        FUSED batched MUL_MAT_ID kernel (a compacted sub-graph on a cached CPU backend —
//                        NOT the per-expert vec_dot loop). Fixes the two Stage-1 losses (lost fusion +
//                        serial submit). Below BATCH_MINM (decode) falls through to the all-CPU path.
//   ORK_MOE_PATHB_FRAC=<f>  Fraction of ACTIVE experts (by largest M_e first) routed to the NPU under
//                        PATH (b) (default 0.5; 0=all CPU-fused, 1=all NPU). Sweep to find the crossover.
//   ORK_MOE_CPU_THREADS=<n>  CPU-backend thread count for the PATH (b) fused sub-graph (default 4 = big cores).
//   ORK_MOE_PATHB_REPACK=1  PATH (b): repack the CPU-share weights into the SAME tiled layout the native
//                        fused MUL_MAT_ID uses (a cached repack-buffer copy per experts tensor), so the
//                        CPU share dispatches the REPACKED kernel — a fair fight vs the repacked baseline.
//                        Default OFF (then the CPU share uses the slower standard fused kernel).
//   ORK_MOE_PATHB_PARK=1  PATH (b): keep the CPU sub-graph in the NATIVE-efficient layout (b'=src1, full
//                        ids' with NPU slots redirected to a park expert) instead of compacting into
//                        single-expert columns. Matches the native per-token batching (much faster CPU
//                        share); cost = the park expert recomputes the NPU slots' rows (discarded).
//   ORK_OFF=1            Diagnostic: force EVERYTHING to CPU (supports_op returns false). Same-binary
//                        CPU baseline for A/B benchmarks.
//   (QKV/gate-up group fusion is DEFAULT-ON for M>=2 (ORK_FUSE_MINM): +11% @M2 .. +17% @M64, bit-exact,
//    decode M=1 untouched — see graph_compute. ORK_NO_FUSE disables; ORK_FUSE forces fusion at ALL M.)
//   ORK_QUANT=4          int4 tier — a BUILD-TIME override, not a run-time mode. Set it for the run that CREATES
//                        the .orkpack; afterwards the pack is self-describing (its footer records the tier) and
//                        loading it selects int4 on its own, so steady-state runs need nothing set. Also use it
//                        to deliberately rebuild an existing pack at a different tier. int4 = compact int4
//                        STORAGE + W8A8-inflate compute on the NPU (route B — int4 weights inflate int4->int8 and
//                        run on the robust int8 kernel; coherent, no wedge). The win is the .orkpack STORAGE
//                        (~3.4× smaller than int8), not the compute (~int8 speed). Native W4A4 (single-row,
//                        wedge-prone at prefill) is OPT-IN via ORK_MIXED_W4A4.
//                        *** RECOMMENDED int4 setup: build the orkpack from the model's UNQUANTIZED (F16/F32/BF16)
//                        GGUF — the NF4 codebook is then auto-selected (best fidelity; building from an already-
//                        quantized source is warned and falls back to the lossier uniform int4). Measured Qwen3-1.7B
//                        @ P=128 (RK3588): NF4-from-F16 = 215 tok/s prefill / 6.77 decode (edges the int8 ref ~178)
//                        vs uniform-from-Q8 = 172 / 2.96. So: F16 source -> ORK_QUANT=4 -> compact NF4 .orkpack. ***
//   (no ORK_HADAMARD)    The block-Hadamard rotation is IMPLIED by NATIVE W4A4 and cannot be turned off:
//                        RK3588's W4A4 MAC is symmetric per-channel and incoherent unrotated (PPL ~104 vs ~24),
//                        so un-rotated 4-bit compute was never a usable configuration — only a way to get a
//                        silently garbage run. Note this is about the 4-bit COMPUTE path (ORK_MIXED_W4A4), not
//                        the int4 STORAGE tier (ORK_QUANT=4), which computes W8A8 and rotates nothing. See
//                        ork_w4a4_native_on for why ggml-ork owns this rule and ork-driver cannot.
//   ORK_MIXED_DISPATCH=1 Per-tensor dispatch driven by the GGUF's OWN mixed quantization: accept sub-5-bit
//                        (q4) sources onto the NPU (instead of CPU) and pick the compute path per tensor by
//                        source precision. Default: 4-bit tier computes W8A8 (dequant q4->int8, fast +
//                        coherent ≈ Q8_0; the compact-int4 win is realized in the .orkpack STORAGE, not the
//                        compute — native W4A4 prefill is single-row/no-M-amortization so it loses); >4-bit
//                        stays W8A8. On Q4_K_M this gives NPU prefill (~3.8× CPU) with a compact mixed .orkpack.
//   ORK_MIXED_W4A4=1     With ORK_MIXED_DISPATCH: opt the 4-bit tier into NATIVE W4A4 compute (int4 MAC)
//                        instead of W8A8. For the decode/streaming regimes where int4's bandwidth wins;
//                        loses at prefill (measured run ~54× the W8A8 path — see ORK_PROFILE [ork i4prof]).
//   ORK_ZC_OUT=1         EXPERIMENTAL/BUGGY. Output zero-copy (single-tile ~90% wrong). Off.
//   ORK_HYBRID=1         EXPERIMENTAL. Hybrid CPU/NPU weight loading.
//   ORK_MINM=<n>         Min M to route a matmul to the NPU (default 32). Tuning, not experimental.
//   ORK_NOREUSE=1 / ORK_NOCACHE=1   Disable activation reuse / weight cache (debug).
//   ORK_NO_AFFINITY=1    Don't pin NPU-driver threads to big cores (default: pin).
//   ORK_PROFILE=1        Per-section timing, printed on backend free.
//   ORK_VERBOSE=1        Verbose per-op trace to stderr (debug).
//   ORK_ORKPACK_TIERMAP=<file>  (write/convert) name<TAB>tier map from gguf_tier_map --emit-map:
//                        a tensor's int4/int8 tier comes from this map BY NAME (overrides the
//                        source-type verdict). Lets an fp16 source GGUF inherit a Q4_K_M's
//                        int4/int8 ALLOCATION while quantizing VALUES from the clean fp16.
//   ORK_IMATRIX=<file>   (write/convert) GGUF imatrix (llama.cpp in_sum2/counts form). For each
//                        int4-tier tensor, load its per-INPUT-channel importance (in_sum2/counts,
//                        length == packer K) and pack via ork_i4a8_mm_pack_im (imatrix-weighted
//                        clip-grid scale). NF4 + imatrix compose.
// ===================================================================================

#include "ggml-impl.h"
#include "ggml-ork.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"     // ggml_get_type_traits_cpu: reuse the NEON-optimized expert vec_dot for cold experts
#include "ggml-alloc.h"   // PATH (b): ggml_backend_alloc_ctx_tensors_from_buft (repacked CPU-share sub-graph)
// PATH (b): the CPU repack buffer-type getter is internal to ggml-cpu (repack.h) but exported in the
// shared lib (C++ linkage); forward-declare it so the CPU-share sub-graph can repack its weights into
// the SAME tiled layout ggml's native fused MUL_MAT_ID uses — a fair fight against the repacked baseline.
ggml_backend_buffer_type_t ggml_backend_cpu_repack_buffer_type(void);
#include "gguf.h"

#include <vector>
#include <map>
#include <algorithm>
#include <cstring>
#include <cstdlib>   // LEVER3: atexit
#include <cmath>
#include <ctime>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <deque>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <sched.h>
#include <pthread.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

extern "C" {
#include "ork_npu.h"
#include "ork_spine.h"   // generalized heterogeneous CPU<->NPU DAG dispatcher (ORK_FFN_SPINE path)
#include "ork_native_cpu.h"   // #12: CPU-side NEON GEMV over the ork-native tiered format (shared with the NPU)
}

#include <sys/mman.h>
#include <cstdint>
#include <climits>   // INT_MAX (PATH B down_minM default-off)
#include <string>
#include <fcntl.h>
#include <unistd.h>
#include <strings.h>   // strcasecmp (env_enabled)

// Truthy-VALUE env gate (not mere presence). Returns true only when the named var is set to one of
// 1/true/yes/on (case-insensitive); UNSET or 0/false/off/empty -> false. Use this for any flag whose
// "off" must be expressible as VALUE=0 (the obvious way to disable a feature on a command line), not
// only by unsetting it — a bare getenv()!=nullptr presence-check treats `FOO=0` as ENABLED, a footgun.
static bool env_enabled(const char * name) {
    const char * v = getenv(name);
    if (!v || !*v) return false;
    return v[0]=='1' ||
           !strcasecmp(v, "true") || !strcasecmp(v, "yes") || !strcasecmp(v, "on");
}

// ---- .orkpack persist format: a self-populating on-disk cache of pre-tiled (mixed int8 / int4) weights ----
// File: [ blobs: per weight, packed bytes then (int8 only) N bscale floats ][ index ][ footer@EOF ].
// First run writes it (one slow pass) to the DERIVED <model>.orkpack; later runs mmap it and load the bytes
// straight into DMA — no dequant/quant/tile. Each weight's (K,N,dtype) is re-checked on load AND the
// footer carries ork_pack_format_version() (ork-driver's MAJOR ver): a tile-layout / quant change bumps
// that major, so an incompatible file is rejected wholesale at startup and regenerated (the read path
// falls through to write mode). (K,N,dtype) alone can't catch a tiling change — same-(K,N) blobs from
// an incompatible major have identical size — which is exactly what the token guards.
//
// Per-tensor tier is carried in `dtype` (the field is back-compatible — v1 files only ever wrote dtype==1):
//   dtype == ORKPACK_DT_I8 (1): blob = ork_w_dump bytes (tiled int8), followed by bscale_n bscale floats.
//   dtype == ORKPACK_DT_I4 (4): blob = ork_i4a8_w_dump bytes (self-describing 'O4N1': K,N,quant_kind +
//                               bscale[N] + nibble store). bscale lives INSIDE the blob → bscale_n==0,
//                               bscale_off unused. Loaded via ork_i4a8_mm_load, runs via ork_i8_mm_run.
//   dtype == ORKPACK_DT_I4_NATIVE (5): blob = ork_w_dump bytes of a DT_I4 (native-W4A4) weight — the
//                               FWHT-rotated, per-channel-int4-quantized, int4-TILED bytes — followed by
//                               bscale_n bscale floats. Loaded via ork_i4_mm_load, runs W4A4 via
//                               ork_i4_mm_run (the mul_mat_i4_hadamard / group_i4 compute path). This is
//                               the cold-pack fix: the expensive dequant->rotate->int4-quant->tile is done
//                               ONCE at convert and reloaded as a plain DMA copy.
// The struct layout is unchanged from v1, so v1 (all-int8) files load unmodified; VERSION bumps to 2 to
// mark files that may contain int4 entries (both versions are accepted on read).
#define ORKPACK_MAGIC   "ORKPK01"
#define ORKPACK_VERSION 5u   // v5 adds quant_sig (build-config precision signature) to the footer; v4 adds bf_size
                             // to each entry (full-K Bf blob after the Bb blob, so orkd maps Bf directly); v3 adds ork_fmt.
#define ORKPACK_DT_I8         1u
#define ORKPACK_DT_I4         4u
#define ORKPACK_DT_I4_NATIVE  5u
// bf_size>0 (int8 tier only) => bf_size bytes of the full-K Bf blob follow the Bb blob contiguously (i.e. at
// blob_off + blob_size), before bscale_off. 0 => no Bf (K outside the Bf envelope, or a pre-v4 concept).
struct orkpack_entry  { uint32_t K, N, dtype, bscale_n; uint64_t blob_off, blob_size, bscale_off, bf_size; };
// The footer is the pack's self-describing header metadata (EXIF-style): validation keys on it, NOT the filename.
//   ork_fmt   = ork_pack_format_version() at write — a tile-layout/quant MAJOR change bumps it => tiled bytes incompatible.
//   quant_sig = ork_build_sig() at write — the build-config PRECISION signature (forced ORK_QUANT + hybrid + hadamard).
//               The same (K,N) tensor packs to DIFFERENT bytes under int4 vs int8, and ork_fmt does NOT distinguish
//               them, so a stored quant_sig != this run's => wrong precision, file rejected + regenerated. The .q4/.q8
//               filename is only a convenience so both can coexist on disk; THIS field is the authoritative guard.
// magic stays last so it remains the final 8 bytes of the file regardless of footer growth. Adding a field bumps VERSION.
struct orkpack_footer { uint64_t index_off; uint32_t n_entries; uint32_t version; uint32_t ork_fmt; uint32_t quant_sig; char magic[8]; };

// Build-config precision signature stored in the footer (see above). Env-derived so the standalone validity check
// (pre-init, no ctx) and the write path compute it identically. Encodes the knobs that change PACKED CONTENT:
// forced quant precision (ORK_QUANT) and hybrid split (ORK_HYBRID).
//
// The precision field is DESCRIPTIVE, not prescriptive. It records the tier a pack was BUILT at so that loading
// the pack SELECTS that tier (ork_sig_qbits -> ctx->persist_qbits) — an int4 .orkpack drives the int4 path with
// no env set. It used to be a pure equality gate, which meant an int4 pack was REJECTED and rebuilt as int8
// unless the run happened to re-supply the same knobs. Only ORK_SIG_HY_BIT stays prescriptive: hybrid changes
// WHICH tensors are packed at all, so a mismatch there is a genuinely unusable file.
#define ORK_SIG_QB_MASK 0x0ffu   // forced-precision char: '4', '8', or 0 = source-driven default
#define ORK_SIG_HY_BIT  0x100u   // ORK_HYBRID split
#define ORK_SIG_HD_BIT  0x200u   // hadamard — now IMPLIED by native W4A4 (see ork_w4a4_native_on); vestigial in the sig
static uint32_t ork_build_sig(void) {
    const char * q = getenv("ORK_QUANT");
    uint32_t qb = (q && *q) ? (uint32_t) (unsigned char) q[0] : 0u;   // '4','8',… or 0 = source-driven default
    uint32_t hy = (getenv("ORK_HYBRID") != nullptr) ? 1u : 0u;
    // hd is DERIVED, not read: native W4A4 is always rotated (see ork_w4a4_native_on), so it carries no independent
    // information. Deriving it keeps the emitted value bit-identical to what the old ORK_QUANT=4 +
    // ORK_HADAMARD=1 build wrote (0x234) and what a plain int8 build wrote (0x0) — no existing pack is
    // invalidated by removing the knob.
    uint32_t hd = (qb == (uint32_t) '4') ? 1u : 0u;
    return (qb & ORK_SIG_QB_MASK) | (hy << 8) | (hd << 9);
}

// The weight tier a pack was BUILT at, decoded from its stored signature. This is what makes an .orkpack
// self-describing: load an int4 pack and you get the int4 path.
static int ork_sig_qbits(uint32_t sig) {
    return ((sig & ORK_SIG_QB_MASK) == (uint32_t) '4') ? 4 : 8;
}

// NATIVE W4A4 — the only path that actually issues 4-bit MACs (ork_i4_mm_*). Opt-in via ORK_MIXED_W4A4,
// because it is fragile at prefill (single-row, no M-amortization).
//
// ORK_QUANT=4 does NOT select it. That is the int4 STORAGE tier: weights are stored 4-bit, then inflated
// int4->int8 on the NPU and computed by the robust W8A8 kernel. No 4-bit MAC ever runs, so there is nothing
// to rotate — which is exactly why the storage tier is the coherent, recommended int4 default.
//
// Where a 4-bit MAC DOES run, the block-Hadamard rotation is UNCONDITIONAL. RK3588's W4A4 is symmetric
// per-channel and incoherent unrotated (PPL ~104 vs ~24), so un-rotated W4A4 was never a configuration worth
// selecting — only a way to get a silently garbage run. Hence the old ORK_HADAMARD opt-in is gone, and the
// un-rotated grouped route (mul_mat_i4) with it: selecting native W4A4 now means rotated, always.
//
// ork-driver cannot own this rule. A block-Hadamard is orthogonal, so (R*A)·(R*B) == A*B and the driver's
// matmul is rotation-invariant by construction — it never observes whether the caller rotated, so it can
// neither enforce nor check. ggml-ork applies R to the weight column at pack AND to the activation row at
// run (both at the same block size), so ggml-ork is the only layer that can hold the invariant.
static bool ork_w4a4_native_on(void) { return env_enabled("ORK_MIXED_W4A4"); }

// Is a pack with this stored signature usable by this run? Precision is ADOPTED, not required to match —
// unless ORK_QUANT explicitly forces a tier, in which case a conflicting pack is stale and gets rebuilt.
static bool ork_sig_compatible(uint32_t sig) {
    if ((sig & ORK_SIG_HY_BIT) != (ork_build_sig() & ORK_SIG_HY_BIT)) return false;
    const char * q = getenv("ORK_QUANT");
    if (q && *q) return ork_sig_qbits(sig) == ((q[0] == '4') ? 4 : 8);
    return true;
}

// Custom-loader memory relief: once a weight is packed NPU-resident, its source GGUF plane is dead weight.
// Evicting those mmap'd pages keeps the source's RSS shrinking as packed RSS grows (peak ~max(src,packed)
// not src+packed). Page-aligned MADV_DONTNEED drops only clean, file-backed pages (re-faulted on demand);
// opt-in via ORK_EVICT_SRC because with --no-mmap the mapping is anonymous and DONTNEED would zero data.
static void ork_evict_src(const void * p, size_t n) {
    static int on = -1;
    if (on < 0) on = getenv("ORK_EVICT_SRC") ? 1 : 0;
    if (!on || !p || !n) return;
    uintptr_t a = (uintptr_t) p, end = a + n;
    uintptr_t pa = (a + 4095u) & ~(uintptr_t) 4095u;
    uintptr_t pe = end & ~(uintptr_t) 4095u;
    if (pe > pa) madvise((void *) pa, (size_t) (pe - pa), MADV_DONTNEED);
}

struct ggml_backend_ork_context;   // fwd
static size_t g_ork_weight_bytes = 0;   // diagnostic only; ork weights live on CPU buffers, so this stays ~0
// Residence budget = the process RAM ceiling (total RAM - 1 GiB for the OS). Weights reside across the
// IOMMU domains (each ~4 GiB; up to ~64 GiB total), and the wcache tracks the resident set; sizing the
// budget to system RAM keeps a model that fits FULLY resident (no eviction, no churn). Only a model
// larger than this budget evicts the LRU weight. This is the oRKLLM global memory limit — override with
// ORK_WCACHE_BUDGET_MB (min(model, frontend cap, sysram-1GiB)); unset = sysram-1GiB.
static size_t ork_wcache_budget() {
    const char * e = getenv("ORK_WCACHE_BUDGET_MB");
    if (e) return (size_t) atoll(e) * 1024 * 1024;
    size_t ram = (size_t) sysconf(_SC_PHYS_PAGES) * (size_t) sysconf(_SC_PAGE_SIZE);
    size_t rsv = (size_t) 1024 * 1024 * 1024;
    return ram > rsv ? ram - rsv : (ram ? ram : (size_t) 4 * 1024 * 1024 * 1024);
}

// ---- STREAM-POOL tier (ORK_STREAM_POOL=1) -------------------------------------------------------
// Retarget the weight-streaming LRU at a RAM-resident inflated-int8 cache (ork_stream_pool) instead of
// pure IOVA residency. Two budgets form a hierarchy:
//   RAM-int8 LRU  (ORK_STREAM_RAM_BUDGET_MB, big — much larger than the 4 GiB IOVA window): how much
//                 inflated int8 we hold resident in CPU RAM. A hit here SKIPS the expensive re-inflate.
//   IOVA hot tier (ork_wcache_budget(), <= 4 GiB, evict-RARE): the subset currently MEM_CREATE-mapped.
// Measured uABI costs drive this: inflate(fill) 0.4-2.7ms + MEM_DESTROY(unmap) 0.5-2ms are expensive
// (paid once on add / only on eviction); MEM_CREATE(map) is cheap (~48-169us, paid per (re)map on a hit).
// So: keep hot entries MAPPED, evict the IOVA mapping rarely, and NEVER re-inflate on a RAM hit.
// RAM tier = the process RAM ceiling (total RAM - 1 GiB for the OS): how much inflated int8 we hold
// resident in CPU RAM (>> the 4 GiB IOVA window). This is the oRKLLM global memory limit; override with
// ORK_STREAM_RAM_BUDGET_MB. The hot IOVA subset is capped separately by ork_wcache_budget() (< 4 GiB).
static size_t ork_stream_ram_budget() {
    static size_t b = 0;
    if (!b) {
        const char * e = getenv("ORK_STREAM_RAM_BUDGET_MB");
        if (e) b = (size_t) atoll(e) * 1024 * 1024;
        else { size_t ram = (size_t) sysconf(_SC_PHYS_PAGES) * (size_t) sysconf(_SC_PAGE_SIZE);
               size_t r = (size_t) 1024 * 1024 * 1024; b = ram > r ? ram - r : (ram ? ram : (size_t)10240*1024*1024); }
    }
    return b;
}

// a packed quantized weight + its scales, kept NPU-resident and reused.
//   int8 (W8A8):  gsize==0, bscale [N]            (per output channel)
//   int4 (W4A4):  gsize==G,  bscale [(K/G)*N]      (per K-group, per channel)
struct ork_weight {
    ork_w * w = nullptr;
    ork_stream_entry * se = nullptr;   // STREAM-POOL tier: RAM-resident inflated int8 (map/unmap cheap)
    std::vector<float> bscale;
    int gsize = 0;
    size_t   bytes = 0;       // resident NPU bytes (for the streaming LRU budget)
    size_t   ram_bytes = 0;   // STREAM-POOL: RAM bytes held by `se` (for the RAM-LRU budget)
    uint64_t last_use = 0;    // monotonic tick of last access (LRU eviction order)
    bool     is_expert = false;   // MoE STREAM: a routed-expert slice (evicted by ork_wcache_evict_experts; dense stays)
};

// One reusable slot in the MoE expert pool: a packed weight whose DMA buffer is reused (repack-in-place)
// across different experts of the SAME shape, so the NPU IOMMU isn't churned/fragmented by alloc+free.
struct ork_moe_slot {
    ork_w * w = nullptr;
    std::vector<float> bscale;
    const void * key = nullptr;       // host ptr of the expert currently packed here (nullptr = empty)
};

// Fused-attention (GGML_OP_FLASH_ATTN_EXT) scratch. PER-CONTEXT (was the process-global g_attnp) so two ork
// backends can run real attention graphs concurrently without clobbering each other's QK^T/scores/softmax
// buffers. Pool = NPU scratch weights (wqk/wav/wet) + host densify/score buffers, warm-reused across calls of
// the same shape. attn_sm = the softmax scratch (int8 scores, per-row max, exp/reduce/normalize buffers).
// `dom` = the IOMMU domain this pool's fp16 scratch weights live in. The pool is keyed by domain (one pool
// per domain, see ctx->attnp) so the fused attention always runs in the domain that is ALREADY ACTIVE and
// therefore adds NO `switch iommu domain` — the switch is what a stuck doorbell job turns into a 60 s stall
// on the next submit. See attn_pool_ensure.
struct ork_attn_pool { int DK,DV,N,nkvp,H,dom; ork_w **wqk, **wav, **wet;
    ork_f16 *Qf, *KT, *Vf, *Pf, *Oh16; float *scores, *outf, *invS; ork_mm_task_f16 *tk; };
struct ork_attn_sm {
    int16_t *xi=nullptr,*ei=nullptr; float *mx=nullptr,*ss=nullptr; ork_f16 *e=nullptr; size_t cap=0;
    int8_t *q8=nullptr; size_t cap2=0; int8_t maxq[8192];
    ork_f16 *invf=nullptr; int invf_n=0;
    ork_w *ones=nullptr; int ones_n=0;
};

struct ggml_backend_ork_context {
    ork_npu * npu = nullptr;
    bool via_orkd = false;     // true = NPU access routed through the orkd daemon (ork_npu_uses_orkd). In this
                               // mode the client has NO local NPU fd, so weight residency + matmuls must use the
                               // ORKD-ROUTED driver APIs only (ork_i8_mm_pack / ork_i8_mm_run — the daemon owns
                               // the buffers). The zero-copy .orkpack import, DMA activation buffers, and the
                               // fused/stream/chain/MoE runners are fd-local (not routed) and are gated OFF here.
    int qbits = 8;              // 8 = W8A8, 4 = W4A4. Adopted from the loaded .orkpack (persist_qbits); ORK_QUANT overrides.
    int persist_qbits = 0;     // tier the loaded .orkpack was BUILT at (4/8), 0 = no pack read. Set by ork_persist_init.
    int hadamard = 0;          // 1 iff NATIVE W4A4 is selected; rotation is inseparable from it (see ork_w4a4_native_on)
    int no_reuse = 0;          // ORK_NOREUSE=1: disable activation-quant reuse (A/B benchmark)
    int no_cache = 0;          // ORK_NOCACHE=1: re-pack the weight every matmul (A/B benchmark)
    bool slice_route = false;  // ORK_SLICE_ROUTE=1: route wide int8 matmuls (K>4096 / N>8192) through the
                               // sliced doorbell (ork_mm_run_sliced) instead of the M>1 blocking mcworker —
                               // wedge-safe (doorbell owns the submit). Default off; A/B vs the blocking path.
    std::unordered_map<ork_w *, ork_w_sliced *> slice_ws;   // resident wide weight -> its sliced-doorbell pack
    std::unordered_map<ork_w *, std::unordered_set<int>> slice_forced;   // memo: (weight -> M's) proven wedge-prone this session -> skip the doomed run_i8, go straight to sliced
    bool hybrid = false;       // use hybrid loading (FFN 4-bit, Attn 8-bit)
    std::vector<float>    f32;   // dequantized src0 plane [N*K] (cache-miss scratch)
    std::vector<int8_t>   bi;    // weights quantized int8 B[K*N] (cache-miss scratch)
    std::vector<int8_t>   ai;    // activations quantized int8 A[M*K]
    std::vector<float>    as;    // per-row activation scale [M]
    std::vector<int32_t>  ci;    // int32 matmul result [M*N] before dequant
    std::vector<float>    arot;  // rotated activation row [K] scratch (Hadamard int4 path)
    // Static-graph DMA scratch (ORK_GU_CHAIN): NPU-coherent, alloc-once/grow buffers for the FFN segment's
    // shared int8 input + the gate/up int32 outputs. Passing these to run_chain_i8 makes dma_find HIT, so it
    // skips the per-task bcreate/memcpy/bsync/bdestroy (the ioctl bulk) and dedups the shared-input sync.
    void * dma_in = nullptr;  size_t dma_in_sz = 0;
    void * dma_up = nullptr;  size_t dma_up_sz = 0;
    void * dma_gt = nullptr;  size_t dma_gt_sz = 0;
    void * dma_moeC[64] = {}; size_t dma_moeC_sz[64] = {};   // #14 multi-core NONBLOCK: per-domain resident MoE output. Sized to the system domain ceiling (orkd owned_dom is a 64-bit mask -> 63 domains, ~155 GiB, beyond any RK3588); the auto-sizer picks the actual count. No small fixed cap.
    // model weights are constant during inference, so pack+quantize each once (NPU-resident) and
    // reuse, keyed by the weight plane's host pointer. The transformer pattern ork-driver is for.
    std::unordered_map<const void *, ork_weight> wcache;
    // Keys the ACTIVE multi-weight op (e.g. the FFN SwiGLU chain) still holds references to. A multi-weight
    // op resolves several weights up-front and uses them across later phases; a subsequent resolve's LRU
    // ork_wcache_evict must NOT free one it still references (would dangle the held ork_weight& -> the
    // ffn_swiglu_chain use-after-free, ASAN-confirmed 2026-07-25). Pinned keys are skipped by eviction.
    std::unordered_set<const void *> wcache_pin;
    size_t   wcache_bytes = 0;   // resident NPU bytes across wcache — streaming LRU budget tracker
    // ORK_FFN_CHAIN: PER-TENSOR (scalar-scale) packed weights for the round-trip-free SwiGLU chain (the
    // fused SiLU stage's scalar R can't carry per-channel scale). Keyed by weight host ptr. Separate from
    // wcache (which is per-channel). Plus a per-layer fused-SiLU LUT cache keyed by the gate weight ptr.
    struct ork_pt_weight { ork_w * w = nullptr; float scale = 0; };
    std::unordered_map<const void *, ork_pt_weight>    ptcache;
    std::unordered_map<const void *, std::vector<int16_t>> lutcache;   // gate ptr -> fused-SiLU LUT (1030)
    // ORK_FFN_CHAIN SmoothQuant: per-FFN-layer smoothing so per-tensor activation quant stays coherent (LLM
    // activation outliers are per-input-channel; migrate them into the weights via x'=x/s, W'=s*W — equivalent).
    // Computed once per layer (first sight, from that batch's per-channel |x| max + weight max), keyed by Wg ptr.
    struct ork_ffn_smooth {
        ork_w * wg = nullptr; ork_w * wu = nullptr; ork_w * wd = nullptr;
        float sg = 0, su = 0, sd = 0;                 // per-tensor int8 scales of the smoothed weights
        std::vector<float> s;                          // per-input-channel smoothing factor [K] (gate/up share x)
        std::vector<float> f;                          // per-Nff ewmul->glu_i8 factor (folds down-smoothing + s_glu)
        std::vector<int16_t> lut;                      // fused-SiLU LUT for (s_x'*sg -> s_silu)
        double s_x = 0, s_silu = 0, s_up = 0, s_glu = 0; // calibrated per-tensor activation/intermediate scales
        int up_mult = 0, up_shift = 0;
        bool ready = false;
        // ORK_FFN_GATE_F16: fp16 gate matmul + fused fp16 SiLU (precise silu, no int8 quant of the activation).
        // N-CHUNKED: one contiguous [K,Nff] fp16 buffer (e.g. 24MB) fails MEM_CREATE when the domain's 4GiB
        // IOVA is fragmented by the resident orkpack (only <=~12MB contiguous IOVA ranges survive). Split into
        // <=wg_f16_cn-column chunks, each a single-tile ork_w (ork_f16_mm_run_silu requires Sn==1 per weight).
        std::vector<ork_w *> wg_f16;                   // gate weight packed fp16 as -S*Wg, one ork_w per N-chunk (empty => not gate-f16 path)
        int wg_f16_cn = 0;                             // columns per chunk (last chunk may be shorter)
        // ORK_FFN_F16: ALL-fp16 FFN inner — up/down ALSO fp16 so NO int8 activation quant (fp32->fp16 cast
        // only) and NO int32->fp32 per-channel dequant. up is N-chunked like the gate (24MB single tile fails
        // MEM_CREATE under IOVA fragmentation); down packs as one weight (N=Kd=2048 => tiles <=~8MB, fit).
        std::vector<ork_w *> wu_f16;                   // up weight, raw fp16, N-chunked (plain ork_f16_mm_run per chunk)
        ork_w * wd_f16 = nullptr;                      // down weight, raw fp16, single ork_f16_mm_pack (K=Nff K-sliced)
        bool f16_all = false;                          // this layer prepped for the all-fp16 path
        std::vector<int16_t> lut_f16;                  // universal fp16 silu LUT (calibrated once per layer's gate range)
        double f16_out = 0;                            // dequant: silu(gate) = C_out * f16_out
        // ORK_FFN_SILU_CPU_GMAX: per-layer precision policy. High-|gate| layers lose the most to int8 fused
        // silu (fixed-range LUT); when this layer's gmax exceeds the threshold, use the EXACT path instead —
        // per-channel gate matmul on-NPU + fp32 silu on CPU (baseline quality, ~2.3x the fused-silu gate cost).
        bool silu_cpu = false;
        // ORK_FFN_SILU_I16: like silu_cpu (int8 gate matmul, stable integer datapath) but the SiLU is the
        // on-NPU int16 op (ork_i16_npu_silu, ~325x more accurate than the int8 fused LUT: 0.28 vs 92 err
        // @gmax132) instead of CPU fp32 — the coherent, integer-datapath replacement for the fragile fp16 gate.
        bool silu_i16 = false;
        // ORK_FFN_SILU_I16_FUSED: FUSED int16 — fc.wg int8-mm -> int16-out (set_i16_out, on-device, no host
        // int32->fp32->int16 round-trip, no separate per-channel gate matmul) feeding the int16 SiLU resident.
        // Scale-bridge: gate_i16 = acc*(gate16_mult>>gate16_shift) ≈ g/gate16_is (g=acc*s_x*sg, is=gmax/32000).
        int gate16_mult = 0, gate16_shift = 0;
        double gate16_is = 0;
        double gmax = 0;                               // this layer's calibrated max|gate| (the sensitivity signal)
        // ORK_FFN_F16_JIT: IOVA-headroom variant of the all-fp16 path. Instead of RESIDING fp16 gate/up/down
        // (2x IOVA -> only ~5 layers fit a 4GiB domain), keep each weight host-side as compact int8 +
        // per-channel bscale and inflate it into ONE shared fp16 scratch (ctx->f16_scratch, reused across all
        // JIT layers) right before each matmul. Resident IOVA = a handful of shared scratches, so the
        // fp16-path layer count is decoupled from the IOVA cap => gmax becomes a pure coherence dial. The
        // fp16 MAC then runs int8-precision weights x unquantized fp16 activations = emulated W8A16.
        bool f16_jit = false;                          // this layer prepped for the JIT-inflate all-fp16 path
        // ORK_FFN_F16_CPUSILU: gate = PLAIN fp16 matmul (raw Wg) + EXACT CPU silu, instead of the fused per-
        // tensor silu LUT (ork_f16_mm_run_silu). The fused-per-tensor LUT can't represent silu over a wide gate
        // range (blk.2 gmax~132) -> deterministic garbage (PPL 9072/10554). Plain matmul + fp32 CPU silu is
        // exact. Costs: no on-NPU silu fusion + no LUT-load submit (also removes a mode-switch surface).
        bool f16_cpusilu = false;
        struct jit_w { std::vector<int8_t> i8; std::vector<float> bs; int N; };  // host int8 [K*N] + bscale[N]
        std::vector<jit_w> jg, ju;                     // gate(-S*Wg) / up chunks (K=fc K, N=cw)
        jit_w jd;                                      // down (K=Nff, N=Kd)
    };
    std::unordered_map<const void *, ork_ffn_smooth>   ffncache;       // gate ptr -> smoothed layer state
    // (a) runtime-adaptive gmax: auto-captured per-FFN-layer gate range (name, gmax) as prep sees each
    // layer. The model-specific "tuning data" — reported at free, and the source (b) persists into the
    // orkpack. NOTE: the gmax gate cannot rescue the fused int8 silu (its int8 OUTPUT is coarse -> broadly
    // inaccurate at ALL gmax, PPL 55; that's why all-CPU-silu is the shipped default). gmax gating only
    // pays off for a FUTURE all-NPU selective path (int16-silu, int16 output) — currently blocked (#35).
    std::vector<std::pair<std::string, float>> gmax_profile;
    // (b) persisted gmax tuning: loaded at init from the <orkpack>.gmax sidecar (written at free). A model
    // whose profile is already on disk skips the first-pass recompute AND can set per-layer policy at LOAD.
    // Sidecar (not embedded in the orkpack blob) because the in-blob write needs the chain-convert path,
    // which currently heap-corrupts (deep bug, separate). No active consumer yet (shipped policy is
    // all-CPU-silu); it's the foundation for a future all-NPU selective int16-silu gate (#35).
    std::unordered_map<std::string, float> gmax_loaded;
    std::unordered_map<uint64_t, ork_w *> f16_scratch;                 // (K<<32|N) -> reusable fp16 scratch (ORK_FFN_F16_JIT)
    uint64_t wcache_tick  = 0;   // monotonic clock for LRU last_use
    // STREAM-POOL tier (ORK_STREAM_POOL=1): RAM-resident inflated-int8 cache w/ cheap map/unmap.
    ork_stream_pool * spool = nullptr;     // created at init when enabled (NULL => fall back to plain wcache)
    size_t   spool_ram_bytes = 0;          // RAM bytes held across all stream entries (RAM-LRU budget)
    long     spool_remaps = 0, spool_ram_evicts = 0, spool_iova_unmaps = 0;  // diagnostics
    // .orkpack persist (path derived from the model; ORK_ORKPACK_PATH overrides): 0 off, 1 read (mmap'd), 2 write
    int      persist_mode = 0;
    void *   persist_map = nullptr; size_t persist_map_sz = 0;
    std::unordered_map<std::string, orkpack_entry> persist_idx;             // read-mode index
    FILE *   persist_out = nullptr; std::string persist_tmp, persist_final; // write-mode
    std::vector<std::pair<std::string, orkpack_entry>> persist_built; uint64_t persist_off = 0;
    std::unordered_set<std::string> persist_dumped;   // names already written to .orkpack (skip re-dump on convert-decode re-pack)
    long persist_hits = 0, persist_misses = 0;   // weights loaded from .orkpack vs packed (diagnostic)
    // MoE expert weights are too numerous to keep ALL packed NPU-resident (the IOMMU exhausts ~2k).
    // Fixed pool PER SHAPE: a bounded set of slots allocated once, reused round-robin via repack-in-place
    // (NO alloc/free → no IOMMU fragmentation). Dense/attn weights stay in wcache (resident forever).
    std::unordered_map<int64_t, std::deque<ork_moe_slot>> moe_pools;  // shape (K<<32|N) -> slots
    std::unordered_map<int64_t, size_t>                   moe_rr;     // shape -> round-robin cursor
    std::unordered_map<const void *, ork_moe_slot *>      moe_loc;    // expert host ptr -> its slot
    // ---- M1b/M3 STATIC HOT-EXPERT PARTITION (decode) ----
    // The round-robin pool above is shared per-SHAPE across ALL 22 MoE layers, so 22*32 distinct experts
    // churn through it and the default 384-slot cap allocates >4 GiB of fresh DMA buffers -> the IOMMU
    // CREATE EFAULTs (the live-path soft-reset). The partition bounds residency PER LAYER-TENSOR instead:
    // pin only the top-K hottest experts of each `_exps` tensor resident on the NPU (LRU within the cap,
    // freeing IOVA on evict); route every other (cold) expert to a CPU GEMV. This keeps the resident set
    // = K * n_moe_tensors * per-proj bytes, comfortably < 4 GiB, and never wedges the NPU.
    // hot pool: layer-tensor host ptr -> {expert host ptr -> resident slot}. One pool per (layer,proj).
    struct ork_hot_slot { ork_w * w = nullptr; std::vector<float> bscale; const void * key = nullptr; uint64_t last_use = 0; };
    std::unordered_map<const void *, std::unordered_map<const void *, ork_hot_slot>> moe_hot; // tensorbase -> (expert ptr -> slot)
    uint64_t moe_hot_tick = 0;
    size_t   moe_hot_bytes = 0;     // resident NPU bytes pinned by the hot partition (budget tracker)
    size_t   moe_hot_peak  = 0;     // PEAK resident bytes (gate: must stay < 4 GiB)
    long     moe_hot_hits = 0, moe_cold_cpu = 0;   // runtime hit-rate: NPU-routed vs CPU-routed expert calls
    // reuse the quantized activation across consecutive matmuls that share the same src1 input
    // (Q/K/V off the normed hidden state; FFN gate/up off the same x) — skips redundant per-matmul
    // activation int8-quant. Holds for the data in ctx->ai/as while last_* matches.
    const void * last_src1 = nullptr; int last_M = 0, last_K = 0; int last_type = 0;
    // ORK_PROFILE=1: accumulate where time goes, report on free (split decode M=1 vs prefill M>1)
    double t_quant = 0, t_run = 0, t_deq = 0; long n_mm = 0; int profile = 0;
    double t_actq = 0; long n_actq = 0;   // LEVER3: pure activation-quant arithmetic (NEON absmax+quantize loop), split out of t_quant
    double t_run_dec = 0, t_run_pf = 0; long n_dec = 0, n_pf = 0, m_pf = 0;
    // STREAMING sub-breakdown (ORK_PROFILE): where weight-resolution time goes for the 7B streamed prefill.
    // These split the t_quant bucket's weight-side from the activation-quant. Microseconds.
    double s_resolve = 0;   // total ork_resolve_weight_i8 wall (hits+packs+loads)
    double s_dequant = 0;   // pack-miss: src0 (Q8_0/etc) -> f32 dequant
    double s_tile    = 0;   // pack-miss: f32 -> per-channel int8 quant + transpose into bi[]
    double s_pack    = 0;   // pack-miss: ork_i8_mm_pack (tile into IOVA dma-buf)
    double s_load    = 0;   // .orkpack hit: ork_i8_mm_load (Bf full-K rebuild)
    double s_install = 0;   // ork_spool_install: dump + inflate-into-RAM (add_i8) + free + map
    double s_remap   = 0;   // cheap MEM_CREATE re-map of an IOVA-evicted RAM-resident entry
    long   n_hit = 0, n_packmiss = 0, n_loadhit = 0, n_remap = 0;   // event counts
    // MoE chained-handler phase breakdown (ORK_PROFILE): where the 0.97 t/s goes.
    double moe_prequant = 0, moe_pack = 0, moe_gather = 0, moe_chain = 0, moe_scatter = 0; long moe_calls = 0;
    double moe_deq = 0, moe_quant = 0;   // pack/repack sub-split: Q4_K->f32 dequant vs f32->int8 quant+tile
    double moe_cold = 0; long moe_cold_calls = 0;   // cold-expert CPU GEMV (threaded ggml vec_dot) wall time
    long moe_pack_calls = 0;   // [VERIFY] number of first-touch live packs/loads (get_hot misses)
    long moe_chain_S_sum = 0, moe_fallback_calls = 0; double moe_fallback_t = 0;   // [VERIFY] S total + per-task fallback
    // ---- PATH (b): fusion-preserving + concurrent MoE split (ORK_MOE_PATHB) ----
    // A cached CPU backend used to compute the CPU-share of the MoE via ggml's REAL fused batched
    // MUL_MAT_ID kernel (NOT the per-expert vec_dot loop) on a compacted sub-graph. Created lazily.
    ggml_backend_t cpu_backend = nullptr;          // ggml_backend_cpu_init() (lazy)
    int      pathb_cpu_threads = 0;                // ORK_MOE_CPU_THREADS (default 4)
    // ---- ASYNC CROSS-STREAM (ORK_ASYNC / ggml_backend_ork_graph_compute_async) ----
    // A per-context worker thread runs the (blocking) graph_compute body so the launching stream can do CPU
    // work while this stream's NPU submits are in flight. At most ONE in-flight job per backend. The worker
    // holds the process-global g_npu_queue_mu (the RKNPU is single-stream) so cross-context NPU work serializes.
    std::thread async_thr;
    bool        async_inflight = false;
    enum ggml_status async_status = GGML_STATUS_SUCCESS;
    // Per-context fused-attention scratch (moved off the old process-global g_attnp / softmax statics) so
    // concurrent ork backends don't corrupt each other's attention buffers. Freed in ggml_backend_ork_free.
    // PER-IOMMU-DOMAIN fused-attention pools, keyed by domain id. One pool per domain that actually hosts
    // attention layers (2 on the 35B: dom 0 + dom 1), so each layer's attention runs co-domain with its own
    // resident int8 weights and the run never switches domains for attention. A pool is ~1-2 MB, so a
    // handful of them is noise. See attn_pool_ensure for why co-domain matters.
    std::unordered_map<int, struct ork_attn_pool> attnp = {};
    struct ork_attn_sm   attn_sm = {};
    // Resident-KV for the int8 DECODE attention path (ORK_ATTN_KV): pack K^T/V ONCE per (layer, kv-head)
    // then append one key/value per token (no per-call repack — the perf lever, vs the default repack path).
    // Keyed on k->data (the per-layer K-cache view base, stable across decode steps). Fixed per-head int8
    // scales chosen at the first pack. Freed in ggml_backend_ork_free / on a sequence reset (nkv shrink).
    struct ork_kv_layer {
        std::vector<ork_kv_resident *> kv;   // [Hkv] resident K^T/V bundles
        std::vector<float> ks, vs;            // [Hkv] fixed int8 scales (from first pack)
        int packed = 0;                       // keys packed so far (== last nkv)
        int Lmax = 0, Hkv = 0, DK = 0, DV = 0;
        ork_w *ones = nullptr;                // ORK_ATTN_FUSED: resident ones[Lmax,32] reduce weight (Σ = e·ones)
        int fused = -1;                       // -1 uninit; 0/1 = this layer's resident scales are per-head/global
    };
    std::unordered_map<const void *, ork_kv_layer> attn_kv;
    // Per-layer DECODE FFN calibration cache (ORK_FFN_DEC): route the SwiGLU inner through the fused orkd
    // chain (ork_mm_ffn_orkd — one submit) at decode. Per-tensor int8; the intermediate int8 scales (is/os/
    // us/gs) are calibrated ONCE per layer on the first real decode activation (representative), then reused;
    // the activation scale + gate/up requant are recomputed live per token. Keyed on the gate weight ptr.
    struct ork_ffn_dec { bool ready=false; float is=1,os=1,us=1,gs=1, wg_s=1,wu_s=1,wd_s=1; int K=0,Nff=0,Kd=0; };
    std::unordered_map<const void *, ork_ffn_dec> ffndec;
    // Repacked CPU-share weights: per src0->data, a repack-buffer-backed copy of the full experts tensor
    // (the SAME tiled layout the native fused MUL_MAT_ID uses) so the CPU share is a fair fight vs the
    // repacked baseline. Built once on first touch (ORK_MOE_PATHB_REPACK=1). ctx+buffer kept alive here.
    struct ork_repack_as { struct ggml_context * gctx = nullptr; ggml_backend_buffer_t buf = nullptr; struct ggml_tensor * t = nullptr; };
    std::unordered_map<const void *, ork_repack_as> pathb_repack;
    // PATH (b) profiling/diagnostics
    double pathb_npu_t = 0, pathb_cpu_t = 0, pathb_combine_t = 0, pathb_wall_t = 0; long pathb_calls = 0;
    long pathb_npu_experts = 0, pathb_cpu_experts = 0;
    // ---- EXPERIMENT: phase-aware backbone eviction (#1, ORK_MOE_PHASE_EVICT) ----
    // At DECODE (M==1) the dense backbone is bandwidth-bound and the NPU barely earns its IOVA, while the
    // ~2.8 GiB it pins starves the MoE hot-expert cache. supports_op DECLINES dense MUL_MAT at M==1 (CPU
    // takes it, cheap at M=1), and at the prefill->decode boundary we BULK-FREE the backbone wcache so the
    // freed IOVA goes to experts; the next prefill repopulates it. A clean bulk free (not incremental LRU)
    // avoids rk_iommu fragmentation. last_graph_decode tracks the previous graph's phase for the boundary.
    int  phase_evict = 0;        // ORK_MOE_PHASE_EVICT (cached at init)
    int  last_graph_decode = -1; // -1 unknown; 0 prefill (max M>1); 1 decode (max M==1)
    long backbone_evicts = 0; size_t backbone_evict_bytes = 0;  // diagnostics
    // ---- EXPERIMENT: routing-frequency profiler (ORK_MOE_PROFILE_FREQ=<file>) ----
    // Accumulate per-(_exps tensor, expert) selection counts so the mixed-orkpack tier map can rank
    // experts by routing frequency (hottest -> int8/NPU-resident, cold tail -> int4/CPU). Dumped on free.
    std::map<std::string, std::vector<long>> moe_freq;   // exps-tensor-name -> per-expert hit count
    // ---- MULTI-DOMAIN RESIDENCE (ORK_DOMAINS / ORK_DOMAIN_LAYERS) ----
    // The rk_iommu 32-bit IOVA cap (~4 GiB) is PER iommu_domain_id, so a model bigger than 4 GiB stays
    // FULLY resident (no streaming, no per-token map/unmap) by spreading its layers' weights across
    // domains. n_domains>1 enables it: each weight's domain = min(layer_idx / domain_layers, n_domains-1)
    // (non-blk tensors -> domain 0). Set the pack domain on ork-driver before each pack/load so the
    // resident tiles land there and the matmul submits against it. Per-domain resident totals are tracked
    // for the load-time report. Default n_domains=1 -> unchanged single-domain behavior.
    int    n_domains = 1;        // ORK_DOMAINS (default 1 = off)
    int    domain_layers = 0;    // ORK_DOMAIN_LAYERS (layers per domain; 0 = auto from ORK_DOMAINS+max layer)
    size_t domain_fill_cap = (size_t) 3000 * 1024 * 1024;  // per-domain fill ceiling (ork_domain_for); auto = EVEN
                                 // inflated_total/n_domains so domains fill uniformly with equal IOVA headroom,
                                 // instead of greedily to 3.0 GiB (which packs early domains tight + overflows last)
    size_t domain_bytes[64] = {0};   // resident NPU bytes placed in each domain (report). Sized to the 64-domain system ceiling (see dma_moeC).
    long   mem_create_runtime = 0;   // # of weight packs/loads AFTER the load phase (must stay ~0 = no churn)
    int    load_phase = 1;           // 1 during initial residence fill; cleared at first decode/steady state
    int    domain_cursor = 0;        // byte-balanced fill: current domain being filled (advances as domains near the IOVA cap)
    int    domain_last_layer = -1;   // last blk.N layer index seen by ork_weight_domain (advance domains only at layer boundaries)
    // FOOTPRINT-DERIVED RESIDENCE MODE (set once by the auto-sizer from the resident footprint vs a RAM
    // budget). RESIDENT (0): the whole footprint fits -> load once, NEVER evict, zero per-token churn
    // (librkllmrt / multi-domain model). STREAM (1): footprint > RAM budget -> stream by layer (co-resident
    // within a layer's chain, evict the previous layer at each boundary). This REPLACES per-call allow_evict
    // as the eviction POLICY (allow_evict stays only as the intra-submit co-residence safety; #55 migrates
    // the dense callers). Consumed by the MoE prefill path; residence_ram_budget is the capacity it compares to.
    int    residence_stream = 0;         // 0 = RESIDENT (fits), 1 = STREAM-by-layer (oversized)
    size_t residence_footprint = 0;      // auto-sizer's inflated resident-weight footprint (bytes)
    size_t residence_ram_budget = 0;     // capacity the footprint is compared against (bytes)
    // PER-DOMAIN FUSION (>4GiB): the fused per-tensor gate (fc.wg) is NOT a separate/extra weight — it REPLACES
    // the per-channel gate 1:1 (same K×N, same Bb+Bf, differs only in scale), so in fused mode the per-channel
    // gate is never packed and fc.wg takes its place in the SAME layer's domain via ork_weight_domain(). It is
    // packed as an IMPORT (like every other weight) so it doesn't fragment the domain's 32-bit IOVA with a
    // native-alloc outlier. No dedicated fc.wg domains, no extra volume, no per-layer domain-switch thrashing.
};
static ggml_backend_ork_context * g_ork_ctx = nullptr;
static bool g_ork_hybrid_loading = false;
// ---- MoE AUTO-PROFILE (no env knobs: the MODEL TYPE selects the quantization/placement scheme) ----
// Set the first time ork is asked about a GGML_OP_MUL_MAT_ID op (i.e. the model is a routed MoE); llama.cpp
// plans the graph at load, so this is known before any weight is packed or any op computed. It turns on the
// MEASURED-OPTIMAL MoE profile for this hardware (RK3588), which used to require three env vars:
//   * orkpack tier: expert/ffn tensors -> int4 (NF4 when the source is f16+), attn/dense stay int8
//   * experts computed on the CPU via the batched NF4/int4 NEON kernel (ork claims MUL_MAT_ID to do it)
//   * at DECODE (M==1) the dense/attn backbone is declined to the CPU (the NPU submit floor loses at M=1)
// Measured on qwen3.6-35B-A3B (q8_0-attn base): prefill 36.9 (1.59x native) / decode 6.59 t/s.
// Escape hatches (research only): ORK_MOE_AUTO=0 disables the profile; ORK_MOE_NPU forces experts to the NPU.
static bool g_ork_is_moe = false;
static inline bool ork_moe_auto() {
    static const bool off = (getenv("ORK_MOE_AUTO") && atoi(getenv("ORK_MOE_AUTO")) == 0);
    return !off && g_ork_is_moe;
}
// ---- Load-time product config (the two user-facing options; see ggml-ork.h) ----
static bool g_ork_cfg_set          = false;   // has ggml_backend_ork_set_load_config been called this process?
static bool g_ork_cfg_dflash       = false;   // enable the speculative block-diffusion drafter
static bool g_ork_cfg_silu_int8fused = false; // false = int16 coherent (unfused standalone NPU silu, default);
                                              // true = int8 fully-fused through-and-through (SiLU fused into gate matmul)
static inline double ork_now_us_e(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec*1e6 + t.tv_nsec*1e-3; }

// MULTI-DOMAIN RESIDENCE: place each weight in an IOMMU domain, filled SELF-CALIBRATING + sequential.
// ork is a compute-only accelerator: the model's weights live on CPU/mmap buffers (not ork buffers, so
// g_ork_weight_bytes stays ~0) and arrive one matmul at a time, so the resident total is NOT known up
// front, and the exact usable IOVA per domain isn't a fixed number (alignment/fragmentation + the full-K
// Bf rebuild inflate it). So instead of a magic per-domain byte cap, we fill the CURRENT domain until a
// pack actually fails (bcreate EFAULT = that domain's real IOVA limit), then the caller advances the
// cursor and retries the weight in the next domain (see ork_domain_advance). This uses the full usable
// window of each domain and needs neither the total nor a hardcoded cap. Returns the current domain
// (0 when multi-domain is off).
// Parse the transformer layer index from a weight name ("blk.<N>.<...>"), or -1 for a non-layer weight
// (token_embd / output / *_norm). Used for LAYER-ALIGNED domain assignment.
static int ork_layer_of(const char * name) {
    if (!name) return -1;
    const char * p = strstr(name, "blk.");
    if (!p) return -1;
    p += 4; if (*p < '0' || *p > '9') return -1;
    int l = 0; while (*p >= '0' && *p <= '9') l = l * 10 + (*p++ - '0');
    return l;
}
// Domain a weight resides in. LAYER-ALIGNED (ctx->domain_layers>0): a whole layer's weights -> ONE domain
// (domain = layer/domain_layers), so a per-layer fused FFN chain never crosses a domain boundary (enables
// per-domain fusion for >4GiB models). Deterministic by layer, so the lazily-packed fused gate lands in the
// SAME domain as its up/down (both keyed by the same layer). Non-layer NPU weights (e.g. lm_head) -> the LAST
// domain (it's under-filled by ceil rounding). When domain_layers==0 (byte-balanced auto sizing) the fill
// still advances ONLY at a layer boundary, so a layer stays co-domain — preserving ork_dispatch_i8's
// cross-core RR chains. `layer` = ork_layer_of(name), or -1 if non-layer/unknown.
static int ork_weight_domain(ggml_backend_ork_context * ctx, size_t bytes, int layer) {
    if (ctx->n_domains <= 1) return 0;
    if (ctx->domain_layers > 0) {
        int d = (layer >= 0) ? layer / ctx->domain_layers : ctx->n_domains - 1;  // non-layer -> last (spare) domain
        if (d >= ctx->n_domains) d = ctx->n_domains - 1;
        if (d < 0) d = 0;
        return d;
    }
    // Advance to the next domain BEFORE this one hits its IOVA EFAULT. Critical: a domain filled to its
    // ~4 GiB limit EFAULTs and soft-resets the NPU, which corrupts the IOMMU so the NEXT domain fails to
    // start (measured: fill domain 0 to EFAULT → domain 1 dies on its first tile). domain_probe shows the
    // real per-domain limit is ~4.16 GiB; cap at 3.0 GiB to leave margin for the full-K Bf inflation and a
    // one-weight overshoot — so no domain ever EFAULTs and the hand-off to the next domain is clean.
    const size_t cap = ctx->domain_fill_cap;   // EVEN target (footprint/n_domains), <= 3.0 GiB hard ceiling
    // Advance ONLY at a LAYER boundary (the layer index changed), never mid-layer, so a layer's whole
    // matmul set (Q/K/V, gate/up) stays co-domain. That (a) lets ork_dispatch_i8's independent QKV/gate-up
    // chains run round-robin across cores (run_stream is single-domain), and (b) keeps decode's per-token
    // domain swaps minimal. A layer's weights are tens of MB — far under the cap margin — so deferring the
    // advance to the next layer never risks an EFAULT. `layer` is the ork_layer_of() index passed in.
    const bool new_layer = (layer >= 0 && layer != ctx->domain_last_layer);
    int d = ctx->domain_cursor;
    if (d < ctx->n_domains - 1 && ctx->domain_bytes[d] + bytes > cap && new_layer) d = ++ctx->domain_cursor;
    if (layer >= 0) ctx->domain_last_layer = layer;
    return d;
}
// A pack/load just failed in the current domain (its IOVA window is full). Advance to the next domain if
// one remains; returns the new domain, or -1 when the last domain is also exhausted (a real OOM).
static int ork_domain_advance(ggml_backend_ork_context * ctx) {
    if (ctx->domain_cursor >= ctx->n_domains - 1) return -1;
    int d = ++ctx->domain_cursor;
    ork_npu_set_pack_domain(ctx->npu, d);
    fprintf(stderr, "[ork] domain %d full — advancing residence to domain %d\n", d - 1, d);
    return d;
}

// ORK_SLICE_ROUTE: free + forget a weight's sliced-doorbell twin (if any) whenever its ork_w is freed, so a
// reused ork_w pointer can never resolve to a stale twin. No-op unless slice_route packed a twin for this w.
static inline void ork_slice_ws_drop(ggml_backend_ork_context * ctx, ork_w * w) {
    if (!w) return;
    ctx->slice_forced.erase(w);   // drop the wedge-prone memo too (a reused ork_w ptr must not mis-skip run_i8)
    if (ctx->slice_ws.empty()) return;
    auto it = ctx->slice_ws.find(w);
    if (it != ctx->slice_ws.end()) { ork_mm_free_sliced(ctx->npu, it->second); ctx->slice_ws.erase(it); }
}

// Evict least-recently-used weights (reclaiming IOVA via ork_mm_free) until `need` more bytes fit under
// the budget. Only per-tile-owned weights (int8 / per-channel int4) actually return IOVA; the current
// op's weight is never in the cache yet, so it is never evicted.
static void ork_wcache_evict(ggml_backend_ork_context * ctx, size_t need) {
    // Convert mode (building .orkpack): keep ~0 resident — pack→dump→free each weight (evicted by the next
    // pack). This makes conversion fit ANY model size (≤1 weight in the 4 GiB window) and avoids thrash.
    const size_t budget = ctx->persist_mode == 2 ? 0 : ork_wcache_budget();
    while (ctx->wcache_bytes + need > budget && !ctx->wcache.empty()) {
        auto lru = ctx->wcache.end();
        for (auto it = ctx->wcache.begin(); it != ctx->wcache.end(); ++it)
            if (!ctx->wcache_pin.count(it->first) &&                    // never evict a weight the active op still references
                (lru == ctx->wcache.end() || it->second.last_use < lru->second.last_use)) lru = it;
        if (lru == ctx->wcache.end()) break;   // all remaining entries are pinned by the active op — can't free more
        ork_slice_ws_drop(ctx, lru->second.w);
        ork_mm_free(ctx->npu, lru->second.w);
        ctx->wcache_bytes -= lru->second.bytes;
        ctx->wcache.erase(lru);
    }
}

// MoE STREAM per-layer evict (task #54): reclaim IOVA from routed-expert entries (is_expert) ONLY, LRU-first,
// until resident expert bytes fall to `budget`. DENSE weights are never touched (they stay resident every
// layer). Called once per _exps MUL_MAT_ID BETWEEN grouped runs, so it never frees an expert an in-flight
// chain still references (eviction is between synchronous calls -> no UAF). Bounds the streamed working set
// (~a couple of layers of experts); prior layers age out as prefill advances -> the 35B's ~30 GB int8-inflate
// stays within ~`budget` resident instead of overflowing IOVA/RAM. (se/spool tier not evicted here; the
// STREAM path imports fresh, no spool.)
static void ork_wcache_evict_experts(ggml_backend_ork_context * ctx, size_t budget) {
    size_t eb = 0; for (auto & kv : ctx->wcache) if (kv.second.is_expert) eb += kv.second.bytes;
    while (eb > budget) {
        auto lru = ctx->wcache.end();
        for (auto it = ctx->wcache.begin(); it != ctx->wcache.end(); ++it)
            if (it->second.is_expert && !ctx->wcache_pin.count(it->first) &&
                (lru == ctx->wcache.end() || it->second.last_use < lru->second.last_use)) lru = it;
        if (lru == ctx->wcache.end()) break;   // remaining experts pinned by the active op
        eb                -= lru->second.bytes;
        ctx->wcache_bytes -= lru->second.bytes;
        ork_slice_ws_drop(ctx, lru->second.w);
        ork_mm_free(ctx->npu, lru->second.w);
        ctx->wcache.erase(lru);
    }
}

// ---- STREAM-POOL two-tier eviction ------------------------------------------------------------
// IOVA tier (evict-RARE): unmap the LRU *mapped* stream entries until `need` more IOVA bytes fit under
// the IOVA budget. The entry STAYS in RAM (next map is cheap, no re-inflate). Never unmaps a still-needed
// entry — the op's own entry is mapped AFTER this call, so it isn't a candidate here.
static void ork_spool_iova_evict(ggml_backend_ork_context * ctx, size_t need) {
    const size_t budget = ork_wcache_budget();
    while (ctx->wcache_bytes + need > budget) {
        ork_weight * lru = nullptr; auto lru_it = ctx->wcache.end();
        for (auto it = ctx->wcache.begin(); it != ctx->wcache.end(); ++it)
            if (it->second.se && ork_stream_entry_mapped(it->second.se) &&
                (!lru || it->second.last_use < lru->last_use)) { lru = &it->second; lru_it = it; }
        if (!lru) break;   // nothing mapped to evict
        ork_stream_pool_unmap(ctx->spool, lru->se);
        ctx->wcache_bytes -= lru->bytes; lru->bytes = 0; lru->w = nullptr;
        ctx->spool_iova_unmaps++; (void) lru_it;
    }
}
// Forcibly unmap the single LRU mapped entry (other than `keep`). Returns its freed bytes, or 0 if none.
// Used to recover from a PRIME/MEM_CREATE import failure: the rk_iommu IOVA window fragments, so the
// byte-budget can be met yet a fresh import still EFAULTs — drop more resident maps and retry.
static size_t ork_spool_unmap_one_lru(ggml_backend_ork_context * ctx, ork_stream_entry * keep) {
    ork_weight * lru = nullptr;
    for (auto it = ctx->wcache.begin(); it != ctx->wcache.end(); ++it)
        if (it->second.se && it->second.se != keep && ork_stream_entry_mapped(it->second.se) &&
            (!lru || it->second.last_use < lru->last_use)) lru = &it->second;
    if (!lru) return 0;
    size_t freed = lru->bytes;
    ork_stream_pool_unmap(ctx->spool, lru->se);
    ctx->wcache_bytes -= lru->bytes; lru->bytes = 0; lru->w = nullptr;
    ctx->spool_iova_unmaps++;
    return freed ? freed : 1;
}
// Map an entry, retrying after dropping more LRU maps on import failure (IOVA fragmentation). 0 ok / -1.
static int ork_spool_map_retry(ggml_backend_ork_context * ctx, ork_stream_entry * e) {
    if (ork_stream_pool_map(ctx->spool, e) == 0) return 0;
    while (ork_spool_unmap_one_lru(ctx, e))
        if (ork_stream_pool_map(ctx->spool, e) == 0) return 0;
    return -1;
}
// RAM tier: remove (free RAM) the LRU stream entries — and their wcache slot — until `need` more RAM
// bytes fit under the RAM budget. Removing also unmaps if mapped (reclaims IOVA). The current op's entry
// is created AFTER this call, so it is never a candidate. A re-touch of an evicted weight re-inflates.
static void ork_spool_ram_evict(ggml_backend_ork_context * ctx, size_t need) {
    const size_t budget = ork_stream_ram_budget();
    while (ctx->spool_ram_bytes + need > budget && !ctx->wcache.empty()) {
        auto lru = ctx->wcache.end();
        for (auto it = ctx->wcache.begin(); it != ctx->wcache.end(); ++it)
            if (it->second.se && (lru == ctx->wcache.end() || it->second.last_use < lru->second.last_use)) lru = it;
        if (lru == ctx->wcache.end()) break;
        if (ork_stream_entry_mapped(lru->second.se)) ctx->wcache_bytes -= lru->second.bytes;
        ork_stream_pool_remove(ctx->spool, lru->second.se);
        ctx->spool_ram_bytes -= lru->second.ram_bytes;
        ctx->wcache.erase(lru);
        ctx->spool_ram_evicts++;
    }
}

// Open the resolved .orkpack: if the file exists and validates, mmap it for READ (load weights by name); else
// open a <path>.tmp for WRITE (this run packs the model and dumps it, then finalize renames it in).
// Find the GGUF this process is serving, from /proc/self/cmdline: prefer the argument after -m/--model
// (or --model=<p>), else the first argument ending in .gguf (covers ork_bench's positional model arg).
// cmdline is populated before main, so this works at backend-init time (the model is not mmap'd yet).
static std::string ork_find_model_path() {
    FILE * f = fopen("/proc/self/cmdline", "rb");
    if (!f) return std::string();
    std::vector<char> buf; char c; while (fread(&c, 1, 1, f) == 1) buf.push_back(c);
    fclose(f);
    std::vector<std::string> av; std::string cur;
    for (char ch : buf) { if (ch == '\0') { if (!cur.empty()) av.push_back(cur); cur.clear(); } else cur += ch; }
    if (!cur.empty()) av.push_back(cur);
    auto is_gguf = [](const std::string & s) {
        return s.size() > 5 && s.compare(s.size()-5, 5, ".gguf") == 0; };
    for (size_t i = 0; i < av.size(); i++) {                      // explicit -m / --model wins
        if ((av[i] == "-m" || av[i] == "--model") && i+1 < av.size() && is_gguf(av[i+1])) return av[i+1];
        if (av[i].rfind("--model=", 0) == 0 && is_gguf(av[i].substr(8)))                  return av[i].substr(8);
    }
    for (const auto & a : av) if (is_gguf(a)) return a;            // positional (ork_bench)
    return std::string();
}
// DEFAULT orkpack path: <model-dir>/<model-basename>.orkpack, derived from the loaded GGUF. No env needed.
// ORK_ORKPACK_PATH overrides it (e.g. to share ONE pack built from the f16 source across several GGUFs —
// sharing one pack across GGUFs during development). DEVELOPMENT OVERRIDE ONLY — the derived path is the
// supported path; a MoE model's pack scheme is fixed by the model type (NF4 experts + int8 attn/dense).
static std::string ork_default_orkpack_path() {
    std::string m = ork_find_model_path();
    if (m.empty()) return std::string();
    const size_t dot = m.rfind(".gguf");
    return m.substr(0, dot) + ".orkpack";
}
static void ork_persist_init(ggml_backend_ork_context * ctx) {
    // orkd: the .orkpack is a FIRST-CLASS citizen — it always loads (no gate). READ imports the pre-tiled
    // bytes into the CLIENT's own dma-buf and hands the fd to the daemon (ORKD_IMPORT / ork_i8_mm_import),
    // which maps it into the client's IOMMU domain — the client manages its own IOVA, the daemon never tiles
    // or owns weights. WRITE mode (packing a fresh .orkpack) still needs a resident NPU tile to dump, which is
    // fd-local, so under orkd we only support READ; if the file is absent, fall back to the live-pack path
    // (write mode is a one-time offline step — generate the .orkpack in a direct-NPU run, then run under orkd).
    // ORK_PERSIST is REMOVED — it made the pack path a required runtime knob. The pack path is now DERIVED
    // from the loaded model. Fail loudly so stale scripts get fixed rather than silently losing the pack.
    if (getenv("ORK_PERSIST")) {
        GGML_ABORT("ork: ORK_PERSIST is no longer supported. The .orkpack path is DERIVED from the loaded "
                   "model (<model-dir>/<model-basename>.orkpack) and needs no configuration — just run with "
                   "-m <model.gguf>. ORK_ORKPACK_PATH=<file.orkpack> exists ONLY as a development override "
                   "(e.g. to point several GGUFs at one pack built from an f16 source). Unset ORK_PERSIST.\n");
    }
    // Path resolution (no env required): ORK_ORKPACK_PATH (development override only) >
    // <model>.orkpack derived from the loaded GGUF. An absent derived pack is BUILT this run (one-time).
    const char * env_p = getenv("ORK_ORKPACK_PATH");
    std::string derived;
    if (!env_p || !*env_p) {
        derived = ork_default_orkpack_path();
        if (derived.empty()) {
            fprintf(stderr, "[ork] no .gguf found on the command line — cannot derive the orkpack path; "
                            "set ORK_ORKPACK_PATH=<file.orkpack> to enable the packed-weight path.\n");
            return;
        }
        fprintf(stderr, "[ork] orkpack (derived from model): %s%s\n", derived.c_str(),
                access(derived.c_str(), R_OK) == 0 ? "" : " (absent -> building this run)");
    }
    const char * p = (env_p && *env_p) ? env_p : derived.c_str();
    if (!p || !*p) return;
    ctx->persist_final = p;    // resolved pack path, valid in BOTH read and write mode (sidecars key off it)
    bool stale = false;   // present-but-incompatible pack seen -> regenerate + delete the old sidecar
    int fd = open(p, O_RDONLY);
    if (fd >= 0) {
        off_t sz = lseek(fd, 0, SEEK_END);
        if (sz > (off_t) sizeof(orkpack_footer)) {
            // AGGRESSIVE LOAD (default): pull the whole orkpack off disk NOW, sequentially — MAP_POPULATE
            // prefaults every page at mmap time (one big sequential read, ~disk-bandwidth-bound) so the
            // per-weight domain fills below hit warm page-cache instead of demand-paging random faults
            // interleaved with the first forward pass (the slow load). fadvise SEQUENTIAL primes readahead.
            // Opt back into lazy demand-paging with ORK_PERSIST_MMAP_LAZY=1 (the old mmap-view behavior).
            int mflags = MAP_PRIVATE;
#ifdef MAP_POPULATE
            const bool lazy_mmap = getenv("ORK_PERSIST_MMAP_LAZY") != nullptr;
            if (!lazy_mmap) mflags |= MAP_POPULATE;
#endif
            posix_fadvise(fd, 0, sz, POSIX_FADV_SEQUENTIAL);
            void * m = mmap(nullptr, sz, PROT_READ, mflags, fd, 0);
            if (m != MAP_FAILED) {
                posix_fadvise(fd, 0, sz, POSIX_FADV_WILLNEED);   // kick readahead for the whole file (belt-and-suspenders w/ MAP_POPULATE)
                orkpack_footer f; memcpy(&f, (char *) m + sz - sizeof f, sizeof f);
                // Reject (=> regenerate below) if the ork-driver pack format is incompatible: an older
                // footer schema (< v3, no ork_fmt) or a different pack-compat token (a tiling/quant
                // change bumps ork-driver's MAJOR version). Same-(K,N) blobs from an incompatible major
                // are the SAME size, so this token is the only thing that catches them.
                bool magic_ok = memcmp(f.magic, ORKPACK_MAGIC, 8) == 0;
                if (magic_ok && f.version != ORKPACK_VERSION) {        // older footer schema (pre-v3, no token)
                    fprintf(stderr, "[ORK PERSIST] %s predates the pack-compat token (footer < v%u) — regenerating\n", p, ORKPACK_VERSION);
                    stale = true;
                } else if (magic_ok && f.ork_fmt != ork_pack_format_version()) {  // v3 file, but tiling/quant major differs
                    fprintf(stderr, "[ORK PERSIST] %s is stale (pack-compat token %u != this build's %u) — regenerating\n",
                            p, f.ork_fmt, ork_pack_format_version());
                    stale = true;
                } else if (magic_ok && !ork_sig_compatible(f.quant_sig)) {        // right format, genuinely unusable config
                    fprintf(stderr, "[ORK PERSIST] %s was built for an incompatible config (quant_sig %u vs this run's %u) — regenerating\n",
                            p, f.quant_sig, ork_build_sig());
                    stale = true;
                }
                if (memcmp(f.magic, ORKPACK_MAGIC, 8) == 0 && f.version == ORKPACK_VERSION &&
                    f.ork_fmt == ork_pack_format_version() && ork_sig_compatible(f.quant_sig) && f.index_off < (uint64_t) sz) {
                    const char * idx = (const char *) m + f.index_off;
                    for (uint32_t i = 0; i < f.n_entries; i++) {
                        uint32_t nl; memcpy(&nl, idx, 4); idx += 4;
                        std::string name(idx, nl); idx += nl;
                        orkpack_entry e; memcpy(&e, idx, sizeof e); idx += sizeof e;
                        ctx->persist_idx.emplace(std::move(name), e);
                    }
                    // ADOPT the pack's tier. ork_persist_init runs BEFORE ctx->qbits is set, so this is the
                    // value the ctx init picks up when ORK_QUANT is unset: loading an int4 pack selects int4.
                    ctx->persist_qbits = ork_sig_qbits(f.quant_sig);
                    ctx->persist_map = m; ctx->persist_map_sz = sz; ctx->persist_mode = 1; close(fd);
                    if (getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK PERSIST] read %s (%zu weights) — loading from disk, no re-conversion\n", p, ctx->persist_idx.size());
                    // Not VERBOSE-gated: a pack silently switching the run's weight tier is the one adoption
                    // a reader must not have to guess at. (Tier only — the 4-bit COMPUTE path stays opt-in.)
                    if (ctx->persist_qbits == 4)
                        fprintf(stderr, "[ORK PERSIST] %s is an int4 pack — selecting the int4 tier (no ORK_QUANT needed)\n", p);
                    return;
                }
                munmap(m, sz);
            }
        }
        close(fd);
    }
    // WRITE mode (build the .orkpack) works under orkd too: the tiling is done PURE-CPU (ork_i8_w_dump_cpu for
    // Bb + ork_i8_w_dump_bf_cpu for Bf) from the live-converted raw int8 — no resident NPU tile, no daemon. So a
    // missing/stale .orkpack self-heals: this run rebuilds it (one-time), future runs READ + import it.
    ctx->persist_final = p; ctx->persist_tmp = std::string(p) + ".tmp";
    // Stale pack: the fresh one is written to <p>.tmp and atomically rename()'d over the old <p> at finalize
    // (create-new-then-replace-old, crash-safe). The <p>.gmax sidecar is NOT covered by that rename, so delete
    // the stale one now — it belongs to the old pack and will be rewritten at free.
    if (stale) { std::string sc = std::string(p) + ".gmax"; unlink(sc.c_str()); }
    ctx->persist_out = fopen(ctx->persist_tmp.c_str(), "wb");
    if (ctx->persist_out) { ctx->persist_mode = 2;
        // Unconditional (not ORK_VERBOSE-gated): building the pack packs weights INLINE this run, so this run's
        // speed is unrepresentative. Applies to every frontend that hits a missing/stale pack (llama-cli/server/
        // oRKLLM build it at load-time warmup; ork_bench builds it in a dedicated untimed pass). Warn + advise rerun.
        fprintf(stderr,
            "[ORK PERSIST] orkpack absent/stale -> BUILDING %s this run (one-time).\n"
            "              WARNING: weights are packed inline, so this run's performance/benchmark timing is SKEWED\n"
            "              and not representative. Re-run once the pack is built for true steady-state numbers.\n",
            ctx->persist_final.c_str()); }
}

// Public helper (see ggml-ork.h): does a usable .orkpack exist at `path` for this build? Mirrors the READ-mode
// validation in ork_persist_init (magic + footer schema + ork-driver pack-compat token) without mapping the whole
// file — just the footer. Lets a tool run a one-time build pass before timing when this returns false.
extern "C" bool ggml_backend_ork_orkpack_valid(const char * path) {
    if (!path || !*path) return false;
    int fd = open(path, O_RDONLY);
    if (fd < 0) return false;
    bool ok = false;
    off_t sz = lseek(fd, 0, SEEK_END);
    if (sz > (off_t) sizeof(orkpack_footer) && lseek(fd, sz - (off_t) sizeof(orkpack_footer), SEEK_SET) >= 0) {
        orkpack_footer f;
        if (read(fd, &f, sizeof f) == (ssize_t) sizeof f)
            ok = memcmp(f.magic, ORKPACK_MAGIC, 8) == 0 && f.version == ORKPACK_VERSION &&
                 f.ork_fmt == ork_pack_format_version() && ork_sig_compatible(f.quant_sig) && f.index_off < (uint64_t) sz;
    }
    close(fd);
    return ok;
}

// Effective bits-per-weight for a source ggml_type (mirrors tools/gguf_tier_map.c's table). <0 = unknown.
// This is the "allocation oracle" inline: a tensor's SOURCE quant precision is the importance signal we
// preserve — a k-quant/UD GGUF already spent more bits on the tensors that matter (output, bumped attn_v /
// ffn_down) and fewer on the bulk, so mirroring it onto our two storage tiers reproduces that allocation.
static double ork_src_type_bits(enum ggml_type t) {
    switch (t) {
    case GGML_TYPE_F32:     return 32.0;
    case GGML_TYPE_F16:     return 16.0;
    case GGML_TYPE_BF16:    return 16.0;
    case GGML_TYPE_Q4_0:    return 4.5;
    case GGML_TYPE_Q4_1:    return 5.0;
    case GGML_TYPE_Q5_0:    return 5.5;
    case GGML_TYPE_Q5_1:    return 6.0;
    case GGML_TYPE_Q8_0:    return 8.5;
    case GGML_TYPE_Q8_1:    return 9.0;
    case GGML_TYPE_Q2_K:    return 2.5625;
    case GGML_TYPE_Q3_K:    return 3.4375;
    case GGML_TYPE_Q4_K:    return 4.5;
    case GGML_TYPE_Q5_K:    return 5.5;
    case GGML_TYPE_Q6_K:    return 6.5625;
    case GGML_TYPE_Q8_K:    return 8.0;
    case GGML_TYPE_IQ1_S:   return 1.5625;
    case GGML_TYPE_IQ1_M:   return 1.75;
    case GGML_TYPE_IQ2_XXS: return 2.0625;
    case GGML_TYPE_IQ2_XS:  return 2.3125;
    case GGML_TYPE_IQ2_S:   return 2.5;
    case GGML_TYPE_IQ3_XXS: return 3.0625;
    case GGML_TYPE_IQ3_S:   return 3.4375;
    case GGML_TYPE_IQ4_NL:  return 4.5;
    case GGML_TYPE_IQ4_XS:  return 4.25;
    default:                return -1.0;   // unknown → conservative high-bit (int8)
    }
}

// ORK_MIXED_DISPATCH: per-tensor W4A4/W8A8 dispatch driven by the GGUF's OWN mixed quantization. A
// k-quant/UD GGUF (e.g. Q4_K_M) already spent MORE bits on the PPL-critical tensors (output/embeddings,
// bumped attn_v / ffn_down → Q6_K/Q5_K/Q8_0) and 4 bits on the robust bulk (Q4_K). So the source type IS
// the sensitivity signal, per tensor, for free: run the =4-bit tensors on the native W4A4 path (int4
// resident, ~2× decode bandwidth) and the >4-bit tensors on the coherent W8A8 path — the direct analog of
// the int8 ORK_FFN_SILU_CPU_GMAX per-layer precision escalation, mirrored onto the int4↔int8 axis. This is
// what protects PPL where a uniform Hadamard rotation alone did not: Hadamard tames the 4-bit bulk, and the
// few tensors it can't (the quantizer already flagged them >4-bit) fall to W8A8. Opt-in — the default
// (sub-5-bit → CPU) is unchanged until this is board-validated. Returns 1 only when the env is truthy.
static int ork_mixed_dispatch_on(void) {
    static int v = -1;
    if (v < 0) v = env_enabled("ORK_MIXED_DISPATCH") ? 1 : 0;
    return v;
}

// ---- ORK_ORKPACK_TIERMAP: external name<TAB>tier allocation map (Phase 4 STEP B) ------------
// Lets a clean fp16 source GGUF inherit a *different* GGUF's int4/int8 allocation by name (e.g. a
// Q4_K_M's "which tensors are int4"), so we quantize VALUES from fp16 but keep Q4_K_M's ALLOCATION.
// Format: one line per tensor, `name<TAB>tier`, tier in {int4,int8,4,8}. Lines starting with '#' and
// blanks are ignored. Returns: 4 / 8 if the name is in the map, -1 if no map or name absent.
static int ork_tiermap_lookup(const char * name) {
    static int loaded = 0;
    static std::map<std::string,int> map;   // empty when no ORK_ORKPACK_TIERMAP
    if (!loaded) {
        loaded = 1;
        const char * p = getenv("ORK_ORKPACK_TIERMAP");
        if (p && *p) {
            FILE * f = fopen(p, "r");
            if (!f) { fprintf(stderr, "[ORK TIERMAP] cannot open %s — ignoring\n", p); }
            else {
                char line[1024];
                while (fgets(line, sizeof line, f)) {
                    if (line[0] == '#' || line[0] == '\n' || line[0] == '\0') continue;
                    char * tab = strchr(line, '\t');
                    if (!tab) continue;
                    *tab = '\0';
                    const char * tv = tab + 1;
                    int tier = (strstr(tv, "int4") || tv[0] == '4') ? 4 :
                               (strstr(tv, "int8") || tv[0] == '8') ? 8 : 0;
                    if (tier) map[std::string(line)] = tier;
                }
                fclose(f);
                fprintf(stderr, "[ORK TIERMAP] loaded %zu entries from %s\n", map.size(), p);
            }
        }
    }
    if (!name) return -1;
    auto it = map.find(std::string(name));
    return it == map.end() ? -1 : it->second;
}

// ---- ORK_IMATRIX: GGUF importance matrix (llama.cpp in_sum2/counts form) ----------------------
// Returns a pointer to the per-INPUT-channel importance vector for `name` (length == K, the matmul
// contraction dim) or nullptr if no imatrix / tensor absent / length mismatch. Importance[k] =
// in_sum2[k] / counts (the mean squared activation of input channel k). Orientation: the imatrix
// stores `<name>.in_sum2` with ne[0] == the weight's input dim == the K passed to the packer — so the
// vector aligns directly with ork_i4a8_mm_pack_im's per-input-channel importance contract.
static const float * ork_imatrix_lookup(const char * name, int K) {
    static int loaded = 0;
    static struct gguf_context * gg = nullptr;
    static struct ggml_context * meta = nullptr;
    static std::map<std::string, std::vector<float>> cache;   // name -> importance[K]
    if (!loaded) {
        loaded = 1;
        const char * p = getenv("ORK_IMATRIX");
        if (p && *p) {
            struct gguf_init_params ip = { /*no_alloc=*/false, /*ctx=*/&meta };
            gg = gguf_init_from_file(p, ip);
            if (!gg) fprintf(stderr, "[ORK IMATRIX] cannot open/parse %s — ignoring\n", p);
            else     fprintf(stderr, "[ORK IMATRIX] loaded %s (%lld tensors)\n", p,
                             (long long) gguf_get_n_tensors(gg));
        }
    }
    if (!gg || !name) return nullptr;
    auto it = cache.find(std::string(name));
    if (it != cache.end()) return it->second.empty() ? nullptr : it->second.data();

    std::vector<float> & out = cache[std::string(name)];   // inserts empty (negative cache by default)
    std::string s2 = std::string(name) + ".in_sum2";
    std::string sc = std::string(name) + ".counts";
    struct ggml_tensor * t2 = ggml_get_tensor(meta, s2.c_str());
    struct ggml_tensor * tc = ggml_get_tensor(meta, sc.c_str());
    if (!t2 || t2->type != GGML_TYPE_F32) return nullptr;
    if ((int) t2->ne[0] != K) {
        fprintf(stderr, "[ORK IMATRIX] %s in_sum2 len %lld != K %d — orientation mismatch, skipping\n",
                name, (long long) t2->ne[0], K);
        return nullptr;     // never pass a wrong-length vector to the packer
    }
    const float * in_sum2 = (const float *) t2->data;
    float cnt = (tc && tc->type == GGML_TYPE_F32 && tc->data) ? *(const float *) tc->data : 1.0f;
    if (cnt <= 0.0f) cnt = 1.0f;
    out.resize(K);
    for (int k = 0; k < K; k++) out[k] = in_sum2[k] / cnt;   // mean squared activation per input channel
    return out.data();
}

// Decide the on-disk tier for one tensor when WRITING a mixed .orkpack (Phase 2.2 / 4.3). Two layers:
//
//   (A) SOURCE-TYPE policy (Phase 4.3, default ON): map the tensor's source ggml_type → effective bits;
//       bits >= 5 → int8 (F32/F16/BF16/Q8_*/Q6_K/Q5_K/Q5_*/Q4_1), bits < 5 → int4 (Q4_*/Q3_K/Q2_K/IQ*).
//       This makes the mixed .orkpack MIRROR the source GGUF's own allocation (a Q4_K_M's Q4_K bulk →
//       int4, its bumped Q6_K / Q8_0 output+embeddings → int8) — identical rule to tools/gguf_tier_map.c.
//       For an all-high-bit source (Q8_0, F16, unknown/<0) every tensor stays int8 → NO regression vs v1.
//       Disable with ORK_ORKPACK_TIER_FROM_SRC=0 (then only the explicit env overrides below apply).
//
//   (B) explicit env OVERRIDES (always win over the source-type verdict):
//       ORK_ORKPACK_I4_ABOVE_MB=<mb>  force int4 on any tensor whose int8 blob (~K*N bytes) exceeds <mb>
//       ORK_ORKPACK_I4_FFN=1          force int4 on the FFN/expert tensors, int8 the rest
//
// int4 needs K%32==0 && N%32==0; tensors that don't satisfy it stay int8 regardless. Returns 4 or 8.
static int ork_orkpack_tier(const char * name, int K, int N, enum ggml_type src_type) {
    static int init = 0, i4_ffn = 0, from_src = 1, q4_force = 0; static long i4_above_bytes = -1;
    if (!init) {
        init = 1;
        const char * a = getenv("ORK_ORKPACK_I4_ABOVE_MB");
        if (a && *a) i4_above_bytes = atoll(a) * 1024 * 1024;
        i4_ffn = getenv("ORK_ORKPACK_I4_FFN") ? 1 : 0;
        const char * fs = getenv("ORK_ORKPACK_TIER_FROM_SRC");   // default ON; "0" disables
        if (fs && fs[0] == '0' && fs[1] == '\0') from_src = 0;
        const char * q = getenv("ORK_QUANT");                    // ORK_QUANT=4: force compact i4a8 STORAGE for every
        if (q && q[0] == '4') q4_force = 1;                       // eligible tensor (compute stays W8A8-inflate on the NPU)
    }
    if ((K % 32) != 0 || (N % 32) != 0) return 8;          // int4 shape constraint → int8 regardless
    if (q4_force) return 4;                                // ORK_QUANT=4 forces int4 storage regardless of source type

    // (A0) external tier map (ORK_ORKPACK_TIERMAP) — wins over source-type so an fp16 source can
    // inherit a Q4_K_M's allocation by name. A mapped int4 still respects the shape constraint above.
    int tm = ork_tiermap_lookup(name);
    if (tm == 4 || tm == 8) return tm;

    bool want_i4 = false;
    // (A) source-type-driven default. SUPPRESSED under ORK_MIXED_DISPATCH: there the 4-bit tier COMPUTES
    // W8A8, so store int8 by default to preserve quality — int4 STORAGE re-quantizes to ork's crude
    // symmetric int4 (absmax/7), measured +36% PPL on wikitext (12.18 vs 9.23 for int8 storage, CPU 8.96).
    // int4 storage stays available via the explicit overrides below (the RAM-constrained >4GB case), ideally
    // with ORK_IMATRIX / NF4 to soften the loss.
    if (from_src && !ork_mixed_dispatch_on()) {
        double bits = ork_src_type_bits(src_type);
        if (bits >= 0.0 && bits < 5.0) want_i4 = true;     // low-bit source → int4 tier; unknown/high-bit → int8
    }
    // (B) explicit env overrides (force int4)
    if (i4_above_bytes >= 0 && (long) K * N >= i4_above_bytes) want_i4 = true;
    if ((i4_ffn || ork_moe_auto()) && name && (strstr(name, "ffn_") || strstr(name, "exps") ||
                           strstr(name, "expert") || strstr(name, "shexp"))) want_i4 = true;
    return want_i4 ? 4 : 8;
}

// Synthetic per-expert persist key: a routed MoE expert is one slice `e` of a 3D `_exps` tensor, so it
// has no ggml name of its own. We persist/load each slice under "<exps-tensor-name>#<e>" (e.g.
// "blk.13.ffn_gate_exps.weight#7"). Format is unambiguous: '#' never appears in a ggml tensor name.
static inline std::string ork_expert_key(const char * exps_name, int e) {
    return std::string(exps_name) + "#" + std::to_string(e);
}

// Append a freshly-packed weight to the write file and record its index entry, choosing the int8 or int4
// tier per ork_orkpack_tier(). int8: dump the already-packed tiled bytes (ow.w) + bscale[N]. int4: pack a
// temporary int4-W4A8 weight from the f32 plane (n-major [N][K], as ggml's to_float produced) and dump its
// self-describing 'O4N1' blob (carries K,N,quant_kind,bscale internally → no separate bscale trailer).
// ORK_ORKPACK_CPU: force the .orkpack tiling entirely onto the CPU (ork_i8_w_dump_cpu) — no bcreate/NPU
// pack at all, so the big cores (NEON dequant/quant + tile) do all the work and the NPU stays free.
// Turns the whole persist route CPU-only regardless of NPU idle/busy. int4-tier weights fall back to the
// int8 CPU dump (the compact int4 form needs the NPU packer). Off = the idle-gated hybrid.
static bool ork_orkpack_cpu_only() { static const int v = getenv("ORK_ORKPACK_CPU") != nullptr; return v; }

// bi_i8 (optional): the raw int8 weights [K*N, k-major]. When the int8 tier is chosen and no NPU-packed
// ow.w is supplied (the hybrid CPU route — NPU busy, or persistence-only weights that never went
// resident), the blob is tiled on the CPU straight from bi_i8 (ork_i8_w_dump_cpu, byte-identical to the
// NPU pack+dump) — no bcreate/IOVA. The int4 tier's NPU pack (ork_i4a8_mm_pack) is likewise gated on the
// NPU being idle; if it's busy the weight falls through to the int8 CPU dump (correct, just less compact).
static void ork_persist_write(ggml_backend_ork_context * ctx, const char * name, int K, int N,
                              const ork_weight & ow, const float * f32_plane, enum ggml_type src_type,
                              const int8_t * bi_i8 = nullptr) {
    if (ctx->persist_mode != 2 || !ctx->persist_out) return;
    if (!ctx->persist_dumped.insert(name).second) return;   // already dumped — a convert-decode re-pack, don't duplicate
    // Quality NOTE (once per build): quantizing an ALREADY-QUANTIZED source compounds error, and the NF4
    // codebook can't recover it (it fits the ORIGINAL weight distribution). We route the codebook by the SOURCE,
    // not an env flag: unquantized source (F16/F32/BF16, bits >= 16) -> NF4; quantized source -> uniform + this
    // warning. So the fix is a better SOURCE, not a knob.
    {
        static int warned_qsrc = 0;
        double sbits = ork_src_type_bits(src_type);
        if (!warned_qsrc && sbits > 0.0 && sbits < 16.0) {
            warned_qsrc = 1;
            fprintf(stderr,
                "[ORK PERSIST] NOTE: building this orkpack from an ALREADY-QUANTIZED model (%s, ~%.1f-bit). Re-quantizing\n"
                "              quantized weights compounds error, and the NF4 codebook only helps from full-precision\n"
                "              weights. For the best pack (especially int4), regenerate it from the model's UNQUANTIZED\n"
                "              weights (an F16/F32/BF16 GGUF) — the NF4 codebook is then applied automatically.\n",
                ggml_type_name(src_type), sbits);
        }
    }

    int tier = f32_plane ? ork_orkpack_tier(name, K, N, src_type) : 8;   // no f32 plane available → int8 only
    if (getenv("ORK_VERBOSE"))
        fprintf(stderr, "[ORK PERSIST] tier %s K=%d N=%d src=%s -> int%d\n",
                name, K, N, ggml_type_name(src_type), tier);
    if (tier == 4 && f32_plane) {   // int4: PURE-CPU pack straight to the compact i4a8 blob. No NPU packer, no
                                    // ork_npu_busy() gate — persist_write runs mid-forward while the NPU is busy with
                                    // the matmul, so the old NPU-packer path (gated !ork_npu_busy) ALWAYS skipped and
                                    // packed ZERO int4 weights. ork_i4a8_pack_cpu_blob is bit-identical to
                                    // ork_i4a8_mm_pack_im + ork_i4a8_w_dump (same on-disk blob), so the read path is unchanged.
        const float * im = ork_imatrix_lookup(name, K);   // per-input-channel importance, length K (or NULL)
        if (im && getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK PERSIST] imatrix %s (K=%d)\n", name, K);
        // Codebook routed by SOURCE (not an env flag): full-precision weights (F16/F32/BF16, >=16 bit) → NF4
        // non-uniform codebook (fits the original distribution); an already-quantized source → uniform (NF4 can't
        // recover compounded quant error — the warning above tells the user to rebuild from unquantized weights).
        // MoE AUTO-PROFILE: a routed-MoE model gets ONE pack scheme decided by the MODEL TYPE — always the
        // NF4 codebook for its int4 tier, regardless of the GGUF's own precision. (Otherwise a pack built from
        // an already-quantized GGUF would fall back to uniform int4, which measures 1.4x SLOWER in the M=1
        // kernel, 90 vs 65 us/expert, and is no more accurate.) Non-MoE keeps the source-routed choice.
        int nf4 = (ork_src_type_bits(src_type) >= 16.0 || ork_moe_auto()) ? 1 : 0;
        size_t tb = ork_i4a8_pack_cpu_blob(ctx->npu, K, N, f32_plane, im, nf4, nullptr, 0);
        if (tb) {
            std::vector<char> tmp(tb);
            ork_i4a8_pack_cpu_blob(ctx->npu, K, N, f32_plane, im, nf4, tmp.data(), tb);
            orkpack_entry e{}; e.K = K; e.N = N; e.dtype = ORKPACK_DT_I4; e.bscale_n = 0;
            e.blob_off = ctx->persist_off; e.blob_size = tb; e.bscale_off = 0;   /* e.bf_size = 0 (value-init) */
            fwrite(tmp.data(), 1, tb, ctx->persist_out); ctx->persist_off += tb;
            ctx->persist_built.emplace_back(std::string(name), e);
            if (getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK PERSIST] int4(cpu) %s K=%d N=%d (%zu B)\n", name, K, N, tb);
            return;
        }
        // CPU int4 pack failed → fall through to int8 (never persist a broken entry)
    }
    // int8 tier: tile on the CPU from the raw int8 weights when forced CPU-only or when there's no
    // NPU-packed weight (hybrid CPU route — no NPU/IOVA); otherwise dump the NPU-packed tiles. Both
    // produce byte-identical blobs.
    // Under orkd ow.w is a daemon handle (no local Bb to dump) — must CPU-tile from the raw int8; direct mode
    // may still dump the NPU-packed tiles. Both produce byte-identical Bb blobs.
    const bool cpu_dump = ctx->via_orkd ? true : ((ork_orkpack_cpu_only() && bi_i8) || !ow.w);
    if (cpu_dump && !bi_i8) {   // no raw int8 available for the CPU dump (only under orkd would we get here) — can't persist
        if (getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK PERSIST] skip %s (no raw int8 for CPU dump)\n", name);
        return;
    }
    size_t tb = cpu_dump ? ork_i8_w_dump_cpu(ctx->npu, K, N, bi_i8, nullptr, 0)
                         : ork_w_dump(ow.w, nullptr, 0);
    std::vector<char> tmp(tb);
    if (cpu_dump) ork_i8_w_dump_cpu(ctx->npu, K, N, bi_i8, tmp.data(), tb);
    else          ork_w_dump(ow.w, tmp.data(), tb);
    // Bf (full-K re-tiled) blob, from the raw int8, stored RIGHT AFTER Bb (contiguous) so the orkd import maps
    // it directly with no runtime rebuild. Only within the Bf run envelope (K%512==0 && K<=4096); else bf==0
    // and the loader/daemon leaves Bf NULL (wide-K weights run the Bb K-split path). Needs bi_i8 (have it here).
    size_t bf = bi_i8 ? ork_i8_w_dump_bf_cpu(ctx->npu, K, N, bi_i8, nullptr, 0) : 0;
    std::vector<char> bftmp(bf);
    if (bf) ork_i8_w_dump_bf_cpu(ctx->npu, K, N, bi_i8, bftmp.data(), bf);
    orkpack_entry e{}; e.K = K; e.N = N; e.dtype = ORKPACK_DT_I8; e.bscale_n = (uint32_t) ow.bscale.size();
    e.blob_off = ctx->persist_off; e.blob_size = tb; e.bf_size = bf;
    fwrite(tmp.data(), 1, tb, ctx->persist_out); ctx->persist_off += tb;
    if (bf) { fwrite(bftmp.data(), 1, bf, ctx->persist_out); ctx->persist_off += bf; }
    e.bscale_off = ctx->persist_off;
    fwrite(ow.bscale.data(), sizeof(float), ow.bscale.size(), ctx->persist_out);
    ctx->persist_off += ow.bscale.size() * sizeof(float);
    ctx->persist_built.emplace_back(std::string(name), e);
}

// Native-W4A4 persist (ORKPACK_DT_I4_NATIVE): dump the already-ROTATED, per-channel-int4-quantized, int4-
// TILED DT_I4 weight (ork_w_dump, dtype-agnostic) + its per-channel bscale, so the mul_mat_i4_hadamard /
// group_i4 cold pack (dequant->FWHT-rotate->int4->tile) is done ONCE at convert and reloaded as a plain DMA
// copy (ork_i4_mm_load). The twin of the int8-tier dump above, for the W4A4 COMPUTE path.
static void ork_persist_write_i4native(ggml_backend_ork_context * ctx, const char * name, int K, int N, const ork_weight & ow) {
    if (ctx->persist_mode != 2 || !ctx->persist_out || !ow.w) return;
    if (!ctx->persist_dumped.insert(name).second) return;   // already dumped (convert-decode re-pack)
    size_t tb = ork_w_dump(ow.w, nullptr, 0);
    if (!tb) return;
    std::vector<char> tmp(tb); ork_w_dump(ow.w, tmp.data(), tb);
    orkpack_entry e{}; e.K = K; e.N = N; e.dtype = ORKPACK_DT_I4_NATIVE; e.bscale_n = (uint32_t) ow.bscale.size();
    e.blob_off = ctx->persist_off; e.blob_size = tb;
    fwrite(tmp.data(), 1, tb, ctx->persist_out); ctx->persist_off += tb;
    e.bscale_off = ctx->persist_off;
    fwrite(ow.bscale.data(), sizeof(float), ow.bscale.size(), ctx->persist_out);
    ctx->persist_off += ow.bscale.size() * sizeof(float);
    ctx->persist_built.emplace_back(std::string(name), e);
    if (getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK PERSIST] i4-native %s K=%d N=%d (%zu B + %u scales)\n", name, K, N, tb, e.bscale_n);
}
// Read a native-W4A4 weight by name (read mode): fills `ow` and returns true on a matching hit (skip the
// cold rotate+pack), false to pack normally. Per-(K,N,dtype) re-checked so a stale .orkpack can't feed
// wrong weights. Single/current pack-domain (native W4A4 is the <4GB compute path; multi-domain is later).
static bool ork_persist_load_i4native(ggml_backend_ork_context * ctx, const char * name, int K, int N, ork_weight & ow) {
    if (ctx->persist_mode != 1 || !ctx->persist_map) return false;
    auto pit = ctx->persist_idx.find(name);
    if (pit == ctx->persist_idx.end() || pit->second.K != (uint32_t) K || pit->second.N != (uint32_t) N ||
        pit->second.dtype != ORKPACK_DT_I4_NATIVE) return false;
    const orkpack_entry & e = pit->second;
    const char * blob = (const char *) ctx->persist_map + e.blob_off;
    // Multi-domain residence: byte-balance this dense int4 weight across domains (same as the int8 resolve +
    // expert paths) so domain_bytes stays accurate (else experts over-admit to domain 0 and overfill it) and
    // retry the next domain on IOVA exhaustion. IMPORT first (bimport, multi-domain-safe); bcreate fallback.
    int _dom = ork_weight_domain(ctx, (size_t) K * N / 2, ork_layer_of(name));
    ork_npu_set_pack_domain(ctx->npu, _dom);
    ow.w = ork_i4_mm_load_import(ctx->npu, K, N, blob, e.blob_size);
    if (!ow.w) ow.w = ork_i4_mm_load(ctx->npu, K, N, blob, e.blob_size);
    while (!ow.w && (_dom = ork_domain_advance(ctx)) >= 0) {
        ow.w = ork_i4_mm_load_import(ctx->npu, K, N, blob, e.blob_size);
        if (!ow.w) ow.w = ork_i4_mm_load(ctx->npu, K, N, blob, e.blob_size);
    }
    if (!ow.w) return false;
    ow.gsize = 0; ow.bscale.resize(e.bscale_n);
    if (e.bscale_n) memcpy(ow.bscale.data(), (const char *) ctx->persist_map + e.bscale_off, (size_t) e.bscale_n * sizeof(float));
    ow.bytes = ork_w_bytes(ow.w); ctx->wcache_bytes += ow.bytes;
    if (ctx->n_domains > 1 && _dom < 64) ctx->domain_bytes[_dom] += ow.bytes;
    ctx->persist_hits++;
    if (getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK PERSIST] i4-native LOAD %s K=%d N=%d dom=%d\n", name, K, N, _dom);
    return true;
}

// Persist ALL n_expert slices of a routed MoE `_exps` tensor (GGML_OP_MUL_MAT_ID src0) in convert/write
// mode. A complete .orkpack must capture every expert, not just the ones the convert prompt routed to, so
// this iterates the whole 3D tensor independent of routing. Each slice is dequantized to its f32 plane
// [N][K] (as ggml's to_float produces), packed once to int8, then handed to ork_persist_write under the
// synthetic key "<name>#<e>" — reusing the exact int8/int4 tier choice + dump path the dense weights use
// (int4 re-packs from the f32 plane internally; the int8 temp weight is only needed for the int8-tier dump).
// Guarded to write mode and deduped via persist_dumped (per synthetic key), so a convert-decode re-pack
// never double-writes. Resident NPU bytes stay ~0: the temp int8 weight is freed before the next slice.
// Fuse dequant(src->f32) + per-output-channel int8 quant for ONE expert slice into caller buffers,
// PARALLEL across the N output channels (each n independent → disjoint regions, no locking), un-pinned
// across all online cores. CPU-only (touches no NPU/ctx state) so it can run on a helper thread while
// the main thread does the serial NPU pack/DMA of another slice.
static void ork_expert_dequant_quant(const struct ggml_tensor * src0, int e, int K, int N,
                                      enum ggml_type type, ggml_to_float_t to_float,
                                      float * f32, int8_t * bi, float * bscale) {
    const char * x = (const char *) src0->data + (size_t) e * src0->nb[2];
    const size_t nb01 = src0->nb[1];
    // Parallel across the N output channels (each independent → no locking) via the PERSISTENT OpenMP pool
    // — NOT a per-slice std::thread spawn/join, which is what capped a MoE convert at ~1 core (thousands of
    // expert slices × pool-teardown). Inner loops are NEON: the max-scan and the quantize both vectorize
    // over K. vcvtnq_s32_f32 rounds ties-to-even, matching lrintf's default rounding → bit-identical output.
    #pragma omp parallel
    {
        ork_unpin_current_thread();                 // each pool thread: spread off the big-core inference pin
        #pragma omp for schedule(dynamic, 16)
        for (int n = 0; n < N; n++) {
            float * frow = f32 + (size_t) n * K;
            if (type == GGML_TYPE_F32) memcpy(frow, x + n*nb01, (size_t) K*sizeof(float));
            else                        to_float(x + n*nb01, frow, K);
            // max |frow[k]| over k (NEON)
            float32x4_t vmax = vdupq_n_f32(1e-9f);
            int k = 0;
            for (; k + 4 <= K; k += 4) vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(frow + k)));
            float mx = vmaxvq_f32(vmax);
            for (; k < K; k++) { float a = fabsf(frow[k]); if (a > mx) mx = a; }
            const float scale_val = mx / 127.0f, inv = 127.0f / mx;
            bscale[n] = scale_val;
            // quantize q=round(frow*inv), clamp [-127,127] (NEON compute; strided store bi[k*N+n])
            const float32x4_t vinv = vdupq_n_f32(inv);
            const int32x4_t   qlo = vdupq_n_s32(-127), qhi = vdupq_n_s32(127);
            k = 0;
            for (; k + 4 <= K; k += 4) {
                int32x4_t q = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(frow + k), vinv));
                q = vminq_s32(vmaxq_s32(q, qlo), qhi);
                bi[(size_t)(k)  *N + n] = (int8_t) vgetq_lane_s32(q, 0);
                bi[(size_t)(k+1)*N + n] = (int8_t) vgetq_lane_s32(q, 1);
                bi[(size_t)(k+2)*N + n] = (int8_t) vgetq_lane_s32(q, 2);
                bi[(size_t)(k+3)*N + n] = (int8_t) vgetq_lane_s32(q, 3);
            }
            for (; k < K; k++) { int q = (int) lrintf(frow[k] * inv); bi[(size_t) k*N + n] = (int8_t) (q > 127 ? 127 : q < -127 ? -127 : q); }
        }
    }
}

static void ork_persist_write_experts(ggml_backend_ork_context * ctx, const struct ggml_tensor * src0,
                                      int K, int N, enum ggml_type type, ggml_to_float_t to_float) {
    if (ctx->persist_mode != 2 || !ctx->persist_out) return;
    const int n_expert = (int) src0->ne[2];

    // Experts still needing a (re)build this run — skip any already dumped.
    std::vector<int> todo;
    for (int e = 0; e < n_expert; e++)
        if (!ctx->persist_dumped.count(ork_expert_key(src0->name, e))) todo.push_back(e);
    if (todo.empty()) return;

    // NATIVE W4A4 (int4 tier; hadamard implied): emit each expert as ORKPACK_DT_I4_NATIVE — FWHT-rotate the
    // weight columns, per-channel int4-quant (mx/7), ork_i4_mm_pack + ork_w_dump — the exact form the run
    // path's native-W4A4 expert branch loads via ork_i4_mm_load (twin of the dense ork_persist_write_i4native).
    // SERIAL over experts because ork_i4_mm_pack is a single-stream NPU op (inner column loop is OMP-parallel);
    // NOT the parallel-CPU O4N1/int8 loop below. Deduped per synthetic key like the int8/O4N1 path.
    if (ctx->qbits == 4 && ctx->hadamard) {
        const int fb = K & (-K);
        std::vector<float> f32((size_t) N * K); std::vector<int8_t> bi((size_t) K * N); std::vector<float> bs(N);
        const size_t nb2 = src0->nb[2], nb01 = src0->nb[1];
        for (int e : todo) {
            const std::string key = ork_expert_key(src0->name, e);
            if (ctx->persist_dumped.count(key)) continue;
            const char * x = (const char *) src0->data + (size_t) e * nb2;
            #pragma omp parallel for schedule(static)
            for (int n = 0; n < N; n++) {
                float * col = f32.data() + (size_t) n * K;
                if (type == GGML_TYPE_F32) memcpy(col, x + (size_t) n*nb01, (size_t) K*sizeof(float));
                else                        to_float(x + (size_t) n*nb01, col, K);
                for (int o = 0; o < K; o += fb) ork_fwht_norm(col + o, fb);
                float mx = 1e-9f; for (int k = 0; k < K; k++) { float v = fabsf(col[k]); if (v > mx) mx = v; }
                float s = mx / 7.0f; bs[n] = s;
                for (int k = 0; k < K; k++) { int q = (int) lrintf(col[k] / s);
                    bi[(size_t) k*N + n] = (int8_t) (q > 7 ? 7 : q < -8 ? -8 : q); }
            }
            ork_w * w = ork_i4_mm_pack(ctx->npu, K, N, bi.data());
            if (!w) { fprintf(stderr, "[ORK PERSIST] expert %s#%d i4-native pack FAILED\n", src0->name, e); continue; }
            size_t tb = ork_w_dump(w, nullptr, 0);
            std::vector<char> blob(tb); ork_w_dump(w, blob.data(), tb);
            ork_mm_free(ctx->npu, w);
            orkpack_entry ent{}; ent.K = K; ent.N = N; ent.dtype = ORKPACK_DT_I4_NATIVE; ent.bscale_n = (uint32_t) N;
            if (ctx->persist_dumped.insert(key).second) {
                ent.blob_off = ctx->persist_off; ent.blob_size = tb;
                fwrite(blob.data(), 1, tb, ctx->persist_out); ctx->persist_off += tb;
                ent.bscale_off = ctx->persist_off;
                fwrite(bs.data(), sizeof(float), N, ctx->persist_out); ctx->persist_off += (size_t) N * sizeof(float);
                ctx->persist_built.emplace_back(key, ent);
            }
        }
        return;
    }

    // COMPUTE→DMA PIPELINE. The per-expert work splits into a parallel CPU half (dequant+quant) and a
    // serial NPU/IO half (ork_i8_mm_pack's bcreate + IOMMU-map + bsync DMA, then dump). The serial half
    // is single-stream and leaves every core idle (~70% idle measured). So double-buffer: while THIS
    // thread runs the serial NPU pack/dump of expert i, a helper thread dequant+quants expert i+1 (all
    // cores) into the other buffer. Producer touches no NPU/ctx state → safe alongside the consumer.
    // Consumed strictly in `todo` order, so the .orkpack is bit-identical to the serial version.
    // FLATTEN over experts: the true bottleneck was serialization (per-expert produce/tile barriers +
    // the serial per-tile NPU bcreate) leaving cores idle — nothing was resource-bound. So make it
    // embarrassingly parallel: each core independently packs a WHOLE expert on the CPU (dequant → int4/int8
    // pack, NO NPU/bcreate) and appends under a short critical section. No per-expert barrier, no serial NPU
    // pack → saturates all cores AND frees the NPU. Byte-identical to the serial ork_persist_write path:
    // ork_i4a8_pack_cpu_blob == ork_i4a8_mm_pack_im+dump (validated bit-exact), int8 CPU tile == NPU tile.
    (void) ork_orkpack_tier(src0->name, K, N, type);   // pre-warm tier static-init (avoid first-touch race)
    (void) ork_imatrix_lookup(src0->name, K);          // pre-warm imatrix map (thread-safe reads afterward)
    const size_t nb2 = src0->nb[2], nb01 = src0->nb[1];
    #pragma omp parallel
    {
        // NOTE: do NOT un-pin here. Spreading pack workers across all 8 cores is a net LOSS on RK3588 —
        // the 4 little A55s are ~half-speed and drag the tail (measured: OMP=8 1821 KB/s vs OMP=4-big 2023).
        // Keep threads on whatever affinity the caller set (pin the convert to the big cluster, taskset 4-7).
        std::vector<float> f32((size_t) N * K);
        std::vector<int8_t> bi; std::vector<float> bs; std::vector<char> blob;
        #pragma omp for schedule(dynamic, 1)
        for (size_t idx = 0; idx < todo.size(); idx++) {
            const int e = todo[idx];
            const char * x = (const char *) src0->data + (size_t) e * nb2;
            for (int n = 0; n < N; n++) {                          // dequant this expert → f32 [N][K]
                float * fr = f32.data() + (size_t) n * K;
                if (type == GGML_TYPE_F32) memcpy(fr, x + n*nb01, (size_t) K * sizeof(float));
                else                        to_float(x + n*nb01, fr, K);
            }
            const std::string key = ork_expert_key(src0->name, e);
            const int tier = ork_orkpack_tier(src0->name, K, N, type);
            orkpack_entry ent{}; ent.K = K; ent.N = N;
            if (tier == 4) {
                const float * im = ork_imatrix_lookup(src0->name, K);
                int nf4 = (ork_src_type_bits(type) >= 16.0 || ork_moe_auto()) ? 1 : 0;   // MoE auto: ALWAYS NF4 (one scheme per model type); else source-routed
                size_t tb = ork_i4a8_pack_cpu_blob(ctx->npu, K, N, f32.data(), im, nf4, nullptr, 0);
                blob.resize(tb); ork_i4a8_pack_cpu_blob(ctx->npu, K, N, f32.data(), im, nf4, blob.data(), tb);
                ent.dtype = ORKPACK_DT_I4; ent.bscale_n = 0; ent.blob_size = tb; ent.bscale_off = 0;
                #pragma omp critical (ork_persist_flat)
                if (ctx->persist_dumped.insert(key).second) {
                    ent.blob_off = ctx->persist_off;
                    fwrite(blob.data(), 1, tb, ctx->persist_out); ctx->persist_off += tb;
                    ctx->persist_built.emplace_back(key, ent);
                }
            } else {
                bi.resize((size_t) K * N); bs.resize(N);
                for (int n = 0; n < N; n++) {                      // per-channel int8 quant (NEON) — matches NPU path
                    const float * fr = f32.data() + (size_t) n * K;
                    float32x4_t vmax = vdupq_n_f32(1e-9f); int k = 0;
                    for (; k + 4 <= K; k += 4) vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(fr + k)));
                    float mx = vmaxvq_f32(vmax); for (; k < K; k++) { float a = fabsf(fr[k]); if (a > mx) mx = a; }
                    const float scale = mx / 127.0f, inv = 127.0f / mx; bs[n] = scale;
                    const float32x4_t vinv = vdupq_n_f32(inv); const int32x4_t lo = vdupq_n_s32(-127), hi = vdupq_n_s32(127);
                    k = 0;
                    for (; k + 4 <= K; k += 4) {
                        int32x4_t q = vminq_s32(vmaxq_s32(vcvtnq_s32_f32(vmulq_f32(vld1q_f32(fr + k), vinv)), lo), hi);
                        bi[(size_t)(k)  *N+n]=(int8_t)vgetq_lane_s32(q,0); bi[(size_t)(k+1)*N+n]=(int8_t)vgetq_lane_s32(q,1);
                        bi[(size_t)(k+2)*N+n]=(int8_t)vgetq_lane_s32(q,2); bi[(size_t)(k+3)*N+n]=(int8_t)vgetq_lane_s32(q,3);
                    }
                    for (; k < K; k++) { int q=(int)lrintf(fr[k]*inv); bi[(size_t)k*N+n]=(int8_t)(q>127?127:q<-127?-127:q); }
                }
                size_t tb = ork_i8_w_dump_cpu_st(ctx->npu, K, N, bi.data(), nullptr, 0);
                blob.resize(tb); ork_i8_w_dump_cpu_st(ctx->npu, K, N, bi.data(), blob.data(), tb);
                ent.dtype = ORKPACK_DT_I8; ent.bscale_n = (uint32_t) N; ent.blob_size = tb;
                #pragma omp critical (ork_persist_flat)
                if (ctx->persist_dumped.insert(key).second) {
                    ent.blob_off = ctx->persist_off;
                    fwrite(blob.data(), 1, tb, ctx->persist_out); ctx->persist_off += tb;
                    ent.bscale_off = ctx->persist_off;
                    fwrite(bs.data(), sizeof(float), N, ctx->persist_out); ctx->persist_off += (size_t) N * sizeof(float);
                    ctx->persist_built.emplace_back(key, ent);
                }
            }
        }
    }
}

// Write the index + footer and atomically rename the .tmp into place (skip if nothing was packed).
static void ork_persist_finalize(ggml_backend_ork_context * ctx) {
    if (ctx->persist_mode != 2 || !ctx->persist_out) return;
    if (ctx->persist_built.empty()) {
        fclose(ctx->persist_out); ctx->persist_out = nullptr; unlink(ctx->persist_tmp.c_str()); return;
    }
    uint64_t index_off = ctx->persist_off;
    for (auto & kv : ctx->persist_built) {
        uint32_t nl = (uint32_t) kv.first.size();
        fwrite(&nl, 4, 1, ctx->persist_out);
        fwrite(kv.first.data(), 1, nl, ctx->persist_out);
        fwrite(&kv.second, sizeof(orkpack_entry), 1, ctx->persist_out);
    }
    orkpack_footer f; memset(&f, 0, sizeof f);
    f.index_off = index_off; f.n_entries = (uint32_t) ctx->persist_built.size(); f.version = ORKPACK_VERSION;
    f.ork_fmt = ork_pack_format_version();   // stamp the ork-driver pack-compat token (its MAJOR ver)
    f.quant_sig = ork_build_sig();           // stamp the build-config precision signature (authoritative on read)
    memcpy(f.magic, ORKPACK_MAGIC, 8);
    fwrite(&f, sizeof f, 1, ctx->persist_out);
    fflush(ctx->persist_out); fclose(ctx->persist_out); ctx->persist_out = nullptr;
    rename(ctx->persist_tmp.c_str(), ctx->persist_final.c_str());
    // Unconditional success line: report the full path (directory + filename) and weight count so the user sees
    // exactly where the pack landed. Split dir/file for clarity when the path is absolute.
    {
        const std::string & fp = ctx->persist_final;
        size_t slash = fp.find_last_of('/');
        std::string dir  = slash == std::string::npos ? std::string(".") : fp.substr(0, slash);
        std::string file = slash == std::string::npos ? fp : fp.substr(slash + 1);
        fprintf(stderr, "[ORK PERSIST] SUCCESS: orkpack written (%u weights, %.1f MiB) -> %s\n"
                        "              dir: %s  file: %s\n",
                f.n_entries, (double) ctx->persist_off / (1024.0 * 1024.0), fp.c_str(), dir.c_str(), file.c_str());
    }
}

// Resolve the int8 weight plane `x` (= src0->data) to a packed, cached ork_weight, identically for the
// non-chain AND chain int8 matmul paths so every matmul entry persists the SAME way. Three tiers:
//   1. wcache hit  — already resident, just return it.
//   2. .orkpack read (persist_mode==1) — load the pre-tiled bytes (int8 or int4-W4A8) straight into DMA.
//   3. pack-miss — dequant src0->f32, per-channel int8-quantize, pack, and (write mode) dump to .orkpack.
// Returns ctx->wcache.end() only on a pack/load failure; otherwise an iterator to the resident weight.
// Caller does the LRU touch (it->second.last_use) and the matmul. Keying / dedup is by src0->data
// (wcache) and src0->name (persist index / persist_dumped) — same keys both paths use, so no double-write.
// STREAM-POOL: convert a freshly-resolved wcache entry (whose ow.w is a just-packed/loaded IOVA-resident
// int8 weight) into a RAM-resident stream entry: dump the tiled int8 bytes, add to the pool (the ONE-TIME
// inflate/copy held in RAM), free the temporary IOVA weight, then cheap-map it back to IOVA. RAM-evicts
// first (budget by RAM), then IOVA-evicts (budget by the 4 GiB window). Returns false on failure (caller
// keeps the plain IOVA weight as fallback — still correct, just not RAM-tiered). it->second.w holds the
// temp weight on entry; on success se is set, w is the live mapped handle (NULL — run goes via se).
static bool ork_spool_install(ggml_backend_ork_context * ctx,
                              std::unordered_map<const void *, ork_weight>::iterator it, int K, int N) {
    ork_weight & ow = it->second;
    if (!ctx->spool || !ow.w) return false;
    size_t need = ork_w_dump(ow.w, nullptr, 0);    // tiled int8 blob size
    if (!need) return false;
    std::vector<uint8_t> blob(need);
    if (ork_w_dump(ow.w, blob.data(), need) != need) return false;
    // Build the RAM-resident entry FIRST (don't free the temp IOVA weight until add succeeds — on failure
    // e.g. N%32!=0 the entry stays a valid plain IOVA weight, so the matmul is still correct).
    ork_spool_ram_evict(ctx, need);
    ork_stream_entry * se = ork_i8_stream_pool_add(ctx->spool, K, N, blob.data(), need);
    if (!se) return false;    // fall back to ow.w (still resident & counted)
    // free the temp IOVA weight (it was counted in wcache_bytes by the caller); the RAM entry replaces it
    ctx->wcache_bytes -= ow.bytes;
    ork_slice_ws_drop(ctx, ow.w);
    ork_mm_free(ctx->npu, ow.w); ow.w = nullptr; ow.bytes = 0;
    ow.se = se; ow.ram_bytes = ork_stream_entry_bytes(se); ctx->spool_ram_bytes += ow.ram_bytes;
    ork_spool_iova_evict(ctx, ow.ram_bytes);
    if (ork_spool_map_retry(ctx, se) != 0) return false;   // mapped-fail: entry in RAM, remap retried on next touch
    ow.bytes = ow.ram_bytes; ctx->wcache_bytes += ow.bytes;
    return true;
}

static std::unordered_map<const void *, ork_weight>::iterator
ork_resolve_weight_i8(ggml_backend_ork_context * ctx, const struct ggml_tensor * src0,
                      int K, int N, size_t nb01, enum ggml_type type, ggml_to_float_t to_float,
                      bool allow_evict, int expert = -1, int proc_prec = 0) {
    // expert>=0 OVERLOAD (MoE prefill layer-streamer): resolve ONE expert slice of a 3D _exps tensor
    // through the SAME streaming path as dense (zero-copy import + ork_weight_domain + evict), keyed by
    // the per-expert pointer + orkpack entry. expert<0 = the original dense behavior, byte-for-byte.
    // proc_prec PROCESSING PRECISION: 0 = int8-inflate (default; int4 nibbles -> int8 resident, amortized
    // compute but ~2x resident bytes) — the dense/current path. 1 = int4-NATIVE-RESIDENT (W4A4): keep the
    // weight DT_I4 in IOVA (no inflate -> ~half the bytes so a big MoE fits RESIDENT), run via ork_i4_mm_run.
    // proc_prec=1 needs the orkpack to store the expert as a native DT_I4 dump (ork_i4_mm_pack) + the caller's
    // run path to use the int4 doorbell — else the DT_I4 load returns NULL and the caller falls to CPU.
    // allow_evict: the non-chain path uses one weight at a time, so it may stream-evict the LRU to
    // free IOVA. The chain path needs ALL `count` weights co-resident at submit, so it passes false
    // (matching the original chain pack, which never evicted) — otherwise packing weight i frees the
    // already-packed weight i-1 that the chain still references → use-after-free at submit.
    const double _r0 = ctx->profile ? ork_now_us_e() : 0;
    const char * x = (const char *) src0->data + (expert >= 0 ? (size_t) expert * src0->nb[2] : 0);   // MoE: per-expert slice key
    auto it = ctx->wcache.find(x);
    if (it != ctx->wcache.end()) {
        if (ctx->spool && it->second.se && !ork_stream_entry_mapped(it->second.se)) {
            // STREAM-POOL hit, but IOVA-unmapped (evicted from the hot tier): re-map (cheap, ~170us;
            // NO re-inflate). Make IOVA room first by unmapping the LRU mapped entries.
            const double _m0 = ctx->profile ? ork_now_us_e() : 0;
            if (allow_evict) ork_spool_iova_evict(ctx, it->second.ram_bytes);
            if (ork_spool_map_retry(ctx, it->second.se) == 0) {
                it->second.bytes = it->second.ram_bytes;
                ctx->wcache_bytes += it->second.bytes;
                ctx->spool_remaps++;
                if (ctx->profile) { ctx->s_remap += ork_now_us_e() - _m0; ctx->n_remap++; }
            } else return ctx->wcache.end();
        }
        if (ctx->profile) { ctx->s_resolve += ork_now_us_e() - _r0; ctx->n_hit++; }
        return it;
    }

    if (ctx->persist_mode == 1) {
        // .orkpack hit: load pre-tiled bytes straight into DMA (no dequant/quant/tile). Per-weight
        // (K,N,dtype) is re-checked so a stale file can't feed wrong weights — mismatch → pack below.
        auto pit = ctx->persist_idx.find(expert >= 0 ? ork_expert_key(src0->name, expert) : std::string(src0->name));   // MoE: per-expert orkpack entry
        if (pit != ctx->persist_idx.end() && pit->second.K == (uint32_t) K && pit->second.N == (uint32_t) N &&
            (pit->second.dtype == ORKPACK_DT_I8 || pit->second.dtype == ORKPACK_DT_I4)) {
            const orkpack_entry & e = pit->second;
            // STREAM-POOL bugfix (see pack-miss path): unmap LRU (cheap remap later) instead of erasing.
            if (allow_evict) {
                if (ctx->spool) ork_spool_iova_evict(ctx, (size_t) K * N);
                else            ork_wcache_evict(ctx, (size_t) K * N);
            }
            ork_weight ow;
            const char * blob = (const char *) ctx->persist_map + e.blob_off;
            int _dom = ork_weight_domain(ctx, (size_t) K * N, ork_layer_of(src0->name));   // multi-domain residence: byte-balanced + layer-aligned (advance only at layer boundaries)
            ork_npu_set_pack_domain(ctx->npu, _dom);
            if (!ctx->load_phase) ctx->mem_create_runtime++;       // any pack/load after fill = churn (must be 0)
            for (;;) {                                             // retry in the next domain on IOVA exhaustion
                // ZERO-COPY dma-buf import is the DEFAULT (single- AND multi-domain). It maps the .orkpack's
                // page-cache pages straight into the NPU IOVA — the NPU reads weights in place, no host gather.
                // The earlier "force copy under multi-domain" workaround is GONE: the >4GiB regression was
                // root-caused (ork-driver 0.6.42) to (1) importing into a fresh non-0 domain — fixed by the
                // per-domain native anchor (ork_dom_prime); (2) too many imported mappings per chained submit
                // and (3) oversized single imports — both fixed by size-bounded chunked import. Validated
                // bit-exact + llama.cpp perplexity 3.30 on the 7B across 5 domains. `ORK_NO_IMPORT` is the only
                // opt-out (falls back to the bcreate COPY load — same pre-tiled bytes, a host memcpy per weight).
                const bool no_import = env_enabled("ORK_NO_IMPORT");
                if (e.dtype == ORKPACK_DT_I4 && proc_prec == 1) {
                    // int4-NATIVE-RESIDENT (W4A4): keep the weight DT_I4 in IOVA (no int8 inflate -> ~half the
                    // resident bytes, so a big MoE fits RESIDENT and escapes the int8-inflate load wall), run
                    // via ork_i4_mm_run / the int4 doorbell. NOTE: ork_i4_mm_load wants a NATIVE DT_I4 dump
                    // (ork_i4_mm_pack + ork_w_dump); on the current O4N1 compact blob it returns NULL -> caller
                    // falls to CPU. Enable end-to-end once ork_persist_write_experts emits native DT_I4.
                    ow.w = ork_i4_mm_load(ctx->npu, K, N, blob, e.blob_size);
                    if (ow.w) { const float * bs = ork_w_bscale(ow.w); if (bs) ow.bscale.assign(bs, bs + N); }
                } else if (e.dtype == ORKPACK_DT_I4) {
                    if (!no_import) ow.w = ork_i4a8_mm_load_import(ctx->npu, K, N, blob, e.blob_size);   // ZERO-COPY: map .orkpack page-cache pages into IOVA
                    if (!ow.w) ow.w = ork_i4a8_mm_load(ctx->npu, K, N, blob, e.blob_size);  // copy (import off / unavailable)
                    if (ow.w) { const float * bs = ork_w_bscale(ow.w); if (bs) ow.bscale.assign(bs, bs + N); }
                } else {
                    const double _l0 = ctx->profile ? ork_now_us_e() : 0;
                    if (ctx->via_orkd) {
                        // orkd: the client allocs a dma-buf, copies the pre-tiled blob (Bb, then the contiguous
                        // Bf region) in, and hands the fd to the daemon (ORKD_IMPORT) which maps it into this
                        // client's domain — no daemon tiling/ownership. Bf sits right after Bb (relative offset
                        // = blob_size), so the daemon lays base matmuls' fast-path Bf as views; bf_size==0 =>
                        // no Bf (wide-K weights run the Bb K-split path).
                        size_t blob_n = (size_t) e.blob_size + (size_t) e.bf_size;
                        ow.w = ork_i8_mm_import(ctx->npu, K, N, blob, blob_n, e.bf_size ? (size_t) e.blob_size : 0);
                    } else {
                        if (!no_import) ow.w = ork_i8_mm_load_import(ctx->npu, K, N, blob, e.blob_size);     // ZERO-COPY: map .orkpack page-cache pages into IOVA
                        if (!ow.w) ow.w = ork_i8_mm_load(ctx->npu, K, N, blob, e.blob_size); // copy (import off / unavailable)
                    }
                    if (ctx->profile) { ctx->s_load += ork_now_us_e() - _l0; ctx->n_loadhit++; }
                    if (ow.w) { const float * bs = (const float *) ((const char *) ctx->persist_map + e.bscale_off);
                                ow.bscale.assign(bs, bs + e.bscale_n); }
                }
                if (ow.w || (_dom = ork_domain_advance(ctx)) < 0) break;   // packed, or all domains exhausted
            }
            if (ow.w) {
                it = ctx->wcache.emplace(x, std::move(ow)).first;
                it->second.bytes = ork_w_bytes(it->second.w);
                ctx->wcache_bytes += it->second.bytes;
                if (ctx->n_domains > 1 && _dom < 64) ctx->domain_bytes[_dom] += it->second.bytes;
                ctx->persist_hits++;
                if (getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK PERSIST] %s %s K=%d N=%d\n",
                    e.dtype == ORKPACK_DT_I4 ? "int4-load" : "int8-load", src0->name, K, N);
                ork_evict_src(x, (size_t) N * nb01);
                // STREAM-POOL: move the loaded int8 weight into the RAM tier (int4 stays IOVA-resident —
                // the pool stores inflated int8; converting it here would lose the RAM-size win).
                if (ctx->spool && e.dtype == ORKPACK_DT_I8) {
                    const double _i0 = ctx->profile ? ork_now_us_e() : 0;
                    bool ok_inst = ork_spool_install(ctx, it, K, N);
                    if (ctx->profile) ctx->s_install += ork_now_us_e() - _i0;
                    if (!ok_inst && !it->second.w && !it->second.se) { ctx->wcache.erase(it); return ctx->wcache.end(); }
                }
                if (ctx->profile) ctx->s_resolve += ork_now_us_e() - _r0;
                return it;
            }
        }
    }

    // MoE experts are ORKPACK-ONLY on this path: a persist miss does NOT live-pack here (the code below
    // assumes a dense 2D Q8_0 weight, not an expert slice) — return miss so the MoE handler falls the
    // expert to the CPU NF4 path.
    if (expert >= 0) { if (ctx->profile) ctx->s_resolve += ork_now_us_e() - _r0; return ctx->wcache.end(); }
    // pack-miss: dequant -> per-channel int8 quant -> pack -> (write mode) persist
    if (ctx->persist_mode) ctx->persist_misses++;
    if (ctx->persist_mode == 1) {   // READ mode + miss = the SILENT slow-path trap: the .orkpack lacks this
        // weight (name/shape/dtype mismatch or incomplete pack) so we fall back to live Q8_0->int8-tile
        // conversion (~25x the orkpack load — measured 16.7s vs 0.66s resolve on the 1.7B). Make it LOUD
        // (bounded) so "it's using Q8_0 not the orkpack" can never hide behind a silent fallback again.
        static long _opk_warned = 0;
        if (_opk_warned < 8)
            fprintf(stderr, "[ork] WARN: %s (K=%d N=%d) NOT in the loaded .orkpack -> live Q8_0->int8-tile "
                    "conversion (SLOW). Rebuild the .orkpack for THIS model (name/shape mismatch or stale pack).\n",
                    src0->name, K, N);
        else if (_opk_warned == 8)
            fprintf(stderr, "[ork] WARN: (further .orkpack-miss warnings suppressed; see persist_misses in profile)\n");
        _opk_warned++;
    }
    if (ctx->profile) ctx->n_packmiss++;
    ctx->f32.resize((size_t) N * K);
    ctx->bi .resize((size_t) K * N);
    float  * f32 = ctx->f32.data();
    int8_t * bi  = ctx->bi.data();
    ork_weight ow; ow.bscale.resize(N);
    const double _d0 = ctx->profile ? ork_now_us_e() : 0;
    // Dequant (Q8_0 -> f32) + per-output-channel symmetric int8 quant, FUSED and PARALLEL across the N
    // output channels. Each channel n is independent: it reads x[n], writes f32[n*K..] and bi[.,n] +
    // bscale[n] — disjoint regions, so no locking. This is the residence-fill / .orkpack-conversion hot
    // path (was single-threaded — ~2 min on a 7B); there is no live inference to protect during a pack,
    // so use ALL cores (big+little), dynamic core count from the OS, no knob.
    {
        int nthr = (int) sysconf(_SC_NPROCESSORS_ONLN); if (nthr < 1) nthr = 1;
        if (nthr > N) nthr = (int) N;
        auto worker = [&](int n0, int n1) {
            // Un-pin: the host pins this worker process to the big cluster for inference, but a
            // pack has no live inference to protect — spread across ALL cores (ork-driver owns
            // the affinity logic). Otherwise these threads inherit the big-core pin and the
            // conversion never saturates the little cores.
            ork_unpin_current_thread();
            for (int n = n0; n < n1; n++) {
                float * frow = f32 + (size_t) n * K;
                if (type == GGML_TYPE_F32) memcpy(frow, x + n*nb01, (size_t) K*sizeof(float));
                else                        to_float((const char *) x + n*nb01, frow, K);
                float mx = 1e-9f;
                for (int k = 0; k < K; k++) { float v = fabsf(frow[k]); if (v > mx) mx = v; }
                float scale_val = mx / 127.0f; ow.bscale[n] = scale_val;
                for (int k = 0; k < K; k++) {
                    int q = (int) lrintf(frow[k] / scale_val);
                    bi[(size_t) k*N + n] = (int8_t) (q > 127 ? 127 : q < -127 ? -127 : q);
                }
            }
        };
        if (nthr <= 1) worker(0, (int) N);
        else {
            std::vector<std::thread> th; int chunk = ((int) N + nthr - 1) / nthr;
            for (int t = 0; t < nthr; t++) { int a = t*chunk, b = std::min((int) N, a+chunk); if (a < b) th.emplace_back(worker, a, b); }
            for (auto & t : th) t.join();
        }
    }
    if (ctx->profile) ctx->s_dequant += ork_now_us_e() - _d0;
    // Multi-domain residence: weights pack once into their IOMMU domain (ork_weight_domain byte-balances
    // them across domains; dom_activate swaps the active domain per submit). A 7B fits ~2 domains fully
    // resident — no eviction/churn. wcache_evict only fires if the whole set exceeds the (large) budget.
    if (allow_evict) {
        if (ctx->spool) ork_spool_iova_evict(ctx, (size_t) K * N);
        else            ork_wcache_evict(ctx, (size_t) K * N);
    }
    const double _p0 = ctx->profile ? ork_now_us_e() : 0;
    int _dom = ork_weight_domain(ctx, (size_t) K * N, ork_layer_of(src0->name));   // multi-domain residence: byte-balanced + layer-aligned (advance only at layer boundaries)
    ork_npu_set_pack_domain(ctx->npu, _dom);
    if (!ctx->load_phase) ctx->mem_create_runtime++;       // any pack after fill = churn (must be 0)
    ow.w = ork_i8_mm_pack(ctx->npu, K, N, bi);
    while (!ow.w && (_dom = ork_domain_advance(ctx)) >= 0)  // domain's IOVA full -> next domain, retry
        ow.w = ork_i8_mm_pack(ctx->npu, K, N, bi);
    if (ctx->profile) ctx->s_pack += ork_now_us_e() - _p0;
    if (!ow.w) return ctx->wcache.end();
    it = ctx->wcache.emplace(x, std::move(ow)).first;
    it->second.bytes = ork_w_bytes(it->second.w);
    ctx->wcache_bytes += it->second.bytes;
    if (ctx->n_domains > 1 && _dom < 64) ctx->domain_bytes[_dom] += it->second.bytes;
    if (ctx->slice_route && !ctx->spool && (K > 4096 || N > 8192)) {   // wide int8: pack a sliced-doorbell twin (A/B; +resident bytes; not under spool, whose ork_w is remapped)
        ork_w_sliced * ws = ork_mm_pack_sliced(ctx->npu, K, N, bi, ORK_DT_I8);   // bi = int8 weight bytes (still valid until ork_evict_src below)
        if (ws) ctx->slice_ws[it->second.w] = ws;
        else if (getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK] slice_route: sliced pack failed K=%d N=%d (IOVA?) -> blocking path\n", K, N);
    }
    ork_persist_write(ctx, src0->name, K, N, it->second, ctx->f32.data(), type, bi);   // .orkpack: dump for next time (f32 plane enables int4 tier; src type drives tier; bi = int8 CPU-dump fallback if NPU busy)
    ork_evict_src(x, (size_t) N * nb01);   // source plane now dead weight (custom loader)
    // STREAM-POOL: move the just-packed int8 weight into the RAM tier (cheap remaps on future hits).
    if (ctx->spool) {
        const double _i0 = ctx->profile ? ork_now_us_e() : 0;
        bool ok_inst = ork_spool_install(ctx, it, K, N);
        if (ctx->profile) ctx->s_install += ork_now_us_e() - _i0;
        if (!ok_inst && !it->second.w && !it->second.se) { ctx->wcache.erase(it); return ctx->wcache.end(); }
    }
    if (getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK DEBUG] packed weight, wcache=%zu resident=%.0fMB, K=%d, N=%d, x=%p\n",
        ctx->wcache.size(), ctx->wcache_bytes/1e6, K, N, (const void *) x);
    if (ctx->profile) ctx->s_resolve += ork_now_us_e() - _r0;
    return it;
}

void ggml_backend_ork_set_hybrid(bool use_hybrid) {
    g_ork_hybrid_loading = use_hybrid;
}

void ggml_backend_ork_set_load_config(bool dflash, bool silu_int8_fused) {
    g_ork_cfg_set = true; g_ork_cfg_dflash = dflash; g_ork_cfg_silu_int8fused = silu_int8_fused;
}
static inline double ork_now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec*1e6 + t.tv_nsec*1e-3; }

// dst = src0 x src1 :  src0 [K=ne00, N=ne01], src1 [K=ne10=ne00, M=ne11], dst [N, M] (row-major [M][N])
// Central dispatcher: pick the right ork-driver entry for a set of independent int8 matmul tasks
// (a single weight, or a chain of matmuls that share src1 — Q/K/V, gate/up — hence data-independent).
//   • 1 task              → ork_i8_mm_run (does its own K-split / N-tiling / M-scheduler multicore)
//   • N, SAME domain      → ork_i8_mm_run_stream — round-robin the independent matmuls across the 3 NPU
//                           cores concurrently (the NPU has ONE active IOMMU domain at a time, so RR is
//                           only valid within a domain; layer-aligned placement keeps a chain co-domain)
//   • N, cross-domain     → ork_i8_mm_run_chain (single-core chain; cross-domain concurrency unsupported)
// On any error, falls back to sequential per-task run_i8. Returns true on success.
// Shared NEON f32->int8 per-row activation quant: amr[k]=round(yr[k]*127/absmax), returns scale=absmax/127.
// Round-half-away-from-zero via copysignf(0.5) — BIT-IDENTICAL to the scalar and to the dense mul_mat_i8
// NEON path. Used to vectorize the MoE chain-path act-quant (was scalar — the profiled ~385ms on 35B decode).
static inline float ork_quant_row_i8(const float * yr, int K, int8_t * amr) {
    float mx = 1e-9f;
#if defined(__ARM_NEON)
    int k = 0;
    float32x4_t v_mx0 = vdupq_n_f32(mx), v_mx1 = vdupq_n_f32(mx);
    for (; k <= K - 8; k += 8) {
        v_mx0 = vmaxq_f32(v_mx0, vabsq_f32(vld1q_f32(yr + k)));
        v_mx1 = vmaxq_f32(v_mx1, vabsq_f32(vld1q_f32(yr + k + 4)));
    }
    mx = vmaxvq_f32(vmaxq_f32(v_mx0, v_mx1));
    for (; k < K; k++) { float v = fabsf(yr[k]); mx = v > mx ? v : mx; }
    const float inv = 127.0f / mx;
    const float32x4_t v_inv = vdupq_n_f32(inv), v_half = vdupq_n_f32(0.5f);
    const uint32x4_t  v_sign = vdupq_n_u32(0x80000000u);
    const int32x4_t   v_hi = vdupq_n_s32(127), v_lo = vdupq_n_s32(-127);
    k = 0;
    for (; k <= K - 8; k += 8) {
        float32x4_t v_q0 = vmulq_f32(vld1q_f32(yr + k),     v_inv);
        float32x4_t v_q1 = vmulq_f32(vld1q_f32(yr + k + 4), v_inv);
        float32x4_t v_h0 = vreinterpretq_f32_u32(vorrq_u32(vandq_u32(vreinterpretq_u32_f32(v_q0), v_sign), vreinterpretq_u32_f32(v_half)));
        float32x4_t v_h1 = vreinterpretq_f32_u32(vorrq_u32(vandq_u32(vreinterpretq_u32_f32(v_q1), v_sign), vreinterpretq_u32_f32(v_half)));
        int32x4_t v_i0 = vcvtq_s32_f32(vaddq_f32(v_q0, v_h0));
        int32x4_t v_i1 = vcvtq_s32_f32(vaddq_f32(v_q1, v_h1));
        v_i0 = vmaxq_s32(vminq_s32(v_i0, v_hi), v_lo);
        v_i1 = vmaxq_s32(vminq_s32(v_i1, v_hi), v_lo);
        int16x8_t v_s16 = vcombine_s16(vmovn_s32(v_i0), vmovn_s32(v_i1));
        vst1_s8(amr + k, vmovn_s16(v_s16));
    }
    for (; k < K; k++) {
        float q = yr[k] * inv;
        int qi = (int) (q + copysignf(0.5f, q));
        amr[k] = (int8_t) (qi > 127 ? 127 : qi < -127 ? -127 : qi);
    }
#else
    for (int kk = 0; kk < K; kk++) { float v = fabsf(yr[kk]); mx = v > mx ? v : mx; }
    const float inv = 127.0f / mx;
    for (int kk = 0; kk < K; kk++) {
        float q = yr[kk] * inv;
        int qi = (int) (q + copysignf(0.5f, q));
        amr[kk] = (int8_t) (qi > 127 ? 127 : qi < -127 ? -127 : qi);
    }
#endif
    return mx / 127.0f;
}

static bool ork_dispatch_i8(ggml_backend_ork_context * ctx, std::vector<ork_mm_task_i8> & tasks) {
    if (tasks.empty()) return true;
    // orkd: run_i8 AND run_chain_i8 are daemon-routed (run_stream_i8 is fd-local). Chain N tasks into ONE
    // routed round-trip (orkd_run_chain_i8 — needs daemon-resident is_orkd weights, which the routed pack
    // guarantees here); if the chain declines (rc!=0), fall back to sequential routed run_i8.
    if (ctx->via_orkd) {
        if (tasks.size() > 1 && ork_i8_mm_run_chain(ctx->npu, (int) tasks.size(), tasks.data()) == 0)
            return true;
        for (size_t t = 0; t < tasks.size(); t++)
            if (ork_i8_mm_run(ctx->npu, tasks[t].w, tasks[t].M, tasks[t].A, tasks[t].C)) return false;
        return true;
    }
    int rc;
    if (tasks.size() == 1) {
        // ORK_SLICE_ROUTE — wedge-safety RESCUE, keyed on the SPECIFIC refusal code, MEMOIZED per (weight, M).
        // First time a shape is out-of-envelope, run_i8 returns ORK_RC_WEDGE_PRONE (not the generic -1) — it
        // declined rather than risk a blocking-submit wedge — and we rescue on the all-doorbell sliced primitive
        // (verified c_base tiles) AND remember (w, M) as wedge-prone. Since that verdict is stable for the
        // session, every later call for the same (w, M) skips the doomed run_i8 attempt (and its refusal spam)
        // and goes straight to sliced. A shape that SUCCEEDS, or fails for any OTHER reason, never rescues and is
        // never memoized → pure no-op, zero cost for working shapes. Needs a pre-packed twin (ORK_SLICE_ROUTE on).
        ork_w_sliced * ws_memo = nullptr;
        if (ctx->slice_route && !ctx->slice_forced.empty()) {
            auto mit = ctx->slice_forced.find(tasks[0].w);
            if (mit != ctx->slice_forced.end() && mit->second.count(tasks[0].M)) {
                auto sit = ctx->slice_ws.find(tasks[0].w);
                if (sit != ctx->slice_ws.end()) ws_memo = sit->second;
            }
        }
        if (ws_memo) {                                   // memoized wedge-prone (w,M): skip run_i8, straight to sliced
            rc = ork_mm_run_sliced(ctx->npu, ws_memo, tasks[0].M, tasks[0].A, tasks[0].C, 0) ? -1 : 0;
        } else {
            const int r = ork_i8_mm_run(ctx->npu, tasks[0].w, tasks[0].M, tasks[0].A, tasks[0].C);
            rc = r ? -1 : 0;
            if (r == ORK_RC_WEDGE_PRONE && ctx->slice_route && !ctx->slice_ws.empty()) {
                auto sit = ctx->slice_ws.find(tasks[0].w);
                if (sit != ctx->slice_ws.end() && sit->second) {
                    rc = ork_mm_run_sliced(ctx->npu, sit->second, tasks[0].M, tasks[0].A, tasks[0].C, 0) ? -1 : 0;
                    if (rc == 0) ctx->slice_forced[tasks[0].w].insert(tasks[0].M);   // memoize ONLY a successful rescue
                    if (getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK] slice-rescue: run_i8 refused (wedge-prone) -> sliced doorbell rc=%d (memoized M=%d)\n", rc, tasks[0].M);
                }
            }
        }
    } else {
        bool same_dom = true; const int d0 = ork_w_domain(tasks[0].w);
        for (size_t t = 1; t < tasks.size(); t++) if (ork_w_domain(tasks[t].w) != d0) { same_dom = false; break; }
        rc = same_dom ? ork_i8_mm_run_stream(ctx->npu, tasks.size(), tasks.data())
                      : ork_i8_mm_run_chain (ctx->npu, tasks.size(), tasks.data());
        if (getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK] dispatch: %zu tasks, same_dom=%d -> %s rc=%d\n",
                                           tasks.size(), (int)same_dom, same_dom ? "run_stream" : "run_chain", rc);
    }
    if (rc != 0) {   // fallback: sequential single-task
        for (size_t t = 0; t < tasks.size(); t++)
            if (ork_i8_mm_run(ctx->npu, tasks[t].w, tasks[t].M, tasks[t].A, tasks[t].C)) return false;
    }
    return true;
}

// ===================== UNIFIED NPU SUBMIT ROUTER =====================
// One chokepoint for every NPU int8 matmul submit. The split's async/blocking choice — and (next) the
// SRAM-backed doorbell — lives HERE, not scattered across the handlers. Three primitives so the caller
// controls the overlap window:
//   ork_submit_async(): try the NON-BLOCKING multi-core doorbell (ork_dyn_begin_mc). Returns a live chain
//     the caller MUST ork_submit_end() AFTER its overlapping CPU work — or NULL if ineligible (M>1 / K not
//     %512 / K>4096 / C not resident / mixed domain), in which case the caller falls back to ork_submit_sync.
//   ork_submit_sync():  blocking dispatch (1->run_i8 / N same-domain->run_stream / N cross->run_chain).
//   ork_submit_end():   rendezvous + writeback + free the async chain. 0=ok, <0=error.
// Canonical split pattern:  h = ork_submit_async(...);  <CPU-complete work>;  rc = h ? ork_submit_end(h) : ork_submit_sync(...);
static inline ork_dyn_chain * ork_submit_async(ggml_backend_ork_context * ctx, std::vector<ork_mm_task_i8> & tasks) {
    return tasks.empty() ? nullptr : ork_dyn_begin_mc(ctx->npu, (int) tasks.size(), tasks.data(), 0 /*all cores*/);
}
static inline int ork_submit_sync(ggml_backend_ork_context * ctx, std::vector<ork_mm_task_i8> & tasks) {
    return ork_dispatch_i8(ctx, tasks) ? 0 : -1;
}
static inline int ork_submit_end(ork_dyn_chain * h) { return h ? (ork_dyn_end(h) < 0 ? -1 : 0) : 0; }

static bool ggml_backend_ork_mul_mat_i8(ggml_backend_ork_context * ctx, struct ggml_tensor * dst) {
    if(getenv("ORK_VERBOSE"))fprintf(stderr, "[ORK] START mul_mat_i8\n"); fflush(stderr);
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    if (getenv("ORK_BUFPROBE")) { static int once=0; if(!once++) fprintf(stderr,
        "[ork bufprobe] src1(act) buft=%s | dst(out) buft=%s\n",
        src1->buffer?ggml_backend_buffer_name(src1->buffer):"(none)",
        dst->buffer?ggml_backend_buffer_name(dst->buffer):"(none)"); }
    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type type = src0->type;
    const int K = (int) ne00, N = (int) ne01, M = (int) ne11;

    const int64_t r2 = ne02 > 0 ? ne12/ne02 : 1;        // broadcast factors (e.g. GQA / attention)
    const int64_t r3 = ne03 > 0 ? ne13/ne03 : 1;

    const auto * tt = ggml_get_type_traits(type);
    ggml_to_float_t const to_float = tt->to_float;

    const int S = (int)(ne12 * ne13);

    // Temp buffers for weight packing
    ctx->f32.resize((size_t) N * K);
    ctx->bi .resize((size_t) K * N);

    for (int chunk_start = 0; chunk_start < S; chunk_start += 1) {
        int chunk_size = std::min(1, S - chunk_start);

        const int M_padded = (M == 1) ? 1 : ((M + 31) / 32) * 32;

        ctx->ai .resize((size_t) chunk_size * M_padded * K);
        ctx->as .resize((size_t) chunk_size * M_padded);
        ctx->ci .resize((size_t) chunk_size * M_padded * N);

        int8_t  * ai  = ctx->ai.data();
        float   * as  = ctx->as.data();
        int32_t * ci  = ctx->ci.data();

        std::vector<ork_mm_task_i8> tasks;
        std::vector<ork_stream_entry *> task_se;   // STREAM-POOL: per-task RAM-resident entry (NULL = plain ow.w)

        const double t0 = ctx->profile ? ork_now_us() : 0;

        for (int t = 0; t < chunk_size; t++) {
            const int s = chunk_start + t;
            const int i13 = s / ne12;
            const int i12 = s % ne12;
            const int64_t i03 = r3 > 0 ? i13/r3 : 0;
            const int64_t i02 = r2 > 0 ? i12/r2 : 0;

            const char  * x = (const char *) src0->data + i02*nb02 + i03*nb03;
            const float * y = (const float *)((const char *) src1->data + i12*nb12 + i13*nb13);

            // weight: check cache / .orkpack / pack
            auto it = ork_resolve_weight_i8(ctx, src0, K, N, nb01, type, to_float, /*allow_evict=*/true);
            if (it == ctx->wcache.end()) return false;
            it->second.last_use = ++ctx->wcache_tick;   // LRU touch (hit or fresh pack)
            const ork_weight & ow = it->second;

            bool reuse = (y == ctx->last_src1 && M == ctx->last_M && K == ctx->last_K && ctx->last_type == 1 && !ctx->no_reuse);
            if (!reuse) {
                // activation: per-row int8 quant with shape padding
                int8_t * ar = ai + t * M_padded * K;
                float * asr = as + t * M_padded;
                const double _aq0 = ctx->profile ? ork_now_us() : 0;   // LEVER3: time pure act-quant arithmetic
                #pragma omp parallel for if (M_padded >= 16)
                for (int m = 0; m < M_padded; m++) {
                    if (m < M) {
                        const float * yr = y + (size_t) m*K;
                        int8_t * amr = ar + (size_t) m*K;
                        float mx = 1e-9f;
#if defined(__ARM_NEON)
                        // pass 1: vectorized abs-max reduction across K.
                        // float max is exact + order-independent, so the lane
                        // ordering does not change the result vs the scalar loop.
                        int k = 0;
                        float32x4_t v_mx0 = vdupq_n_f32(mx);
                        float32x4_t v_mx1 = vdupq_n_f32(mx);
                        for (; k <= K - 8; k += 8) {
                            v_mx0 = vmaxq_f32(v_mx0, vabsq_f32(vld1q_f32(yr + k)));
                            v_mx1 = vmaxq_f32(v_mx1, vabsq_f32(vld1q_f32(yr + k + 4)));
                        }
                        mx = vmaxvq_f32(vmaxq_f32(v_mx0, v_mx1));
                        for (; k < K; k++) { float v = fabsf(yr[k]); mx = v > mx ? v : mx; }
                        asr[m] = mx / 127.0f;
                        const float inv = 127.0f / mx;
                        // pass 2: vectorized quantize. Replicates the scalar
                        // round-half-away-from-zero EXACTLY: half = copysign(0.5,q)
                        // (sign bit of q OR'd into +0.5), q+half in float, then
                        // truncate toward zero (vcvtq_s32_f32), clamp to [-127,127].
                        const float32x4_t v_inv  = vdupq_n_f32(inv);
                        const float32x4_t v_half = vdupq_n_f32(0.5f);
                        const uint32x4_t  v_sign = vdupq_n_u32(0x80000000u);
                        const int32x4_t   v_hi   = vdupq_n_s32(127);
                        const int32x4_t   v_lo   = vdupq_n_s32(-127);
                        k = 0;
                        for (; k <= K - 8; k += 8) {
                            float32x4_t v_q0 = vmulq_f32(vld1q_f32(yr + k),     v_inv);
                            float32x4_t v_q1 = vmulq_f32(vld1q_f32(yr + k + 4), v_inv);
                            // copysignf(0.5f, q): OR the sign bit of q into 0.5
                            float32x4_t v_h0 = vreinterpretq_f32_u32(vorrq_u32(vandq_u32(vreinterpretq_u32_f32(v_q0), v_sign), vreinterpretq_u32_f32(v_half)));
                            float32x4_t v_h1 = vreinterpretq_f32_u32(vorrq_u32(vandq_u32(vreinterpretq_u32_f32(v_q1), v_sign), vreinterpretq_u32_f32(v_half)));
                            // q + half, then truncate toward zero (matches (int) cast)
                            int32x4_t v_i0 = vcvtq_s32_f32(vaddq_f32(v_q0, v_h0));
                            int32x4_t v_i1 = vcvtq_s32_f32(vaddq_f32(v_q1, v_h1));
                            // clamp to [-127,127] in int32, then narrow to int8
                            v_i0 = vmaxq_s32(vminq_s32(v_i0, v_hi), v_lo);
                            v_i1 = vmaxq_s32(vminq_s32(v_i1, v_hi), v_lo);
                            int16x8_t v_s16 = vcombine_s16(vmovn_s32(v_i0), vmovn_s32(v_i1));
                            vst1_s8(amr + k, vmovn_s16(v_s16));
                        }
                        for (; k < K; k++) {
                            float q = yr[k] * inv;
                            int qi = (int) (q + copysignf(0.5f, q));
                            amr[k] = (int8_t) (qi > 127 ? 127 : qi < -127 ? -127 : qi);
                        }
#else
                        for (int k = 0; k < K; k++) { float v = fabsf(yr[k]); mx = v > mx ? v : mx; }
                        asr[m] = mx / 127.0f;
                        const float inv = 127.0f / mx;
                        for (int k = 0; k < K; k++) {
                            float q = yr[k] * inv;
                            int qi = (int) (q + copysignf(0.5f, q));
                            amr[k] = (int8_t) (qi > 127 ? 127 : qi < -127 ? -127 : qi);
                        }
#endif
                    } else {
                        memset(ar + (size_t) m*K, 0, K);
                        asr[m] = 0.0f;
                    }
                }
                if (ctx->profile) { ctx->t_actq += ork_now_us() - _aq0; ctx->n_actq++; }   // LEVER3
                ctx->last_src1 = y;
                ctx->last_M = M;
                ctx->last_K = K;
                ctx->last_type = 1;
            } else {
                if(getenv("ORK_VERBOSE"))fprintf(stderr, "[ORK] i8: reuse activation cache for y=%p\n", y);
                fflush(stderr);
            }
            tasks.push_back({
                ow.w,
                M_padded,
                ai + t * M_padded * K,
                ci + t * M_padded * N
            });
            task_se.push_back(ow.se);
        }

        const double t1 = ctx->profile ? ork_now_us() : 0;

        if(getenv("ORK_VERBOSE"))fprintf(stderr, "[ORK] i8 chain: M=%d, tasks=%zu (S=%d, K=%d, N=%d)\n", M, tasks.size(), S, K, N);
        fflush(stderr);
        int ok = -1;
        if (ctx->spool && task_se[0]) {
            // STREAM-POOL: each weight is a RAM-resident entry mapped to IOVA; run per-task (correctness
            // first — the chain co-residency contract is met by all entries staying mapped here).
            ok = 0;
            for (size_t t = 0; t < tasks.size(); t++)
                if (ork_stream_pool_run(ctx->spool, task_se[t], tasks[t].M, tasks[t].A, tasks[t].C)) { ok = -1; break; }
            if (ok != 0) return false;
        } else {
            if (ork_submit_sync(ctx, tasks)) return false;   // unified router: 1→run_i8 · N same-dom→run_stream · N cross→run_chain
            ok = 0;
        }

        const double t2 = ctx->profile ? ork_now_us() : 0;

        // Dequantize results
        for (int t = 0; t < chunk_size; t++) {
            const int s = chunk_start + t;
            const int i13 = s / ne12;
            const int i12 = s % ne12;
            const int64_t i03 = r3 > 0 ? i13/r3 : 0;
            const int64_t i02 = r2 > 0 ? i12/r2 : 0;
            const char  * x = (const char *) src0->data + i02*nb02 + i03*nb03;
            float * d = (      float *)((      char *)  dst->data + i12*nb2  + i13*nb3);

            auto it = ctx->wcache.find(x);
            const ork_weight & ow = it->second;
            const float * bs = ow.bscale.data();
            const float * asr = as + t * M_padded;
            const int32_t * ctr = ci + t * M_padded * N;

            #pragma omp parallel for if (M >= 16)
            for (int m = 0; m < M; m++) {
                const float rs = asr[m];
                const int32_t * cr = ctr + (size_t) m*N;
                float * dr = d + (size_t) m*N;
#if defined(__ARM_NEON)
                float32x4_t v_rs = vdupq_n_f32(rs);
                int n = 0;
                for (; n <= N - 8; n += 8) {
                    int32x4_t v_cr0 = vld1q_s32(cr + n);
                    int32x4_t v_cr1 = vld1q_s32(cr + n + 4);
                    float32x4_t v_cr_f0 = vcvtq_f32_s32(v_cr0);
                    float32x4_t v_cr_f1 = vcvtq_f32_s32(v_cr1);
                    float32x4_t v_bs0 = vld1q_f32(bs + n);
                    float32x4_t v_bs1 = vld1q_f32(bs + n + 4);
                    float32x4_t v_prod0 = vmulq_f32(v_bs0, v_cr_f0);
                    float32x4_t v_prod1 = vmulq_f32(v_bs1, v_cr_f1);
                    float32x4_t v_dr0 = vmulq_f32(v_prod0, v_rs);
                    float32x4_t v_dr1 = vmulq_f32(v_prod1, v_rs);
                    vst1q_f32(dr + n, v_dr0);
                    vst1q_f32(dr + n + 4, v_dr1);
                }
                for (; n < N; n++) {
                    dr[n] = rs * bs[n] * (float) cr[n];
                }
#else
                for (int n = 0; n < N; n++) dr[n] = rs * bs[n] * (float) cr[n];
#endif
            }
        }

        if (ctx->profile) {
            double t3 = ork_now_us();
            ctx->t_quant += t1 - t0;
            ctx->t_run   += t2 - t1;
            ctx->t_deq   += t3 - t2;
            ctx->n_mm    += chunk_size;
            if (M > 1) {
                ctx->t_run_pf  += t2 - t1;
                ctx->n_pf      += chunk_size;
                ctx->m_pf      += chunk_size * M;
            } else {
                ctx->t_run_dec += t2 - t1;
                ctx->n_dec     += chunk_size;
            }
        }

        if (ctx->no_cache) {
            for (int t = 0; t < chunk_size; t++) {
                const int s = chunk_start + t;
                const int i13 = s / ne12;
                const int i12 = s % ne12;
                const int64_t i03 = r3 > 0 ? i13/r3 : 0;
                const int64_t i02 = r2 > 0 ? i12/r2 : 0;
                const char  * x = (const char *) src0->data + i02*nb02 + i03*nb03;
                auto it = ctx->wcache.find(x);
                if (it != ctx->wcache.end()) {
                    ork_w_free(it->second.w);
                    ctx->wcache.erase(it);
                }
            }
        }
    }

    return true;
}

// int4 (W4A4) with per-group scales. The NPU MAC is same-precision, so int4 weights require int4
// activations too — weights AND activations are per-group int4-quantized (group_size G along K),
// the NPU dequantizes each group's int partial in fp32. ~9.5% matmul error (W4A4 floor; weights at
// 0.5 B/elem). Submit-heavy (K/G submits/core), so coarser/larger G is cheaper but less accurate.
//
// NO LONGER REACHABLE FROM DISPATCH. This is the UN-ROTATED 4-bit route: per-group scales stand in for
// the block-Hadamard, and they do not stand in well enough (grouped is coherent but costs 16x the
// submits; per-channel-without-R is PPL ~104). Selecting native W4A4 now always takes the rotated
// mul_mat_i4_hadamard — see ork_w4a4_native_on. Kept, not deleted: it is the grouped-vs-rotated A/B
// baseline the int4 RE depends on, and re-deriving it costs far more than the dead code does.
__attribute__((unused))
static bool ggml_backend_ork_mul_mat_i4(ggml_backend_ork_context * ctx, struct ggml_tensor * dst) {
    if(getenv("ORK_VERBOSE"))fprintf(stderr, "[ORK] START mul_mat_i4\n"); fflush(stderr);
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type type = src0->type;
    const int K = (int) ne00, N = (int) ne01, M = (int) ne11;
    const int G = (K % 128 == 0) ? 128 : (K % 64 == 0) ? 64 : 32;   // largest std group dividing K
    const int NG = K / G;

    const int64_t r2 = ne12/ne02, r3 = ne13/ne03;
    const auto * tt = ggml_get_type_traits(type);
    ggml_to_float_t const to_float = tt->to_float;

    const int M_padded = (M == 1) ? 1 : ((M + 31) / 32) * 32;

    ctx->f32.resize((size_t) N * K);
    ctx->bi .resize((size_t) K * N);
    ctx->ai .resize((size_t) M_padded * K);
    ctx->as .resize((size_t) M_padded * NG);                 // per-row, per-group activation scale
    float  * f32 = ctx->f32.data();
    int8_t * bi  = ctx->bi.data();
    int8_t * ai  = ctx->ai.data();
    float  * as  = ctx->as.data();

    for (int64_t i13 = 0; i13 < ne13; i13++) {
        for (int64_t i12 = 0; i12 < ne12; i12++) {
            const char  * x = (const char *) src0->data + (i13/r3)*nb03 + (i12/r2)*nb02;
            const float * y = (const float *)((const char *) src1->data + i12*nb12 + i13*nb13);
                  float * d = (      float *)((      char *)  dst->data + i12*nb2  + i13*nb3);

            // W4A4 (grouped) profiling — mirrors the W8A8 path (t_quant = weight pack + activation
            // int4-quant; t_run = ork_i4_mm_run_grouped incl. the in-run per-group fp32 dequant), so
            // ORK_PROFILE shows the int4 pack-vs-run split directly (it was invisible before).
            double _t0 = ctx->profile ? ork_now_us() : 0.0;
            auto it = ctx->wcache.find(x);
            if (it == ctx->wcache.end()) {
                if (type == GGML_TYPE_F32) {
                    for (int64_t n = 0; n < N; n++) memcpy(f32 + n*K, x + n*nb01, (size_t) K*sizeof(float));
                } else {
                    for (int64_t n = 0; n < N; n++) to_float((const char *) x + n*nb01, f32 + n*K, K);
                }
                ork_weight ow; ow.gsize = G; ow.bscale.resize((size_t) NG * N);
                for (int g = 0; g < NG; g++)
                    for (int n = 0; n < N; n++) {
                        float mx = 1e-9f;
                        for (int j = 0; j < G; j++) { float v = fabsf(f32[(size_t) n*K + g*G + j]); if (v > mx) mx = v; }
                        float s = mx / 7.0f; ow.bscale[(size_t) g*N + n] = s;
                        for (int j = 0; j < G; j++) {
                            int q = (int) lrintf(f32[(size_t) n*K + g*G + j] / s);
                            bi[(size_t)(g*G + j)*N + n] = (int8_t) (q > 7 ? 7 : q < -8 ? -8 : q);
                        }
                    }
                ow.w = ork_i4_mm_pack_grouped(ctx->npu, K, N, bi, G);
                if (!ow.w) return false;
                it = ctx->wcache.emplace(x, std::move(ow)).first;
            }
            const ork_weight & ow = it->second;

            bool reuse = (y == ctx->last_src1 && M == ctx->last_M && K == ctx->last_K && ctx->last_type == 2 && !ctx->no_reuse);
            if (!reuse) {
                // activations: per-row, per-group int4 quant with shape padding
                #pragma omp parallel for if (M_padded >= 16)
                for (int m = 0; m < M_padded; m++) {
                    if (m < M) {
                        for (int g = 0; g < NG; g++) {
                            float mx = 1e-9f;
                            for (int j = 0; j < G; j++) { float v = fabsf(y[(size_t) m*K + g*G + j]); if (v > mx) mx = v; }
                            float s = mx / 7.0f; as[(size_t) m*NG + g] = s;
                            for (int j = 0; j < G; j++) {
                                int q = (int) lrintf(y[(size_t) m*K + g*G + j] / s);
                                ai[(size_t) m*K + g*G + j] = (int8_t) (q > 7 ? 7 : q < -8 ? -8 : q);
                            }
                        }
                    } else {
                        memset(ai + (size_t) m*K, 0, K);
                        for (int g = 0; g < NG; g++) {
                            as[(size_t) m*NG + g] = 0.0f;
                        }
                    }
                }
                ctx->last_src1 = y;
                ctx->last_M = M;
                ctx->last_K = K;
                ctx->last_type = 2;
            } else {
                if(getenv("ORK_VERBOSE"))fprintf(stderr, "[ORK] i4 grouped: reuse activation cache for y=%p\n", y);
                fflush(stderr);
            }

            // grouped run dequantizes per group into the fp32 dst directly (handling M != M_padded to prevent out-of-bounds write)
            if(getenv("ORK_VERBOSE"))fprintf(stderr, "[ORK] i4 grouped: M_padded=%d (M=%d), K=%d, N=%d, G=%d\n", M_padded, M, K, N, G);
            fflush(stderr);
            std::vector<float> tmp_d;
            float * d_ptr = d;
            if (M != M_padded) {
                tmp_d.resize((size_t) M_padded * N);
                d_ptr = tmp_d.data();
            }
            double _t1 = ctx->profile ? ork_now_us() : 0.0;
            if (ork_i4_mm_run_grouped(ctx->npu, ow.w, M_padded, ai, as, ow.bscale.data(), d_ptr)) return false;
            if (M != M_padded) {
                memcpy(d, d_ptr, (size_t) M * N * sizeof(float));
            }
            if (ctx->profile) {
                double _t2 = ork_now_us();
                ctx->t_quant += _t1 - _t0; ctx->t_run += _t2 - _t1; ctx->n_mm += 1;
                if (M > 1) { ctx->t_run_pf += _t2 - _t1; ctx->n_pf++; ctx->m_pf += M; }
                else       { ctx->t_run_dec += _t2 - _t1; ctx->n_dec++; }
                if (getenv("ORK_VERBOSE")) fprintf(stderr, "[ork i4prof] W4A4-grouped %s M=%d K=%d N=%d G=%d | quant %.1fms run %.1fms\n",
                    src0->name, M, K, N, G, (_t1-_t0)/1e3, (_t2-_t1)/1e3);
            }
        }
    }
    return true;
}


/* ORK_GPTQ (pack-time, native-W4A4 only). GPTQ needs a calibration Hessian H = A^T A over the SAME input
 * space the weights live in — under QuaRot that is the ROTATED space, so A here is the rotated activation
 * batch (A*R), matching the rotated weight columns. The weight is quantized on FIRST USE inside the forward
 * pass, so the calibration batch is simply whatever prompt the pack pass runs.
 *
 * RANK: H is K*K but a batch contributes only M samples, so rank(H) <= M. With M < K the Hessian is
 * rank-deficient, damping dominates the null space and GPTQ degenerates toward RTN there — not wrong, just
 * weak. Use a calibration prompt with M >= K for the full benefit; we warn when it is not.
 * Cost: O(M*K^2) here plus ork_i4_gptq's three O(K^3) factorisations — a heavy ONE-TIME pack step. */
static void ork_gptq_hessian(int M, int K, int b, const float * y, float * H) {
    memset(H, 0, (size_t)K*K*sizeof(float));
    std::vector<float> a((size_t)K);
    for (int m = 0; m < M; m++) {
        memcpy(a.data(), y + (size_t)m*K, (size_t)K*sizeof(float));
        for (int off = 0; off < K; off += b) ork_fwht_norm(a.data() + off, b);   // same rotation as the weights
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < K; i++) {
            const float ai = a[i];
            if (ai == 0.0f) continue;
            float * hr = H + (size_t)i*K;
            for (int j = 0; j < K; j++) hr[j] += ai * a[j];
        }
    }
}

/* ORK_W4A4_DIAG — split the W4A4 error into its WEIGHT and ACTIVATION halves.
 *
 * Everything invested in W4A4 quality so far (Hadamard, GPTQ) touches only W. A is still plain per-row
 * absmax/7, and in W4A4 the activation term is usually the DOMINANT one — so before paying for a
 * pack-format change to get per-group weight scales, measure which half actually owns the error.
 *
 * On a subsample (rows x channels, so it is cheap enough to run inline) computes the exact rotated
 * product and three perturbations of it, and reports relative Frobenius error:
 *     W-only : quantized W, exact A     — what GPTQ/grouping can improve
 *     A-only : exact W, quantized A     — what NOTHING currently improves
 *     both   : the real W4A4 product
 * Uses the SAME rules the runtime uses (rotate, absmax/7, clamp [-8,7]) so the numbers are the real
 * ones, not a model of them. Measured against RTN weights, which makes it a CONSERVATIVE read on A's
 * share: GPTQ shrinks the W term, so A's dominance can only grow from here. */
static void ork_w4a4_diag(const char * name, int M, int K, int N, int b,
                          const float * y, const float * Wrot, const std::vector<float> & ws) {
    const int MS = M < 32 ? M : 32;                       /* subsample: rows */
    const int NS = N < 128 ? N : 128;                     /*            output channels */
    if (MS < 1 || NS < 1) return;

    std::vector<float> A((size_t)MS*K), Aq((size_t)MS*K);
    for (int m = 0; m < MS; m++) {                        /* rotate + quantize A exactly as the runtime does */
        float * a = A.data() + (size_t)m*K;
        memcpy(a, y + (size_t)m*K, (size_t)K*sizeof(float));
        for (int off = 0; off < K; off += b) ork_fwht_norm(a + off, b);
        float mx = 1e-9f; for (int k = 0; k < K; k++) { float v = fabsf(a[k]); if (v > mx) mx = v; }
        const float s = mx / 7.0f;
        float * aq = Aq.data() + (size_t)m*K;
        for (int k = 0; k < K; k++) { int q = (int) lrintf(a[k]/s); q = q>7?7:(q<-8?-8:q); aq[k] = q*s; }
    }
    double e_ref = 0, e_w = 0, e_a = 0, e_b = 0;
    #pragma omp parallel for schedule(static) reduction(+:e_ref,e_w,e_a,e_b)
    for (int n = 0; n < NS; n++) {
        const float * w  = Wrot + (size_t)n*K;
        const float sw = ws[n];
        std::vector<float> wq((size_t)K);
        for (int k = 0; k < K; k++) { int q = (int) lrintf(w[k]/sw); q = q>7?7:(q<-8?-8:q); wq[k] = q*sw; }
        for (int m = 0; m < MS; m++) {
            const float * a  = A.data()  + (size_t)m*K;
            const float * aq = Aq.data() + (size_t)m*K;
            double r = 0, dw = 0, da = 0, db = 0;
            for (int k = 0; k < K; k++) { r += (double)a[k]*w[k]; dw += (double)a[k]*wq[k];
                                          da += (double)aq[k]*w[k]; db += (double)aq[k]*wq[k]; }
            e_ref += r*r; e_w += (dw-r)*(dw-r); e_a += (da-r)*(da-r); e_b += (db-r)*(db-r);
        }
    }
    if (e_ref <= 0) return;
    const double nrm = sqrt(e_ref);
    fprintf(stderr, "[W4A4-DIAG] %-28s K=%-5d N=%-6d | rel err  W-only %.4f  A-only %.4f  both %.4f  "
                    "| A/W = %.2fx\n", name, K, N,
            sqrt(e_w)/nrm, sqrt(e_a)/nrm, sqrt(e_b)/nrm, sqrt(e_w) > 0 ? sqrt(e_a)/sqrt(e_w) : 0.0);
}

/* ---- TWO-PHASE GPTQ CALIBRATION ----------------------------------------------------------------
 * GPTQ wants H accumulated over MANY calibration batches; ork-driver quantizes a weight on FIRST USE,
 * when exactly one batch has been seen. rank(H) <= samples, so single-shot calibration leaves H rank-4
 * (the convert pass's M) against a K of 1024..3584 and GPTQ collapses to round-to-nearest. Hence two
 * phases:
 *   PHASE 1 (calibrate) — every native-W4A4 weight-miss registers here and keeps accumulating
 *     H += (A*R)^T (A*R) on EVERY subsequent forward. The forward still needs a weight, so it uses the
 *     RTN codes as usual, and persist is SKIPPED so no RTN codes reach the .orkpack.
 *   PHASE 2 (finalize) — ggml_backend_ork_gptq_finalize() re-reads each source tensor, re-rotates it,
 *     runs ork_i4_gptq against the accumulated H, re-packs, and persists THAT.
 * The rotated weight is re-derived at finalize rather than stored: it is O(N*K) to recompute and
 * O(N*K) floats to keep, and the recompute is noise next to GPTQ's three O(K^3) factorisations.
 * H is the memory driver: K*K doubles per weight (103 MB at K=3584), freed as each weight finalizes. */
struct ork_gptq_cal {
    const ggml_tensor * src = nullptr;    // re-read + re-rotate at finalize
    int K = 0, N = 0, b = 0;              // b = Hadamard block (largest pow2 dividing K)
    long samples = 0;                     // rows accumulated; rank(H) <= samples
    std::vector<double> H;                // K*K
};
static std::unordered_map<const void *, ork_gptq_cal> g_gptq_cal;
static bool ork_gptq_on(void)  { static const int e = getenv("ORK_GPTQ") != nullptr; return e; }

/* accumulate H += (A*R)^T (A*R) for one batch of rotated activations.
 *
 * BLOCKED over rows, and that is the whole point. The obvious form — one rank-1 update per row — streams
 * the ENTIRE H in and out once per row: at K=3584 that is 98 MiB x 512 rows = 50 GiB of traffic per weight
 * per batch, and it made phase 1 memory-bound at ~4 min/batch. Accumulating B rows at a time turns it into
 * a rank-B update (a small GEMM), touching H once per BLOCK instead of once per row — B-fold less traffic.
 *
 * The block is held TRANSPOSED (AbT[i][r], row i's B samples contiguous) so the inner dot product over r is
 * unit-stride on both operands; the natural [r][i] layout would make it stride-K, which is the same cache
 * mistake one level down. AbT is B*K floats — 917 KiB at K=3584, B=64 — so it stays in L2. */
static void ork_gptq_accum(ork_gptq_cal & c, int M, const float * y) {
    const int K = c.K;
    const int B = 64;
    std::vector<float> AbT((size_t)K*B);
    std::vector<float> a((size_t)K);
    for (int m0 = 0; m0 < M; m0 += B) {
        const int nb = (M - m0 < B) ? (M - m0) : B;
        for (int r = 0; r < nb; r++) {                            /* rotate the block, store transposed */
            memcpy(a.data(), y + (size_t)(m0+r)*K, (size_t)K*sizeof(float));
            for (int off = 0; off < K; off += c.b) ork_fwht_norm(a.data() + off, c.b);
            for (int i = 0; i < K; i++) AbT[(size_t)i*B + r] = a[i];
        }
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < K; i++) {                             /* H[i][j] += <AbT[i], AbT[j]> over the block */
            const float * ri = AbT.data() + (size_t)i*B;
            double * hr = c.H.data() + (size_t)i*K;
            for (int j = 0; j <= i; j++) {                        /* symmetric: fill lower, mirror after */
                const float * rj = AbT.data() + (size_t)j*B;
                double acc = 0.0;
                for (int r = 0; r < nb; r++) acc += (double)ri[r] * (double)rj[r];
                hr[j] += acc;
            }
        }
    }
    c.samples += M;   /* only the LOWER triangle is accumulated; finalize mirrors it once (see below) */
}

// int4 (W4A4) with PER-CHANNEL scales + a block-Hadamard rotation (implied by the int4 tier). Weights are
// rotated (R·B) and per-channel int4-quantized once at load (cached); activations are rotated (A·R)
// and per-row int4-quantized each matmul; the rotation cancels in fp32 (A·B = (A·R)·(R·B)) but lets
// the coarse per-channel int4 quant stay accurate. Per-channel = full-K SINGLE submit (ork_i4_mm_run),
// not the grouped path's K/G submits. The NPU int MAC is exact; the only loss is the int4 quant the
// rotation tames. See ROADMAP Tier 4a/4b.
static bool ggml_backend_ork_mul_mat_i4_hadamard(ggml_backend_ork_context * ctx, struct ggml_tensor * dst) {
    if(getenv("ORK_VERBOSE"))fprintf(stderr, "[ORK] START mul_mat_i4_hadamard\n"); fflush(stderr);
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type type = src0->type;
    const int K = (int) ne00, N = (int) ne01, M = (int) ne11;
    const int b = K & (-K);                          // largest power-of-2 block dividing K (full FWHT if K is pow2)

    const int64_t r2 = ne12/ne02, r3 = ne13/ne03;
    const auto * tt = ggml_get_type_traits(type);
    ggml_to_float_t const to_float = tt->to_float;

    const int M_padded = (M == 1) ? 1 : ((M + 31) / 32) * 32;

    ctx->f32.resize((size_t) N * K);
    ctx->bi .resize((size_t) K * N);
    ctx->ai .resize((size_t) M_padded * K);
    ctx->as .resize((size_t) M_padded);
    ctx->ci .resize((size_t) M_padded * N);
    ctx->arot.resize((size_t) K);
    float   * f32  = ctx->f32.data();
    int8_t  * bi   = ctx->bi.data();
    int8_t  * ai   = ctx->ai.data();
    float   * as   = ctx->as.data();
    int32_t * ci   = ctx->ci.data();
    float   * arow = ctx->arot.data();

    for (int64_t i13 = 0; i13 < ne13; i13++) {
        for (int64_t i12 = 0; i12 < ne12; i12++) {
            const char  * x = (const char *) src0->data + (i13/r3)*nb03 + (i12/r2)*nb02;
            const float * y = (const float *)((const char *) src1->data + i12*nb12 + i13*nb13);
                  float * d = (      float *)((      char *)  dst->data + i12*nb2  + i13*nb3);

            // W4A4 (per-channel + Hadamard) profiling — t_quant = FWHT-rotate + weight/act int4-quant +
            // pack; t_run = ork_i4_mm_run (single full-K submit); t_deq = the per-channel fp32 scale-apply.
            double _t0 = ctx->profile ? ork_now_us() : 0.0;
            auto it = ctx->wcache.find(x);
            if (it == ctx->wcache.end()) {
                ork_weight ow;
                if (!ork_persist_load_i4native(ctx, src0->name, K, N, ow)) {   // .orkpack MISS -> cold rotate+quant+pack
                    ow.gsize = 0; ow.bscale.resize((size_t) N);   // per-channel scale ws[n]
                    // PARALLEL convert pack: dequant + FWHT-rotate + per-channel int4-quant, one column per
                    // OpenMP iteration (each n is independent — disjoint f32/bi/bscale). This is the per-weight
                    // one-time conversion cost; threading it over N cuts the user's wait ~ncore-fold.
                    #pragma omp parallel for schedule(static)
                    for (int n = 0; n < N; n++) {
                        float * col = f32 + (size_t) n*K;
                        if (type == GGML_TYPE_F32) memcpy(col, x + (size_t) n*nb01, (size_t) K*sizeof(float));
                        else                       to_float((const char *) x + (size_t) n*nb01, col, K);
                        for (int off = 0; off < K; off += b) ork_fwht_norm(col + off, b);   // rotate weight column R·B
                        float mx = 1e-9f;
                        for (int k = 0; k < K; k++) { float v = fabsf(col[k]); if (v > mx) mx = v; }
                        float s = mx / 7.0f; ow.bscale[n] = s;
                        for (int k = 0; k < K; k++) {
                            int q = (int) lrintf(col[k] / s);
                            bi[(size_t) k*N + n] = (int8_t) (q > 7 ? 7 : q < -8 ? -8 : q);
                        }
                    }
                    /* ORK_GPTQ PHASE 1: register this weight for calibration and DO NOT persist — the
                     * RTN codes below are only so the calibration forwards have something to compute with.
                     * ggml_backend_ork_gptq_finalize() re-quantizes with the accumulated H and persists that. */
                    ow.w = ork_i4_mm_pack(ctx->npu, K, N, bi);
                    if (!ow.w) return false;
                    if (getenv("ORK_W4A4_DIAG")) ork_w4a4_diag(src0->name, M, K, N, b, y, f32, ow.bscale);
                    if (ork_gptq_on()) {
                        ork_gptq_cal & c = g_gptq_cal[x];
                        if (c.H.empty()) {
                            c.src = src0; c.K = K; c.N = N; c.b = b;
                            c.H.assign((size_t)K*K, 0.0);
                            fprintf(stderr, "[ORK GPTQ] calibrating %s K=%d N=%d (H = %.0f MiB)\n",
                                    src0->name, K, N, (double)K*K*8/1048576.0);
                        }
                    } else {
                        ork_persist_write_i4native(ctx, src0->name, K, N, ow);   // convert: persist the rotated+tiled bytes
                    }
                }
                it = ctx->wcache.emplace(x, std::move(ow)).first;
            }
            const ork_weight & ow = it->second;
            if (ork_gptq_on()) {                                   /* PHASE 1: keep accumulating H every forward */
                auto ci = g_gptq_cal.find(x);
                if (ci != g_gptq_cal.end()) ork_gptq_accum(ci->second, M, y);
            }
            double _tw = ctx->profile ? ork_now_us() : 0.0;   /* split: weight-handling (_t0.._tw) vs act-quant (_tw.._t1) */

            bool reuse = (y == ctx->last_src1 && M == ctx->last_M && K == ctx->last_K && ctx->last_type == 3 && !ctx->no_reuse);
            if (!reuse) {
                // activations: rotate each row (A·R), per-row int4 quant with shape padding
                #pragma omp parallel for if (M_padded >= 16)
                for (int m = 0; m < M_padded; m++) {
                    if (m < M) {
                        float arow_local[K];
                        memcpy(arow_local, y + (size_t) m*K, (size_t) K*sizeof(float));
                        for (int off = 0; off < K; off += b) {
                            ork_fwht_norm(arow_local + off, b);
                        }
                        float mx = 1e-9f;
                        for (int k = 0; k < K; k++) { float v = fabsf(arow_local[k]); if (v > mx) mx = v; }
                        float s = mx / 7.0f; as[m] = s;
                        for (int k = 0; k < K; k++) {
                            int q = (int) lrintf(arow_local[k] / s);
                            ai[(size_t) m*K + k] = (int8_t) (q > 7 ? 7 : q < -8 ? -8 : q);
                        }
                    } else {
                        memset(ai + (size_t) m*K, 0, K);
                        as[m] = 0.0f;
                    }
                }
                ctx->last_src1 = y;
                ctx->last_M = M;
                ctx->last_K = K;
                ctx->last_type = 3;
            } else {
                if(getenv("ORK_VERBOSE"))fprintf(stderr, "[ORK] i4 hadamard: reuse activation cache for y=%p\n", y);
                fflush(stderr);
            }

            ork_mm_task_i4 task = { ow.w, M_padded, ai, ci };
            if(getenv("ORK_VERBOSE"))fprintf(stderr, "[ORK] i4 chain hadamard: M_padded=%d (M=%d), K=%d, N=%d\n", M_padded, M, K, N);
            fflush(stderr);
            double _t1 = ctx->profile ? ork_now_us() : 0.0;
            if (ork_i4_mm_run(ctx->npu, task.w, task.M, task.A, task.C)) return false;    // full-K single submit, int32 C
            double _t2 = ctx->profile ? ork_now_us() : 0.0;
            #pragma omp parallel for if (M >= 16)
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    d[(size_t) m*N + n] = (float) ci[(size_t) m*N + n] * as[m] * ow.bscale[n];
                }
            }
            if (ctx->profile) {
                double _t3 = ork_now_us();
                ctx->t_quant += _t1 - _t0; ctx->t_run += _t2 - _t1; ctx->t_deq += _t3 - _t2; ctx->n_mm += 1;
                if (M > 1) { ctx->t_run_pf += _t2 - _t1; ctx->n_pf++; ctx->m_pf += M; }
                else       { ctx->t_run_dec += _t2 - _t1; ctx->n_dec++; }
                if (getenv("ORK_VERBOSE")) fprintf(stderr, "[ork i4prof] W4A4-hadamard %s M=%d K=%d N=%d | wload %.1fms actq %.1fms run %.1fms deq %.1fms\n",
                    src0->name, M, K, N, (_tw-_t0)/1e3, (_t1-_tw)/1e3, (_t2-_t1)/1e3, (_t3-_t2)/1e3);
            }
        }
    }
    return true;
}

/* ORK_GPTQ PHASE 2. Called once after the calibration forwards (ork_bench does it). For every weight
 * registered in phase 1: re-read + re-rotate the source tensor, run GPTQ against the H accumulated over
 * ALL calibration batches, re-pack, swap the wcache entry, persist THAT, and free H.
 *
 * Re-deriving the rotated weight here (rather than stashing it in phase 1) trades O(N*K) recompute for
 * O(N*K) floats per weight held across the whole calibration — the recompute is noise beside GPTQ's three
 * O(K^3) factorisations, and it keeps peak memory to the H accumulators alone.
 *
 * Weights are finalized largest-K first so the biggest H is freed earliest, and the running log makes the
 * K^3 cost visible while it happens rather than after. */
/* How many calibration ROWS this model actually needs: the largest K among the registered weights.
 * GPTQ's error feedback is weighted by H^-1, and rank(H) <= rows — so below K rows the damping term owns
 * the null space, H^-1 ~ I/lambda there, the propagation row goes to zero and GPTQ SILENTLY degenerates to
 * round-to-nearest in exactly those directions. That threshold is a property of the weight shapes, not a
 * preference, so the caller derives the batch count from this instead of being told a number. Returns 0
 * when nothing is registered (not a GPTQ run). */
extern "C" int ggml_backend_ork_gptq_min_rows(void) {
    int mx = 0;
    for (const auto & kv : g_gptq_cal) if (kv.second.K > mx) mx = kv.second.K;
    return mx;
}

/* Rows accumulated so far (they are identical across weights — every weight sees every batch). */
extern "C" long ggml_backend_ork_gptq_rows(void) {
    return g_gptq_cal.empty() ? 0 : g_gptq_cal.begin()->second.samples;
}

extern "C" void ggml_backend_ork_gptq_finalize(void) {
    if (!ork_gptq_on() || g_gptq_cal.empty()) return;
    ggml_backend_ork_context * ctx = g_ork_ctx;
    if (!ctx) { fprintf(stderr, "[ORK GPTQ] finalize: no backend context\n"); return; }

    const float damp = getenv("ORK_GPTQ_DAMP") ? (float) atof(getenv("ORK_GPTQ_DAMP")) : 0.01f;
    std::vector<const void *> keys;
    keys.reserve(g_gptq_cal.size());
    for (auto & kv : g_gptq_cal) keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end(), [](const void * a, const void * b){
        return g_gptq_cal[a].K > g_gptq_cal[b].K; });          // biggest H freed first

    fprintf(stderr, "[ORK GPTQ] finalize: %zu weights, damp=%.3g\n", keys.size(), damp);
    double all0 = ork_now_us(); size_t done = 0, failed = 0;

    /* CHUNKED, parallel over WEIGHTS. ork_i4_gptq is internally OpenMP but its inner loops are short at
     * small K, so the cores idle; weights are completely independent, so the outer loop is where the
     * parallelism actually is. OpenMP nesting is off by default, which is exactly right here — each weight
     * gets one thread and the inner regions run serially.
     * Two-stage per chunk because ork_i4_mm_pack touches the NPU and is NOT thread-safe: compute all the
     * codes in parallel, then pack + persist SERIALLY. Chunked rather than one big pass so peak memory stays
     * bounded (holding every weight's codes at once would be the whole model in int8). */
    const int CH = 4;
    for (size_t base = 0; base < keys.size(); base += CH) {
        const size_t n_ch = (keys.size() - base < (size_t)CH) ? keys.size() - base : (size_t)CH;
        std::vector<std::vector<int8_t>> codes(n_ch);
        std::vector<std::vector<float>>  scal(n_ch);
        std::vector<int>                 rcs(n_ch, -1);
        std::vector<double>              secs(n_ch, 0.0);

        #pragma omp parallel for schedule(dynamic, 1) num_threads(CH)
        for (int u = 0; u < (int)n_ch; u++) {
            ork_gptq_cal & c = g_gptq_cal[keys[base + u]];
            const ggml_tensor * src = c.src;
            const int K = c.K, N = c.N, b = c.b;
            double t0 = ork_now_us();

            std::vector<float> W((size_t)N*K);                    /* re-read + re-rotate, as phase 1 did */
            const auto * tt = ggml_get_type_traits(src->type);
            ggml_to_float_t const to_float = tt->to_float;
            const char * x = (const char *) src->data;
            for (int n = 0; n < N; n++) {
                float * col = W.data() + (size_t)n*K;
                if (src->type == GGML_TYPE_F32) memcpy(col, x + (size_t)n*src->nb[1], (size_t)K*sizeof(float));
                else                            to_float(x + (size_t)n*src->nb[1], col, K);
                for (int off = 0; off < K; off += b) ork_fwht_norm(col + off, b);
            }
            /* accum filled only the LOWER triangle (halving its inner loop); mirror once, here. */
            std::vector<float> Hf((size_t)K*K);
            for (int i = 0; i < K; i++) {
                const double * lo = c.H.data() + (size_t)i*K;
                for (int j = 0; j <= i; j++) { const float v = (float) lo[j];
                    Hf[(size_t)i*K + j] = v; Hf[(size_t)j*K + i] = v; }
            }
            std::vector<double>().swap(c.H);                      /* free the accumulator as soon as it is copied */
            codes[u].resize((size_t)N*K); scal[u].resize((size_t)N);
            rcs[u]  = ork_i4_gptq(K, N, W.data(), Hf.data(), -1, codes[u].data(), scal[u].data(), damp);
            secs[u] = (ork_now_us() - t0) / 1e6;
        }

        for (size_t u = 0; u < n_ch; u++) {                       /* SERIAL: the NPU pack is not thread-safe */
            ork_gptq_cal & c = g_gptq_cal[keys[base + u]];
            const ggml_tensor * src = c.src;
            const int K = c.K, N = c.N;
            if (c.samples < K)
                fprintf(stderr, "[ORK GPTQ] %s: %ld calibration rows < K=%d — H rank-deficient; GPTQ tends to "
                                "RTN in the null space\n", src->name, c.samples, K);
            if (rcs[u]) { fprintf(stderr, "[ORK GPTQ] %s: ork_i4_gptq rc=%d — leaving RTN codes\n", src->name, rcs[u]);
                          failed++; continue; }
            auto it = ctx->wcache.find(keys[base + u]);
            if (it == ctx->wcache.end()) { fprintf(stderr, "[ORK GPTQ] %s: wcache entry vanished\n", src->name); failed++; continue; }
            ork_weight & ow = it->second;
            std::vector<int8_t> bi((size_t)K*N);
            for (int n = 0; n < N; n++) {
                ow.bscale[n] = scal[u][n];
                for (int k = 0; k < K; k++) bi[(size_t)k*N + n] = codes[u][(size_t)n*K + k];
            }
            if (ow.w) { ork_mm_free(ctx->npu, ow.w); ow.w = nullptr; }
            ow.w = ork_i4_mm_pack(ctx->npu, K, N, bi.data());
            if (!ow.w) { fprintf(stderr, "[ORK GPTQ] %s: repack FAILED\n", src->name); failed++; continue; }
            ork_persist_write_i4native(ctx, src->name, K, N, ow);
            done++;
            fprintf(stderr, "[ORK GPTQ] %s K=%d N=%d rows=%ld -> GPTQ (%.1f s)  [%zu/%zu, %.1f min]\n",
                    src->name, K, N, c.samples, secs[u], done, keys.size(), (ork_now_us()-all0)/6e7);
        }
    }
    g_gptq_cal.clear();
    fprintf(stderr, "[ORK GPTQ] finalize done: %zu quantized, %zu failed, %.1f min\n",
            done, failed, (ork_now_us()-all0)/6e7);
}

// Fused int8 matmul for a group of independent MUL_MATs that share the SAME src1 input (Q/K/V
// projections off the normed hidden state; FFN gate/up off the same x). Concatenates their weights
// along N into one packed weight, quantizes the shared activation ONCE, runs ONE NPU matmul, then
// scatters the wide int32 result into each dst — turning n submits into 1, amortizing the per-matmul
// submit floor. All g[i] are 2D (ne2==ne3==1), same K and M. Weight cached by g[0]->src0->data.
static bool ggml_backend_ork_mul_mat_group_i8(ggml_backend_ork_context * ctx, struct ggml_tensor ** g, int ng) {
    if(getenv("ORK_VERBOSE"))fprintf(stderr, "[ORK] START mul_mat_group_i8 ng=%d\n", ng); fflush(stderr);
    const struct ggml_tensor * src1 = g[0]->src[1];
    const int K = (int) g[0]->src[0]->ne[0];
    const int M = (int) src1->ne[1];
    int Ntot = 0, off[16];
    for (int i = 0; i < ng; i++) { off[i] = Ntot; Ntot += (int) g[i]->src[0]->ne[1]; }

    const void * key = g[0]->src[0]->data;
    auto it = ctx->wcache.find(key);
    if (it == ctx->wcache.end()) {                       // load-or-(build+pack)+persist the fused weight once
        ork_weight ow; ow.bscale.resize(Ntot);
        // Synthetic name for the FUSED (concatenated) group weight — stable across runs (grouping is
        // deterministic from the graph), distinct from the individual member names. This makes the fused
        // weight a FIRST-CLASS .orkpack entry: it is PERSISTED on the write pass and LOADED back on read, so
        // (a) a generated pack is COMPLETE (grouped gate/up are no longer silently dropped — the 3.8 GiB vs
        // 11 GiB gap) and (b) the auto-sizer BUDGETS it (it's now in persist_idx), fixing the multi-domain
        // OOM where the fused weight was re-JIT-packed at read into domains no one reserved.
        char gname[256];
        snprintf(gname, sizeof gname, "%s#grp%dx%d", g[0]->src[0]->name, ng, Ntot);
        int _dom = 0;
        bool loaded = false;
        if (ctx->persist_mode == 1) {                    // READ: load the fused weight from the .orkpack if present
            auto pit = ctx->persist_idx.find(gname);
            if (pit != ctx->persist_idx.end() && pit->second.K == (uint32_t) K &&
                pit->second.N == (uint32_t) Ntot && pit->second.dtype == ORKPACK_DT_I8) {
                const orkpack_entry & e = pit->second;
                const char * blob = (const char *) ctx->persist_map + e.blob_off;
                _dom = ork_weight_domain(ctx, (size_t) K * Ntot, ork_layer_of(g[0]->src[0]->name));
                ork_npu_set_pack_domain(ctx->npu, _dom);
                for (;;) {                               // import (zero-copy view of the mmap'd pack, or orkd dma-buf), spill on IOVA overflow
                    if (ctx->via_orkd) { size_t blob_n = (size_t) e.blob_size + e.bf_size;
                        ow.w = ork_i8_mm_import(ctx->npu, K, Ntot, blob, blob_n, e.bf_size ? (size_t) e.blob_size : 0); }
                    else { ow.w = ork_i8_mm_load_import(ctx->npu, K, Ntot, blob, e.blob_size);
                           if (!ow.w) ow.w = ork_i8_mm_load(ctx->npu, K, Ntot, blob, e.blob_size); }
                    if (ow.w || (_dom = ork_domain_advance(ctx)) < 0) break;
                }
                if (ow.w) { const float * bs = (const float *) ((const char *) ctx->persist_map + e.bscale_off);
                            ow.bscale.assign(bs, bs + e.bscale_n); loaded = true; }
            }
        }
        if (!loaded) {                                   // build the fused int8 weight from the members, pack, then persist
            ctx->bi.resize((size_t) K * Ntot); int8_t * bi = ctx->bi.data();
            for (int i = 0; i < ng; i++) {
                const struct ggml_tensor * w = g[i]->src[0];
                const int Ni = (int) w->ne[1];
                const auto * tt = ggml_get_type_traits(w->type); ggml_to_float_t to_float = tt->to_float;
                ctx->f32.resize((size_t) Ni * K); float * f32 = ctx->f32.data();
                const char * x = (const char *) w->data;
                if (w->type == GGML_TYPE_F32) for (int n = 0; n < Ni; n++) memcpy(f32 + (size_t) n*K, x + (size_t) n*w->nb[1], (size_t) K*sizeof(float));
                else                          for (int n = 0; n < Ni; n++) to_float(x + (size_t) n*w->nb[1], f32 + (size_t) n*K, K);
                for (int n = 0; n < Ni; n++) {
                    float mx = 1e-9f;
                    for (int k = 0; k < K; k++) { float v = fabsf(f32[(size_t) n*K + k]); if (v > mx) mx = v; }
                    float s = mx / 127.0f; ow.bscale[off[i]+n] = s;
                    for (int k = 0; k < K; k++) {        // fused B[k][off+n] = src0_i[n][k]
                        int q = (int) lrintf(f32[(size_t) n*K + k] / s);
                        bi[(size_t) k*Ntot + off[i]+n] = (int8_t) (q > 127 ? 127 : q < -127 ? -127 : q);
                    }
                }
            }
            // Byte-balanced domain first, spill to the next on IOVA overflow (same as every other pack site).
            _dom = ork_weight_domain(ctx, (size_t) K * Ntot, ork_layer_of(g[0]->src[0]->name));
            ork_npu_set_pack_domain(ctx->npu, _dom);
            ow.w = ork_i8_mm_pack(ctx->npu, K, Ntot, bi);
            while (!ow.w && (_dom = ork_domain_advance(ctx)) >= 0) ow.w = ork_i8_mm_pack(ctx->npu, K, Ntot, bi);
            if (!ow.w) return false;
            ork_persist_write(ctx, gname, K, Ntot, ow, nullptr, g[0]->src[0]->type, bi);   // .orkpack: persist the fused weight (int8 tier) so read-back loads it — complete + budgeted pack
        }
        ow.bytes = ork_w_bytes(ow.w); ctx->wcache_bytes += ow.bytes;
        if (ctx->n_domains > 1 && _dom < 64) ctx->domain_bytes[_dom] += ow.bytes;
        it = ctx->wcache.emplace(key, std::move(ow)).first;
    }
    const ork_weight & ow = it->second;

    const int M_padded = (M == 1) ? 1 : ((M + 31) / 32) * 32;
    ctx->ai.resize((size_t) M_padded*K); ctx->as.resize(M_padded); ctx->ci.resize((size_t) M_padded*Ntot);
    ctx->last_src1 = nullptr;                            // group overwrote ctx->ai — kill reuse cache
    ctx->last_type = 0;
    int8_t * ai = ctx->ai.data(); float * as = ctx->as.data(); int32_t * ci = ctx->ci.data();
    const float * y = (const float *) src1->data;
    for (int m = 0; m < M_padded; m++) {                        // quantize the shared activation once with shape padding
        if (m < M) {
            const float * yr = y + (size_t) m*K; int8_t * ar = ai + (size_t) m*K;
            float mx = 1e-9f;
            for (int k = 0; k < K; k++) { float v = fabsf(yr[k]); mx = v > mx ? v : mx; }
            as[m] = mx / 127.0f; const float inv = 127.0f / mx;
            for (int k = 0; k < K; k++) { float q = yr[k]*inv; int qi = (int)(q + copysignf(0.5f, q));
                ar[k] = (int8_t)(qi > 127 ? 127 : qi < -127 ? -127 : qi); }
        } else {
            memset(ai + (size_t) m*K, 0, K);
            as[m] = 0.0f;
        }
    }
    const double t1 = ctx->profile ? ork_now_us() : 0;
    if(getenv("ORK_VERBOSE"))fprintf(stderr, "[ORK] mul_mat_id i8: M_padded=%d (M=%d), K=%d, N=%d (ng=%d)\n", M_padded, M, K, Ntot, ng);
    fflush(stderr);
    if (ork_i8_mm_run(ctx->npu, ow.w, M_padded, ai, ci)) return false;     // ONE submit for all ng matmuls
    const double t2 = ctx->profile ? ork_now_us() : 0;

    const float * bs = ow.bscale.data();                 // scatter+dequant into each dst
    for (int i = 0; i < ng; i++) {
        const int Ni = (int) g[i]->src[0]->ne[1]; const int o = off[i];
        char * dbase = (char *) g[i]->data;
        const size_t drow = g[i]->nb[1];   // actual dst row stride (bytes) — q/k/v outputs may be strided, not m*Ni
        for (int m = 0; m < M; m++) {
            const float rs = as[m];
            const int32_t * cr = ci + (size_t) m*Ntot + o;
            float * dr = (float *) (dbase + (size_t) m*drow);
#if defined(__ARM_NEON)
            float32x4_t v_rs = vdupq_n_f32(rs);
            int n = 0;
            for (; n <= Ni - 8; n += 8) {
                int32x4_t v_cr0 = vld1q_s32(cr + n);
                int32x4_t v_cr1 = vld1q_s32(cr + n + 4);
                float32x4_t v_cr_f0 = vcvtq_f32_s32(v_cr0);
                float32x4_t v_cr_f1 = vcvtq_f32_s32(v_cr1);
                float32x4_t v_bs0 = vld1q_f32(bs + o + n);
                float32x4_t v_bs1 = vld1q_f32(bs + o + n + 4);
                float32x4_t v_prod0 = vmulq_f32(v_bs0, v_cr_f0);
                float32x4_t v_prod1 = vmulq_f32(v_bs1, v_cr_f1);
                float32x4_t v_dr0 = vmulq_f32(v_prod0, v_rs);
                float32x4_t v_dr1 = vmulq_f32(v_prod1, v_rs);
                vst1q_f32(dr + n, v_dr0);
                vst1q_f32(dr + n + 4, v_dr1);
            }
            for (; n < Ni; n++) {
                dr[n] = rs * bs[o + n] * (float) cr[n];
            }
#else
            for (int n = 0; n < Ni; n++) dr[n] = rs * bs[o+n] * (float) cr[n];
#endif
        }
    }
    if (ctx->profile) { ctx->t_run += t2-t1; ctx->n_mm++;
        if (M > 1) { ctx->t_run_pf += t2-t1; ctx->n_pf++; ctx->m_pf += M; } else { ctx->t_run_dec += t2-t1; ctx->n_dec++; } }
    return true;
}

// W4A4 group fusion — the int4 twin of mul_mat_group_i8. Concatenates ng independent same-input weights
// along N into ONE packed int4 weight (per-channel scale + block-Hadamard rotation, as mul_mat_i4_hadamard),
// quantizes AND rotates the shared activation ONCE (the extra win over i8: q/k/v redundantly re-rotate the
// same input in the per-node path), runs ONE W4A4 submit, scatters per-channel-dequant into each dst.
static bool ggml_backend_ork_mul_mat_group_i4(ggml_backend_ork_context * ctx, struct ggml_tensor ** g, int ng) {
    if(getenv("ORK_VERBOSE"))fprintf(stderr, "[ORK] START mul_mat_group_i4 ng=%d\n", ng); fflush(stderr);
    const struct ggml_tensor * src1 = g[0]->src[1];
    const int K = (int) g[0]->src[0]->ne[0];
    const int M = (int) src1->ne[1];
    const int b = K & (-K);                              // largest pow2 block dividing K (FWHT block)
    int Ntot = 0, off[16];
    for (int i = 0; i < ng; i++) { off[i] = Ntot; Ntot += (int) g[i]->src[0]->ne[1]; }

    const void * key = g[0]->src[0]->data;
    auto it = ctx->wcache.find(key);
    if (it == ctx->wcache.end()) {                       // build + pack the fused int4 weight once (rotated, per-channel)
        ork_weight ow; ow.gsize = 0; ow.bscale.resize(Ntot);
        ctx->bi.resize((size_t) K * Ntot); int8_t * bi = ctx->bi.data();
        for (int i = 0; i < ng; i++) {
            const struct ggml_tensor * w = g[i]->src[0]; const int Ni = (int) w->ne[1];
            const auto * tt = ggml_get_type_traits(w->type); ggml_to_float_t to_float = tt->to_float;
            ctx->f32.resize((size_t) Ni * K); float * f32 = ctx->f32.data();
            const char * x = (const char *) w->data;
            if (w->type == GGML_TYPE_F32) for (int n = 0; n < Ni; n++) memcpy(f32 + (size_t) n*K, x + (size_t) n*w->nb[1], (size_t) K*sizeof(float));
            else                          for (int n = 0; n < Ni; n++) to_float(x + (size_t) n*w->nb[1], f32 + (size_t) n*K, K);
            for (int n = 0; n < Ni; n++) {
                float * col = f32 + (size_t) n*K;
                for (int o2 = 0; o2 < K; o2 += b) ork_fwht_norm(col + o2, b);   // rotate weight column R·B
                float mx = 1e-9f; for (int k = 0; k < K; k++) { float v = fabsf(col[k]); if (v > mx) mx = v; }
                float s = mx / 7.0f; ow.bscale[off[i]+n] = s;
                for (int k = 0; k < K; k++) { int q = (int) lrintf(col[k] / s);
                    bi[(size_t) k*Ntot + off[i]+n] = (int8_t) (q > 7 ? 7 : q < -8 ? -8 : q); }
            }
        }
        // Multi-domain placement + spill (see mul_mat_group_i8): pick a byte-balanced domain, spill on overflow.
        int _dom = ork_weight_domain(ctx, (size_t) K * Ntot / 2, ork_layer_of(g[0]->src[0]->name));   // int4 nibble tile
        ork_npu_set_pack_domain(ctx->npu, _dom);
        ow.w = ork_i4_mm_pack(ctx->npu, K, Ntot, bi);
        while (!ow.w && (_dom = ork_domain_advance(ctx)) >= 0) ow.w = ork_i4_mm_pack(ctx->npu, K, Ntot, bi);
        if (!ow.w) return false;
        ow.bytes = ork_w_bytes(ow.w); ctx->wcache_bytes += ow.bytes;
        if (ctx->n_domains > 1 && _dom < 64) ctx->domain_bytes[_dom] += ow.bytes;
        it = ctx->wcache.emplace(key, std::move(ow)).first;
    }
    const ork_weight & ow = it->second;

    const int M_padded = (M == 1) ? 1 : ((M + 31) / 32) * 32;
    ctx->ai.resize((size_t) M_padded*K); ctx->as.resize(M_padded); ctx->ci.resize((size_t) M_padded*Ntot);
    ctx->last_src1 = nullptr; ctx->last_type = 0;        // group overwrote ctx->ai — kill reuse cache
    int8_t * ai = ctx->ai.data(); float * as = ctx->as.data(); int32_t * ci = ctx->ci.data();
    const float * y = (const float *) src1->data;
    #pragma omp parallel for if (M_padded >= 16)
    for (int m = 0; m < M_padded; m++) {                 // rotate + int4-quant the shared activation ONCE
        if (m < M) {
            float arow[K]; memcpy(arow, y + (size_t) m*K, (size_t) K*sizeof(float));
            for (int o2 = 0; o2 < K; o2 += b) ork_fwht_norm(arow + o2, b);
            float mx = 1e-9f; for (int k = 0; k < K; k++) { float v = fabsf(arow[k]); if (v > mx) mx = v; }
            float s = mx / 7.0f; as[m] = s;
            for (int k = 0; k < K; k++) { int q = (int) lrintf(arow[k] / s);
                ai[(size_t) m*K + k] = (int8_t) (q > 7 ? 7 : q < -8 ? -8 : q); }
        } else { memset(ai + (size_t) m*K, 0, K); as[m] = 0.0f; }
    }
    const double t1 = ctx->profile ? ork_now_us() : 0;
    if (ork_i4_mm_run(ctx->npu, ow.w, M_padded, ai, ci)) return false;    // ONE W4A4 submit for all ng
    const double t2 = ctx->profile ? ork_now_us() : 0;

    const float * bs = ow.bscale.data();
    for (int i = 0; i < ng; i++) {
        const int Ni = (int) g[i]->src[0]->ne[1]; const int o = off[i];
        float * dbase = (float *) g[i]->data;
        #pragma omp parallel for if (M >= 16)
        for (int m = 0; m < M; m++) {
            const float rs = as[m]; const int32_t * cr = ci + (size_t) m*Ntot + o; float * dr = dbase + (size_t) m*Ni;
            for (int n = 0; n < Ni; n++) dr[n] = rs * bs[o+n] * (float) cr[n];
        }
    }
    if (ctx->profile) { ctx->t_run += t2-t1; ctx->n_mm++;
        if (M > 1) { ctx->t_run_pf += t2-t1; ctx->n_pf++; ctx->m_pf += M; } else { ctx->t_run_dec += t2-t1; ctx->n_dec++; } }
    return true;
}

// backend interface

static const char * ggml_backend_ork_get_name(ggml_backend_t backend) { return "ORK"; GGML_UNUSED(backend); }

// LEVER3: profile dump, callable from backend free OR atexit (llama-bench never frees the backend).
static int g_ork_prof_dumped = 0;
static void ork_profile_dump(ggml_backend_ork_context * ctx) {
    if (!ctx || g_ork_prof_dumped) return;
    if (ctx->profile) fprintf(stderr, "[ork DUMP] profile=%d n_mm=%ld t_quant=%.0fms t_run=%.0fms t_deq=%.0fms t_actq=%.0fms\n",
                              ctx->profile, ctx->n_mm, ctx->t_quant/1e3, ctx->t_run/1e3, ctx->t_deq/1e3, ctx->t_actq/1e3);
    if (ctx->profile && ctx->n_mm) {
        g_ork_prof_dumped = 1;
        double tot = ctx->t_quant + ctx->t_run + ctx->t_deq;
        GGML_LOG_INFO("ork profile: %ld matmuls | quant %.0fms (%.0f%%) run %.0fms (%.0f%%) dequant %.0fms (%.0f%%) | %.1f us/matmul (run %.1f us)\n",
            ctx->n_mm, ctx->t_quant/1e3, 100*ctx->t_quant/tot, ctx->t_run/1e3, 100*ctx->t_run/tot,
            ctx->t_deq/1e3, 100*ctx->t_deq/tot, tot/ctx->n_mm, ctx->t_run/ctx->n_mm);
        if (ctx->n_dec) GGML_LOG_INFO("ork profile: decode  (M=1)  %ld matmuls, run %.1f us/matmul\n", ctx->n_dec, ctx->t_run_dec/ctx->n_dec);
        if (ctx->n_pf)  GGML_LOG_INFO("ork profile: prefill (M>1) %ld matmuls, avgM %.1f, run %.1f us/matmul (%.2f us/row)\n",
            ctx->n_pf, (double)ctx->m_pf/ctx->n_pf, ctx->t_run_pf/ctx->n_pf, ctx->t_run_pf/ctx->m_pf);
        // STREAMING breakdown: weight-resolution cost (pulled OUT of the t_quant bucket, which still
        // also holds activation-quant). All times ms; the question = which dominates the 7B streamed pass.
        {
            double sr = ctx->s_resolve, gtot = sr + ctx->t_run + ctx->t_deq;
            fprintf(stderr, "[ork STREAM] matmuls=%ld | quant(act+misc) %.0fms (%.0f%%) resolve %.0fms (%.0f%%) run %.0fms (%.0f%%) deq %.0fms (%.0f%%) | wallsum %.0fms\n",
                ctx->n_mm, ctx->t_quant/1e3, 100*ctx->t_quant/(ctx->t_quant+gtot),
                sr/1e3, 100*sr/(ctx->t_quant+gtot), ctx->t_run/1e3, 100*ctx->t_run/(ctx->t_quant+gtot),
                ctx->t_deq/1e3, 100*ctx->t_deq/(ctx->t_quant+gtot), (ctx->t_quant+gtot)/1e3);
            fprintf(stderr, "[ork STREAM] events: hits=%ld packmiss=%ld loadhit=%ld remap=%ld | DMA-unmaps=%ld ramEvict=%ld iovaUnmap=%ld\n",
                ctx->n_hit, ctx->n_packmiss, ctx->n_loadhit, ctx->n_remap,
                ctx->spool_remaps, ctx->spool_ram_evicts, ctx->spool_iova_unmaps);
            fprintf(stderr, "[ork STREAM] resolve-split: dequant(src->f32) %.0fms tile(f32->i8) %.0fms pack(i8->IOVA) %.0fms | load(.orkpack) %.0fms install(inflate+map) %.0fms remap(MEM_CREATE) %.0fms\n",
                ctx->s_dequant/1e3, ctx->s_tile/1e3, ctx->s_pack/1e3, ctx->s_load/1e3, ctx->s_install/1e3, ctx->s_remap/1e3);
            // LEVER3: split the t_quant bucket into pure activation-quant ARITHMETIC (NEON absmax+quantize loop),
            // weight RESOLVE (s_resolve), and the remaining act-side MISC (t_quant - actq - resolve = vector
            // setup/realloc/loop overhead not inside the timed loop). Answers: is the bucket arithmetic or misc?
            {
                double actq = ctx->t_actq, resolve_in_q = sr;   // both measured inside the t0..t1 window
                double misc = ctx->t_quant - actq - resolve_in_q; if (misc < 0) misc = 0;
                double W = ctx->t_quant + gtot;
                fprintf(stderr, "[ork ACTQ] t_quant=%.0fms = arith(NEON absmax+quant) %.0fms (%.0f%% of wall) + resolve %.0fms (%.0f%%) + misc %.0fms (%.0f%%) | act-quant calls=%ld\n",
                    ctx->t_quant/1e3, actq/1e3, 100*actq/W, resolve_in_q/1e3, 100*resolve_in_q/W, misc/1e3, 100*misc/W, ctx->n_actq);
            }
        }
        if (ctx->moe_calls) {
            double mt = ctx->moe_prequant + ctx->moe_pack + ctx->moe_gather + ctx->moe_chain + ctx->moe_scatter;
            fprintf(stderr, "[ork MoE-chain] %ld calls, %.0fms total | prequant %.0fms (%.0f%%) pack/repack %.0fms (%.0f%%) gather %.0fms (%.0f%%) chain-submit %.0fms (%.0f%%) scatter %.0fms (%.0f%%)\n",
                ctx->moe_calls, mt/1e3,
                ctx->moe_prequant/1e3, 100*ctx->moe_prequant/mt, ctx->moe_pack/1e3, 100*ctx->moe_pack/mt,
                ctx->moe_gather/1e3, 100*ctx->moe_gather/mt, ctx->moe_chain/1e3, 100*ctx->moe_chain/mt,
                ctx->moe_scatter/1e3, 100*ctx->moe_scatter/mt);
            fprintf(stderr, "[ork MoE-chain] pack split: Q4_K->f32 dequant %.0fms (%.0f%%) | f32->int8 quant+tile %.0fms (%.0f%%)\n",
                ctx->moe_deq/1e3, 100*ctx->moe_deq/ctx->moe_pack, ctx->moe_quant/1e3, 100*ctx->moe_quant/ctx->moe_pack);
            fprintf(stderr, "[ork MoE-VERIFY] first-touch live-packs=%ld (%.0fms, %.1f ms/pack) | chain-submit calls=%ld (%.0fms, %.3f ms/submit-call)\n",
                ctx->moe_pack_calls, ctx->moe_pack/1e3, ctx->moe_pack_calls? ctx->moe_pack/ctx->moe_pack_calls/1e3 : 0.0,
                ctx->moe_calls, ctx->moe_chain/1e3, ctx->moe_calls? ctx->moe_chain/ctx->moe_calls/1e3 : 0.0);
            fprintf(stderr, "[ork MoE-VERIFY2] avg S(tasks/call)=%.2f | per-task-fallback: calls=%ld (%.0fms, of total chain %.0fms) | chain-only=%.0fms over %ld calls = %.3f ms/chaincall\n",
                ctx->moe_calls? (double)ctx->moe_chain_S_sum/ctx->moe_calls : 0.0,
                ctx->moe_fallback_calls, ctx->moe_fallback_t/1e3, ctx->moe_chain/1e3,
                (ctx->moe_chain-ctx->moe_fallback_t)/1e3, ctx->moe_calls, ctx->moe_calls? (ctx->moe_chain-ctx->moe_fallback_t)/ctx->moe_calls/1e3:0.0);
        }
        if (ctx->moe_cold_calls)
            fprintf(stderr, "[ork MoE-cold] %ld cold-expert GEMV calls (threaded ggml vec_dot) | %.0fms total | %.1f us/expert\n",
                ctx->moe_cold_calls, ctx->moe_cold/1e3, ctx->moe_cold/ctx->moe_cold_calls);
        if (ctx->pathb_calls) {
            const double npu = ctx->pathb_npu_t, cpu = ctx->pathb_cpu_t, wall = ctx->pathb_wall_t;
            fprintf(stderr, "[ork PATH-B] %ld calls | NPU-experts=%ld CPU-experts=%ld | npu=%.0fms cpu=%.0fms combine=%.0fms wall=%.0fms | overlap-eff=%.2fx (sum/wall) | combine=%.1f%% of wall\n",
                ctx->pathb_calls, ctx->pathb_npu_experts, ctx->pathb_cpu_experts,
                npu/1e3, cpu/1e3, ctx->pathb_combine_t/1e3, wall/1e3,
                wall>0 ? (npu+cpu)/wall : 0.0, wall>0 ? 100*ctx->pathb_combine_t/wall : 0.0);
        }
        // run_multicore phase split: where the per-matmul "run" time actually goes (kernel vs machinery)
        double rt_s = 0, rt_sub = 0, rt_cp = 0; long rt_n = 0;
        ork_npu_run_timing(&rt_s, &rt_sub, &rt_cp, &rt_n);
        if (rt_n) {
            double rt_tot = rt_s + rt_sub + rt_cp;
            GGML_LOG_INFO("ork profile: run_multicore %ld calls | setup %.0fms (%.0f%%) submit %.0fms (%.0f%%) copy %.0fms (%.0f%%) | %.1f us/call (setup %.1f submit %.1f copy %.1f)\n",
                rt_n, rt_s/1e3, 100*rt_s/rt_tot, rt_sub/1e3, 100*rt_sub/rt_tot, rt_cp/1e3, 100*rt_cp/rt_tot,
                rt_tot/rt_n, rt_s/rt_n, rt_sub/rt_n, rt_cp/rt_n);
        }
    }
}
// LEVER3: atexit shim — fires the profile dump even when llama-bench never calls backend free.
static void ork_profile_atexit(void) { ork_profile_dump(g_ork_ctx); }
static void attn_pool_free(ggml_backend_ork_context * ctx);   // fwd decl (defined with the attention path below)

static void ggml_backend_ork_free(ggml_backend_t backend) {
    ggml_backend_ork_context * ctx = (ggml_backend_ork_context *) backend->context;
    if (ctx->async_inflight && ctx->async_thr.joinable()) ctx->async_thr.join();   // drain any in-flight async NPU graph first
    ctx->async_inflight = false;
    ork_profile_dump(ctx);
    if(getenv("ORK_VERBOSE"))fprintf(stderr, "[ORK DEBUG] ggml_backend_ork_free called!\n");
    ork_persist_finalize(ctx);   // .orkpack: write index+footer, rename .tmp into place (write mode)
    if (ctx->persist_mode) fprintf(stderr, "[ORK PERSIST] this run: loaded %ld from disk, packed %ld\n", ctx->persist_hits, ctx->persist_misses);
    // (a) gmax profile report: the auto-captured, model-specific gate-range distribution + an adaptive
    // cutoff (median + 3*MAD = the outlier layers). The fused int8 silu can't be rescued by gmax gating
    // (its int8 OUTPUT is coarse at ALL gmax -> PPL 55), so the SHIPPED policy stays all-CPU-silu; this
    // profile is the tuning DATA that (b) persists into the orkpack and a FUTURE all-NPU selective int16
    // gate (#35) would apply. Printed under ORK_VERBOSE or ORK_FFN_GMAX_REPORT.
    if (!ctx->gmax_profile.empty() && (getenv("ORK_VERBOSE") || getenv("ORK_FFN_GMAX_REPORT"))) {
        std::vector<std::pair<std::string, float>> g = ctx->gmax_profile;
        std::sort(g.begin(), g.end(), [](const std::pair<std::string,float>&a, const std::pair<std::string,float>&b){ return a.second > b.second; });
        std::vector<float> s; for (auto & p : g) s.push_back(p.second); std::sort(s.begin(), s.end());
        float med = s[s.size()/2];
        std::vector<float> dev; for (float v : s) dev.push_back(fabsf(v - med)); std::sort(dev.begin(), dev.end());
        float mad = dev.empty() ? 0.0f : dev[dev.size()/2]; float cut = med + 3.0f * (mad > 1e-6f ? mad : 1.0f);
        int nout = 0; for (auto & p : g) if (p.second > cut) nout++;
        fprintf(stderr, "[ORK FFN-GMAX] %zu FFN layers | median=%.1f MAD=%.1f | adaptive cutoff(med+3MAD)=%.1f -> %d outlier layer(s)\n",
                g.size(), med, mad, cut, nout);
        for (auto & p : g)
            fprintf(stderr, "[ORK FFN-GMAX]   %-28s gmax=%8.2f%s\n", p.first.c_str(), p.second, p.second > cut ? "  <-- outlier" : "");
    }
    // (b) persist the gmax profile to the <orkpack>.gmax sidecar (name<TAB>gmax/line) so a later run
    // loads it. Written whenever we captured a profile and have a persist path — independent of ORK_VERBOSE.
    if (!ctx->gmax_profile.empty()) {
        const char * pp = getenv("ORK_ORKPACK_PATH");   // development override; else derive from the model
        std::string pp_d; if ((!pp || !*pp)) { pp_d = ork_default_orkpack_path(); if (!pp_d.empty()) pp = pp_d.c_str(); }
        if (pp && pp[0]) {
            std::string sp = std::string(pp) + ".gmax";
            FILE * gf = fopen(sp.c_str(), "w");
            if (gf) { for (auto & p : ctx->gmax_profile) fprintf(gf, "%s\t%.6g\n", p.first.c_str(), p.second);
                fclose(gf);
                if (getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK FFN-GMAX] wrote %zu-layer gmax profile -> %s\n", ctx->gmax_profile.size(), sp.c_str()); }
        }
    }
    if (ctx->moe_hot_hits || ctx->moe_cold_cpu) {
        long tot = ctx->moe_hot_hits + ctx->moe_cold_cpu;
        fprintf(stderr, "[ORK MOE PARTITION] hot-K=%s peak-resident=%.3f GiB | expert-calls: NPU(hot)=%ld CPU(cold)=%ld | hit-rate=%.1f%%\n",
            getenv("ORK_MOE_HOT") ? getenv("ORK_MOE_HOT") : "8",
            (double) ctx->moe_hot_peak / (1024.0*1024.0*1024.0),
            ctx->moe_hot_hits, ctx->moe_cold_cpu, tot ? 100.0*ctx->moe_hot_hits/tot : 0.0);
    }
    if (ctx->phase_evict)
        fprintf(stderr, "[ORK PHASE EVICT] backbone bulk-frees=%ld total-freed=%.3f GiB\n",
            ctx->backbone_evicts, ctx->backbone_evict_bytes / (1024.0*1024.0*1024.0));
    if (ctx->n_domains > 1) {
        size_t tot = 0; for (int d = 0; d < ctx->n_domains; d++) tot += ctx->domain_bytes[d];
        fprintf(stderr, "[ork RESIDENT] final: %.2f GiB resident across %d domains; per-token pack/load (churn) = %ld (target 0)\n",
            tot / (1024.0*1024.0*1024.0), ctx->n_domains, ctx->mem_create_runtime);
        for (int d = 0; d < ctx->n_domains; d++)
            fprintf(stderr, "[ork RESIDENT]   domain %d: %.2f GiB\n", d, ctx->domain_bytes[d] / (1024.0*1024.0*1024.0));
    }
    if (const char * fp = getenv("ORK_MOE_PROFILE_FREQ")) {
        if (!ctx->moe_freq.empty()) {
            FILE * f = fopen(fp, "w");
            if (f) {
                fprintf(f, "# tensor\texpert\tcount  (routing-frequency profile for tier-map build)\n");
                for (auto & kv : ctx->moe_freq)
                    for (size_t e = 0; e < kv.second.size(); e++)
                        fprintf(f, "%s\t%zu\t%ld\n", kv.first.c_str(), e, kv.second[e]);
                fclose(f);
                fprintf(stderr, "[ORK MOE FREQ] wrote routing-frequency profile (%zu tensors) -> %s\n", ctx->moe_freq.size(), fp);
            }
        }
    }
    if (ctx->spool)
        fprintf(stderr, "[ork STREAM-POOL] entries=%zu | RAM held=%.2f GiB | IOVA mapped=%.2f GiB | remaps(cheap)=%ld IOVA-unmaps=%ld RAM-evicts=%ld\n",
            ctx->wcache.size(), ctx->spool_ram_bytes/(1024.0*1024.0*1024.0), ctx->wcache_bytes/(1024.0*1024.0*1024.0),
            ctx->spool_remaps, ctx->spool_iova_unmaps, ctx->spool_ram_evicts);
    for (auto & kv : ctx->slice_ws) if (kv.second) ork_mm_free_sliced(ctx->npu, kv.second);   // ORK_SLICE_ROUTE sliced-doorbell twins
    ctx->slice_ws.clear();
    if (ctx->persist_map) munmap(ctx->persist_map, ctx->persist_map_sz);
    if (ctx->spool) ork_stream_pool_free(ctx->spool);   // frees all stream entries' RAM dma-bufs
    if (ctx->dma_in) ork_dma_free(ctx->npu, ctx->dma_in);    // static-graph DMA scratch (ORK_GU_CHAIN)
    if (ctx->dma_up) ork_dma_free(ctx->npu, ctx->dma_up);
    if (ctx->dma_gt) ork_dma_free(ctx->npu, ctx->dma_gt);
    for (int d = 0; d < 64; d++) if (ctx->dma_moeC[d]) ork_dma_free(ctx->npu, ctx->dma_moeC[d]);
    for (auto & kv : ctx->wcache) ork_w_free(kv.second.w);   // w is NULL for stream-pool entries (no-op)
    for (auto & kv : ctx->f16_scratch) if (kv.second) ork_mm_free(ctx->npu, kv.second);   // ORK_FFN_F16_JIT shared scratches
    for (auto & p : ctx->moe_pools) for (auto & s : p.second) if (s.w) ork_w_free(s.w);   // MoE expert pool
    for (auto & tk : ctx->moe_hot) for (auto & es : tk.second) if (es.second.w) ork_w_free(es.second.w);   // hot-expert partition
    if (ctx->cpu_backend) ggml_backend_free(ctx->cpu_backend);   // PATH (b) cached CPU backend
    for (auto & kv : ctx->pathb_repack) { if (kv.second.buf) ggml_backend_buffer_free(kv.second.buf); if (kv.second.gctx) ggml_free(kv.second.gctx); }
    attn_pool_free(ctx);                                         // per-context fused-attention pool (needs ctx->npu)
    { struct ork_attn_sm & s = ctx->attn_sm;                     // per-context softmax scratch
      free(s.xi); free(s.ei); free(s.mx); free(s.ss); free(s.e); free(s.q8); free(s.invf);
      if (s.ones) ork_mm_free(ctx->npu, s.ones); s = (struct ork_attn_sm){}; }
    for (auto & e : ctx->attn_kv) {                              // resident-KV decode buffers (ORK_ATTN_KV)
        for (ork_kv_resident * r : e.second.kv) if (r) ork_kv_resident_free(ctx->npu, r);
        if (e.second.ones) ork_mm_free(ctx->npu, e.second.ones);   // ORK_ATTN_FUSED resident ones[Lmax,32]
    }
    ctx->attn_kv.clear();
    if (ctx->npu) ork_npu_free(ctx->npu);
    delete ctx;
    g_ork_ctx = nullptr;
    delete backend;
}

static bool ork_is_expert(const char * name) {
    if (!name) return false;
    if (strstr(name, "expert") != nullptr) return true;
    if (strstr(name, "exps") != nullptr) return true;
    if (strstr(name, "shexp") != nullptr) return true;
    const char * p = strstr(name, "ffn_gate.");
    if (p && p[9] >= '0' && p[9] <= '9') return true;
    p = strstr(name, "ffn_up.");
    if (p && p[7] >= '0' && p[7] <= '9') return true;
    p = strstr(name, "ffn_down.");
    if (p && p[9] >= '0' && p[9] <= '9') return true;
    return false;
}

enum ork_chain_type {
    ORK_CHAIN_NONE,
    ORK_CHAIN_I8,
    ORK_CHAIN_I4
};

static ork_chain_type get_node_chain_type(ggml_backend_ork_context * ctx, struct ggml_tensor * node) {
    if (node->op != GGML_OP_MUL_MAT) {
        return ORK_CHAIN_NONE;
    }
    struct ggml_tensor * src0 = node->src[0];
    int64_t K = src0->ne[0];
    int64_t N = src0->ne[1];

    const char * name = src0->name;
    bool is_ffn = strstr(name, "ffn_") || ork_is_expert(name);
    bool is_attn = strstr(name, "attn_q") || strstr(name, "attn_k") || strstr(name, "attn_v") || strstr(name, "attn_output");
    
    int target_qbits = ctx->qbits;
    if (ctx->hybrid) {
        if (is_ffn) target_qbits = 4;
        else if (is_attn) target_qbits = 8;
    }
    if (target_qbits == 8) {
        if (K <= 10752 && N <= 4096 && (K % 32 == 0) && (N % 32 == 0)) {
            return ORK_CHAIN_I8;
        }
    } else if (target_qbits == 4) {
        if (K <= 10752 && N <= 4096 && (K % 32 == 0) && (N % 64 == 0)) {
            return ORK_CHAIN_I4;
        }
    }
    return ORK_CHAIN_NONE;
}


static bool ggml_backend_ork_mul_mat_chain_i4(ggml_backend_ork_context * ctx, struct ggml_tensor ** nodes, int count) {
    if(getenv("ORK_VERBOSE"))fprintf(stderr, "[ORK] START mul_mat_chain_i4, count=%d\n", count); fflush(stderr);

    const int M = nodes[0]->src[1]->ne[1];
    const int M_padded = (M == 1) ? 1 : ((M + 31) / 32) * 32;

    size_t total_ai_size = 0;
    size_t total_as_size = 0;
    size_t total_ci_size = 0;
    for (int i = 0; i < count; i++) {
        struct ggml_tensor * dst = nodes[i];
        int K = dst->src[0]->ne[0];
        int N = dst->src[0]->ne[1];
        total_ai_size += (size_t)M_padded * K;
        total_as_size += (size_t)M_padded;
        total_ci_size += (size_t)M_padded * N;
    }
    ctx->ai.resize(total_ai_size);
    ctx->as.resize(total_as_size);
    ctx->ci.resize(total_ci_size);

    int8_t  * ai_base = ctx->ai.data();
    float   * as_base = ctx->as.data();
    int32_t * ci_base = ctx->ci.data();

    size_t ai_offset = 0;
    size_t as_offset = 0;
    size_t ci_offset = 0;

    std::vector<ork_mm_task_i4> tasks;
    const double t0 = ctx->profile ? ork_now_us() : 0;

    std::unordered_map<const void *, std::pair<int8_t *, float *>> chain_act_cache;

    for (int i = 0; i < count; i++) {
        struct ggml_tensor * dst = nodes[i];
        const struct ggml_tensor * src0 = dst->src[0];
        const struct ggml_tensor * src1 = dst->src[1];

        const enum ggml_type type = src0->type;
        const int K = (int) src0->ne[0];
        const int N = (int) src0->ne[1];
        const int M = (int) src1->ne[1];

        const auto * tt = ggml_get_type_traits(type);
        ggml_to_float_t const to_float = tt->to_float;

        const char  * x = (const char *) src0->data;
        const float * y = (const float *) src1->data;

        // weight: check cache / pack
        auto it = ctx->wcache.find(x);
        if (it == ctx->wcache.end()) {
            ctx->f32.resize((size_t) N * K);
            ctx->bi .resize((size_t) K * N);
            float * f32 = ctx->f32.data();
            int8_t * bi = ctx->bi.data();
            if (type == GGML_TYPE_F32) {
                for (int64_t n = 0; n < N; n++) memcpy(f32 + n*K, x + n*src0->nb[1], (size_t) K*sizeof(float));
            } else {
                for (int64_t n = 0; n < N; n++) to_float((const char *) x + n*src0->nb[1], f32 + n*K, K);
            }
            ork_weight ow; ow.gsize = 0; ow.bscale.resize((size_t) N);
            if (ctx->hadamard) {
                const int b = K & (-K);
                for (int n = 0; n < N; n++) {
                    float * col = f32 + (size_t) n*K;
                    for (int off = 0; off < K; off += b) {
                        ork_fwht_norm(col + off, b);
                    }
                }
            }
            for (int n = 0; n < N; n++) {
                float * col = f32 + (size_t) n*K;
                float mx = 1e-9f;
                for (int k = 0; k < K; k++) { float v = fabsf(col[k]); if (v > mx) mx = v; }
                float s = mx / 7.0f; ow.bscale[n] = s;
                for (int k = 0; k < K; k++) {
                    int q = (int) lrintf(col[k] / s);
                    bi[(size_t) k*N + n] = (int8_t) (q > 7 ? 7 : q < -8 ? -8 : q);
                }
            }
            ow.w = ork_i4_mm_pack(ctx->npu, K, N, bi);
            if (!ow.w) return false;
            it = ctx->wcache.emplace(x, std::move(ow)).first;
        }
        const ork_weight & ow = it->second;

        // activation: check cache or quantize
        int8_t * task_A = nullptr;
        float  * task_as = nullptr;
        auto act_it = chain_act_cache.find(src1->data);
        if (act_it != chain_act_cache.end()) {
            task_A = act_it->second.first;
            task_as = act_it->second.second;
        } else {
            task_A = ai_base + ai_offset;
            task_as = as_base + as_offset;
            #pragma omp parallel for if (M_padded >= 16)
            for (int m = 0; m < M_padded; m++) {
                if (m < M) {
                    if (ctx->hadamard) {
                        int b = K & (-K);
                        std::vector<float> arow_local(K);
                        memcpy(arow_local.data(), y + (size_t) m*K, (size_t) K*sizeof(float));
                        for (int off = 0; off < K; off += b) {
                            ork_fwht_norm(arow_local.data() + off, b);
                        }
                        float mx = 1e-9f;
                        for (int k = 0; k < K; k++) { float v = fabsf(arow_local[k]); if (v > mx) mx = v; }
                        float s = mx / 7.0f; task_as[m] = s;
                        for (int k = 0; k < K; k++) {
                            int q = (int) lrintf(arow_local[k] / s);
                            task_A[(size_t) m*K + k] = (int8_t) (q > 7 ? 7 : q < -8 ? -8 : q);
                        }
                    } else {
                        const float * yr = y + (size_t) m*K;
                        int8_t * amr = task_A + (size_t) m*K;
                        float mx = 1e-9f;
                        for (int k = 0; k < K; k++) { float v = yr[k] >= 0.0f ? yr[k] : -yr[k]; mx = v > mx ? v : mx; }
                        task_as[m] = mx / 7.0f;
                        const float inv = 7.0f / mx;
                        for (int k = 0; k < K; k++) {
                            float q = yr[k] * inv;
                            int qi = (int) (q + copysignf(0.5f, q));
                            amr[k] = (int8_t) (qi > 7 ? 7 : qi < -8 ? -8 : qi);
                        }
                    }
                } else {
                    memset(task_A + (size_t) m*K, 0, K);
                    task_as[m] = 0.0f;
                }
            }
            chain_act_cache[src1->data] = {task_A, task_as};
            ai_offset += (size_t)M_padded * K;
            as_offset += (size_t)M_padded;
        }

        tasks.push_back({
            ow.w,
            M_padded,
            task_A,
            ci_base + ci_offset
        });

        ci_offset += (size_t)M_padded * N;
    }

    const double t1 = ctx->profile ? ork_now_us() : 0;

    if(getenv("ORK_VERBOSE"))fprintf(stderr, "[ORK] i4 chain submit: tasks=%zu\n", tasks.size());
    fflush(stderr);
    
    int ok = 0;
    if (tasks.size() == 1) {
        ok = ork_i4_mm_run(ctx->npu, tasks[0].w, tasks[0].M, tasks[0].A, tasks[0].C) ? -1 : 0;
    } else {
        ok = ork_i4_mm_run_chain(ctx->npu, tasks.size(), tasks.data());
    }
    
    if (ok != 0) {
        // Fallback to sequential single-task run
        if(getenv("ORK_VERBOSE"))fprintf(stderr, "[ORK] i4 chain failed (%d), falling back to sequential\n", ok); fflush(stderr);
        for (size_t t = 0; t < tasks.size(); t++) {
            if (ork_i4_mm_run(ctx->npu, tasks[t].w, tasks[t].M, tasks[t].A, tasks[t].C)) {
                return false;
            }
        }
    }

    const double t2 = ctx->profile ? ork_now_us() : 0;

    // Dequantize results
    ci_offset = 0;
    for (int i = 0; i < count; i++) {
        struct ggml_tensor * dst = nodes[i];
        const char * x = (const char *) dst->src[0]->data;
        const struct ggml_tensor * src1 = dst->src[1];
        float * d = (float *) dst->data;
        int N = dst->src[0]->ne[1];
        int M = dst->src[1]->ne[1];

        auto it = ctx->wcache.find(x);
        const ork_weight & ow = it->second;
        const float * bs = ow.bscale.data();
        
        auto act_it = chain_act_cache.find(src1->data);
        const float * task_as = act_it->second.second;
        const int32_t * ctr = ci_base + ci_offset;
        ci_offset += (size_t)M_padded * N;

        #pragma omp parallel for if (M >= 16)
        for (int m = 0; m < M; m++) {
            const float rs = task_as[m];
            const int32_t * cr = ctr + (size_t) m*N;
            float * dr = d + (size_t) m*N;
#if defined(__ARM_NEON)
            float32x4_t v_rs = vdupq_n_f32(rs);
            int n = 0;
            for (; n <= N - 8; n += 8) {
                int32x4_t v_cr0 = vld1q_s32(cr + n);
                int32x4_t v_cr1 = vld1q_s32(cr + n + 4);
                float32x4_t v_cr_f0 = vcvtq_f32_s32(v_cr0);
                float32x4_t v_cr_f1 = vcvtq_f32_s32(v_cr1);
                float32x4_t v_bs0 = vld1q_f32(bs + n);
                float32x4_t v_bs1 = vld1q_f32(bs + n + 4);
                float32x4_t v_prod0 = vmulq_f32(v_bs0, v_cr_f0);
                float32x4_t v_prod1 = vmulq_f32(v_bs1, v_cr_f1);
                float32x4_t v_dr0 = vmulq_f32(v_prod0, v_rs);
                float32x4_t v_dr1 = vmulq_f32(v_prod1, v_rs);
                vst1q_f32(dr + n, v_dr0);
                vst1q_f32(dr + n + 4, v_dr1);
            }
            for (; n < N; n++) {
                dr[n] = rs * bs[n] * (float) cr[n];
            }
#else
            for (int n = 0; n < N; n++) dr[n] = rs * bs[n] * (float) cr[n];
#endif
        }
    }

    if (ctx->profile) {
        double t3 = ork_now_us();
        ctx->t_quant += t1 - t0;
        ctx->t_run   += t2 - t1;
        ctx->t_deq   += t3 - t2;
        ctx->n_mm    += count;
        for (int i = 0; i < count; i++) {
            int M = nodes[i]->src[1]->ne[1];
            double part_run = (t2 - t1) / count;
            if (M > 1) {
                ctx->t_run_pf  += part_run;
                ctx->n_pf      += 1;
                ctx->m_pf      += M;
            } else {
                ctx->t_run_dec += part_run;
                ctx->n_dec     += 1;
            }
        }
    }

    if (ctx->no_cache) {
        for (int i = 0; i < count; i++) {
            struct ggml_tensor * dst = nodes[i];
            const char * x = (const char *) dst->src[0]->data;
            auto it = ctx->wcache.find(x);
            if (it != ctx->wcache.end()) {
                ork_w_free(it->second.w);
                ctx->wcache.erase(it);
            }
        }
    }

    return true;
}

static bool ggml_backend_ork_mul_mat_chain_i8(ggml_backend_ork_context * ctx, struct ggml_tensor ** nodes, int count) {
    if(getenv("ORK_VERBOSE"))fprintf(stderr, "[ORK] START mul_mat_chain_i8, count=%d\n", count); fflush(stderr);

    const int M = nodes[0]->src[1]->ne[1];
    const int M_padded = (M == 1) ? 1 : ((M + 31) / 32) * 32;

    size_t total_ai_size = 0;
    size_t total_as_size = 0;
    size_t total_ci_size = 0;
    for (int i = 0; i < count; i++) {
        struct ggml_tensor * dst = nodes[i];
        int K = dst->src[0]->ne[0];
        int N = dst->src[0]->ne[1];
        total_ai_size += (size_t)M_padded * K;
        total_as_size += (size_t)M_padded;
        total_ci_size += (size_t)M_padded * N;
    }
    ctx->ai.resize(total_ai_size);
    ctx->as.resize(total_as_size);
    ctx->ci.resize(total_ci_size);

    int8_t  * ai_base = ctx->ai.data();
    float   * as_base = ctx->as.data();
    int32_t * ci_base = ctx->ci.data();

    size_t ai_offset = 0;
    size_t as_offset = 0;
    size_t ci_offset = 0;

    std::vector<ork_mm_task_i8> tasks;
    const double t0 = ctx->profile ? ork_now_us() : 0;

    std::unordered_map<const void *, std::pair<int8_t *, float *>> chain_act_cache;

    for (int i = 0; i < count; i++) {
        struct ggml_tensor * dst = nodes[i];
        const struct ggml_tensor * src0 = dst->src[0];
        const struct ggml_tensor * src1 = dst->src[1];

        const enum ggml_type type = src0->type;
        const int K = (int) src0->ne[0];
        const int N = (int) src0->ne[1];
        const int M = (int) src1->ne[1];

        const auto * tt = ggml_get_type_traits(type);
        ggml_to_float_t const to_float = tt->to_float;

        const char  * x = (const char *) src0->data;
        const float * y = (const float *) src1->data;

        // weight: wcache / .orkpack load / pack — identical resolution + persist as the non-chain path,
        // so chain-routed FFN/attn weights are captured into (and reloaded from) the .orkpack too.
        auto it = ork_resolve_weight_i8(ctx, src0, K, N, src0->nb[1], type, to_float, /*allow_evict=*/false);
        if (it == ctx->wcache.end()) return false;
        it->second.last_use = ++ctx->wcache_tick;
        const ork_weight & ow = it->second;

        // activation: check cache or quantize
        int8_t * task_A = nullptr;
        float  * task_as = nullptr;
        auto act_it = chain_act_cache.find(src1->data);
        if (act_it != chain_act_cache.end()) {
            task_A = act_it->second.first;
            task_as = act_it->second.second;
        } else {
            task_A = ai_base + ai_offset;
            task_as = as_base + as_offset;
            const double _aq0 = ctx->profile ? ork_now_us() : 0;   // LEVER3: time chain-path act-quant (now NEON via ork_quant_row_i8)
            #pragma omp parallel for if (M_padded >= 16)
            for (int m = 0; m < M_padded; m++) {
                if (m < M) {
                    task_as[m] = ork_quant_row_i8(y + (size_t) m*K, K, task_A + (size_t) m*K);
                } else {
                    memset(task_A + (size_t) m*K, 0, K);
                    task_as[m] = 0.0f;
                }
            }
            if (ctx->profile) { ctx->t_actq += ork_now_us() - _aq0; ctx->n_actq++; }   // LEVER3
            chain_act_cache[src1->data] = {task_A, task_as};
            ai_offset += (size_t)M_padded * K;
            as_offset += (size_t)M_padded;
        }

        tasks.push_back({
            ow.w,
            M_padded,
            task_A,
            ci_base + ci_offset
        });

        ci_offset += (size_t)M_padded * N;
    }

    const double t1 = ctx->profile ? ork_now_us() : 0;

    if(getenv("ORK_VERBOSE"))fprintf(stderr, "[ORK] i8 chain submit: tasks=%zu\n", tasks.size());
    fflush(stderr);
    
    if (ork_submit_sync(ctx, tasks)) return false;   // unified router: 1→run_i8 · N same-dom→run_stream · N cross→run_chain

    const double t2 = ctx->profile ? ork_now_us() : 0;

    // Dequantize results
    ci_offset = 0;
    for (int i = 0; i < count; i++) {
        struct ggml_tensor * dst = nodes[i];
        const char * x = (const char *) dst->src[0]->data;
        const struct ggml_tensor * src1 = dst->src[1];
        float * d = (float *) dst->data;
        int N = dst->src[0]->ne[1];
        int M = dst->src[1]->ne[1];

        auto it = ctx->wcache.find(x);
        const ork_weight & ow = it->second;
        const float * bs = ow.bscale.data();
        
        auto act_it = chain_act_cache.find(src1->data);
        const float * task_as = act_it->second.second;
        const int32_t * ctr = ci_base + ci_offset;
        ci_offset += (size_t)M_padded * N;

        for (int m = 0; m < M; m++) {
            const float rs = task_as[m];
            const int32_t * cr = ctr + (size_t) m*N;
            float * dr = d + (size_t) m*N;
#if defined(__ARM_NEON)
            float32x4_t v_rs = vdupq_n_f32(rs);
            int n = 0;
            for (; n <= N - 8; n += 8) {
                int32x4_t v_cr0 = vld1q_s32(cr + n);
                int32x4_t v_cr1 = vld1q_s32(cr + n + 4);
                float32x4_t v_cr_f0 = vcvtq_f32_s32(v_cr0);
                float32x4_t v_cr_f1 = vcvtq_f32_s32(v_cr1);
                float32x4_t v_bs0 = vld1q_f32(bs + n);
                float32x4_t v_bs1 = vld1q_f32(bs + n + 4);
                float32x4_t v_prod0 = vmulq_f32(v_bs0, v_cr_f0);
                float32x4_t v_prod1 = vmulq_f32(v_bs1, v_cr_f1);
                float32x4_t v_dr0 = vmulq_f32(v_prod0, v_rs);
                float32x4_t v_dr1 = vmulq_f32(v_prod1, v_rs);
                vst1q_f32(dr + n, v_dr0);
                vst1q_f32(dr + n + 4, v_dr1);
            }
            for (; n < N; n++) {
                dr[n] = rs * bs[n] * (float) cr[n];
            }
#else
            for (int n = 0; n < N; n++) dr[n] = rs * bs[n] * (float) cr[n];
#endif
        }
    }

    if (ctx->profile) {
        double t3 = ork_now_us();
        ctx->t_quant += t1 - t0;
        ctx->t_run   += t2 - t1;
        ctx->t_deq   += t3 - t2;
        ctx->n_mm    += count;
        for (int i = 0; i < count; i++) {
            int M = nodes[i]->src[1]->ne[1];
            double part_run = (t2 - t1) / count;
            if (M > 1) {
                ctx->t_run_pf  += part_run;
                ctx->n_pf      += 1;
                ctx->m_pf      += M;
            } else {
                ctx->t_run_dec += part_run;
                ctx->n_dec     += 1;
            }
        }
    }

    if (ctx->no_cache) {
        for (int i = 0; i < count; i++) {
            struct ggml_tensor * dst = nodes[i];
            const char * x = (const char *) dst->src[0]->data;
            auto it = ctx->wcache.find(x);
            if (it != ctx->wcache.end()) {
                ork_w_free(it->second.w);
                ctx->wcache.erase(it);
            }
        }
    }

    return true;
}



// Dequant one expert-weight output channel (row) -> dst[K] for ork_i8_mm_pack_dequant: fuses ggml's
// Q4_K->f32 with ork-driver's int8 quant+tile so the full f32[N][K] is never materialized (kills the
// DRAM round-trip — alloc + write + read-back of N*K floats — that was part of the MoE repack cost).
struct ork_moe_deq_ctx { const char * x; size_t nb01; ggml_to_float_t to_float; bool is_f32; };
static void ork_moe_deq_row(void * vctx, int n, float * dst, int K) {
    const ork_moe_deq_ctx * c = (const ork_moe_deq_ctx *) vctx;
    if (c->is_f32) memcpy(dst, c->x + (size_t) n * c->nb01, (size_t) K * sizeof(float));
    else           c->to_float(c->x + (size_t) n * c->nb01, dst, K);
}

static void * ork_dma_grow(ork_npu * npu, void ** buf, size_t * sz, size_t need);   // defined below; used by the #14 multi-core NONBLOCK path
// MoE expert matmul (GGML_OP_MUL_MAT_ID), int8 path. Handles any n_tokens.
// dst[N, n_used, n_tokens] = for each (token t, slot j): W[ids[j,t]] (K x N) @ x_t (K).
// We GROUP tokens by their selected expert and run ONE M=count matmul per active expert (M-padded to 32
// like the dense path) — for prefill (M>1) this amortizes the submit floor + the expert-weight read
// across the routed tokens; for decode (M=1) it degenerates to one matmul per selected expert.
static bool ggml_backend_ork_mul_mat_id_i8(ggml_backend_ork_context * ctx, struct ggml_tensor * dst) {
    if(getenv("ORK_VERBOSE"))fprintf(stderr, "[ORK] START mul_mat_id_i8\n"); fflush(stderr);
    const struct ggml_tensor * src0 = dst->src[0];   // experts [K, N, n_expert]
    const struct ggml_tensor * src1 = dst->src[1];   // input   [K, 1, n_tokens]
    const struct ggml_tensor * ids  = dst->src[2];   // ids     [n_used, n_tokens] (i32)
    const enum ggml_type type = src0->type;
    const int K = (int) src0->ne[0];
    const int N = (int) src0->ne[1];
    const int n_used   = (int) ids->ne[0];
    const int n_tokens = (int) src1->ne[2];
    const int n_b1     = (int) src1->ne[1];          // 1 = broadcast (same input to all of a token's experts)

    const auto * tt = ggml_get_type_traits(type); ggml_to_float_t to_float = tt->to_float;
    const double t0 = ctx->profile ? ork_now_us() : 0;
    ctx->last_src1 = nullptr;                         // we use ctx->ai/ci — kill the dense reuse cache

    // .orkpack convert (write mode): persist ALL n_expert slices of this `_exps` tensor (not just the
    // routed ones — a complete pack needs every expert). Deduped per synthetic key so re-visits in a
    // multi-token / multi-step convert don't re-dump. Inference (mode 1/0) skips this entirely.
    if (ctx->persist_mode == 2) {
        ork_persist_write_experts(ctx, src0, K, N, type, to_float);
        // Convert is a one-time pack pass: the actual MoE forward doesn't need the NPU here (the
        // per-expert submit floor makes MoE-on-NPU very slow and it can soft-reset the device on this
        // shape). Compute the result on CPU directly — dequant each routed expert's [N][K] plane and
        // do an f32 GEMV per (token,slot) — so convert is fast and never touches the flaky MoE submit.
        std::vector<float> ew((size_t) N * K);
        char * dbase_w = (char *) dst->data;
        for (int t = 0; t < n_tokens; t++) {
            const int32_t * idp = (const int32_t *)((const char *) ids->data + (size_t) t * ids->nb[1]);
            for (int j = 0; j < n_used; j++) {
                const int e = idp[j];
                const char * x = (const char *) src0->data + (size_t) e * src0->nb[2];
                if (type == GGML_TYPE_F32) {
                    for (int64_t n = 0; n < N; n++) memcpy(ew.data() + n*K, x + n*src0->nb[1], (size_t) K*sizeof(float));
                } else {
                    for (int64_t n = 0; n < N; n++) to_float(x + n*src0->nb[1], ew.data() + n*K, K);
                }
                const float * y = (const float *)((const char *) src1->data + (size_t) (n_b1==1?0:j)*src1->nb[1] + (size_t) t*src1->nb[2]);
                float * dr = (float *)(dbase_w + (size_t) j * dst->nb[1] + (size_t) t * dst->nb[2]);
                for (int n = 0; n < N; n++) {
                    const float * wr = ew.data() + (size_t) n * K;
                    float acc = 0.f; for (int k = 0; k < K; k++) acc += wr[k] * y[k];
                    dr[n] = acc;
                }
            }
        }
        if (ctx->profile) { ctx->t_run += ork_now_us() - t0; ctx->n_mm++; }
        return true;
    }

    char * dbase = (char *) dst->data;
    const bool bcast = (n_b1 == 1);

    // ===== NATIVE W4A4 EXPERT PATH (int4 tier; hadamard implied) =====
    // The int4 twin of the int8 expert compute below. Shares this handler's routing (the src0/src1/ids parse
    // + the persist_mode==2 convert branch above); only the inner per-expert kernel differs: FWHT-rotate +
    // int4-quant the weight AND the activation (QuaRot — an orthonormal rotation preserves the dot product
    // but makes both int4-friendly), ONE ork_i4_mm_run per active expert, per-(row,channel) fp32 dequant.
    // Each expert packs once, resident across IOVA domains (ork_weight_domain/domain_advance) keyed by its
    // host-ptr with is_expert=1 (so ork_wcache_evict_experts can reclaim it, dense untouched). Returns before
    // the int8 machinery, so that path stays byte-identical. Mirrors ggml_backend_ork_mul_mat_group_i4.
    if (ctx->qbits == 4 && ctx->hadamard) {
        const int fb = K & (-K);                                  // FWHT block = largest pow2 dividing K
        std::unordered_map<int, std::vector<std::pair<int,int>>> buckets;
        for (int t = 0; t < n_tokens; t++) {
            const int32_t * idp = (const int32_t *)((const char *) ids->data + (size_t) t * ids->nb[1]);
            for (int j = 0; j < n_used; j++) buckets[idp[j]].push_back(std::make_pair(t, j));
        }
        std::vector<int8_t> bi, ai; std::vector<int32_t> ci; std::vector<float> f32e, as;
        // #54 RESIDENT MULTI-DOMAIN (NOT streaming): experts pack ONCE into their layer's IOMMU domain
        // (ork_weight_domain byte-balances layers across the auto-sized n_domains) and stay resident zero-copy —
        // NO per-layer evict/re-import. The auto-sizer sizes n_domains to the full ~15.6 GiB int4 footprint so
        // the whole model lives across the domains at once. (Legacy ORK_MOE_STREAM_MB eviction removed — it forced
        // 19 GiB through one domain, the wrong regime for this NPU.)
        // #54: pass 1 builds each M>=2 expert's int4 activation into a PERSISTENT slot + gathers a task; a
        // per-expert ork_i4_mm_run submits each through the doorbell; pass 2 dequants. reserve() so push_back
        // never reallocs (task pointers stay valid). All same domain (ork_weight_domain is per-layer deterministic).
        const int n_bkt = (int) buckets.size();
        std::vector<std::vector<int8_t>>  co_ai;  co_ai.reserve(n_bkt);
        std::vector<std::vector<int32_t>> co_ci;  co_ci.reserve(n_bkt);
        std::vector<std::vector<float>>   co_as;  co_as.reserve(n_bkt);
        std::vector<std::vector<std::pair<int,int>>> co_rows; co_rows.reserve(n_bkt);
        std::vector<const float*> co_bs; co_bs.reserve(n_bkt);
        std::vector<int> co_Mr; co_Mr.reserve(n_bkt);
        std::vector<ork_mm_task_i4> co_tasks; co_tasks.reserve(n_bkt);
        for (auto & kv : buckets) {
            const int e = kv.first; auto & rows = kv.second;
            // PREFILL-ONLY on the NPU: route M=1 expert buckets (decode, or a rarely-routed prefill expert) to
            // the CPU. Matches the design (decode runs on CPU via the mmap'd weights) and keeps the int4 NPU
            // path to the M>=2 BCHAIN case only — the M=1 per-row int4 path is not used.
            if ((int) rows.size() == 1) {
                const char * x = (const char *) src0->data + (size_t) e * src0->nb[2];
                std::vector<float> wcol(K);
                const int t = rows[0].first, jj = rows[0].second;
                const float * y = (const float *)((const char *) src1->data + (size_t)(bcast?0:jj)*src1->nb[1] + (size_t) t*src1->nb[2]);
                float * dr = (float *)(dbase + (size_t) jj * dst->nb[1] + (size_t) t * dst->nb[2]);
                for (int n = 0; n < N; n++) {
                    const char * wsrc = x + (size_t) n * src0->nb[1];
                    if (type == GGML_TYPE_F32) memcpy(wcol.data(), wsrc, (size_t) K*sizeof(float));
                    else                        to_float(wsrc, wcol.data(), K);
                    float acc = 0.f; for (int k = 0; k < K; k++) acc += wcol[k] * y[k]; dr[n] = acc;
                }
                continue;
            }
            const void * key = (const char *) src0->data + (size_t) e * src0->nb[2];
            auto it = ctx->wcache.find(key);
            if (it == ctx->wcache.end()) {                        // load-from-orkpack or cold-pack, resident
                ork_weight ow; ow.gsize = 0; ow.bscale.resize(N); ow.is_expert = true;
                const char * x = (const char *) src0->data + (size_t) e * src0->nb[2];
                // Prefer the orkpack's native-int4 expert bytes (these carry the GPTQ codes when the pack was
                // GPTQ-built); on a miss (no orkpack / not-yet-dumped) cold-pack from src0 (FWHT + int4 RTN).
                const orkpack_entry * pe = nullptr;
                if (ctx->persist_mode == 1 && ctx->persist_map) {
                    auto pit = ctx->persist_idx.find(ork_expert_key(src0->name, e));
                    if (pit != ctx->persist_idx.end() && pit->second.dtype == ORKPACK_DT_I4_NATIVE &&
                        pit->second.K == (uint32_t) K && pit->second.N == (uint32_t) N) pe = &pit->second;
                }
                if (!pe) {                                        // cold quantize the plane -> bi + ow.bscale
                    bi.resize((size_t) K * N); f32e.resize((size_t) N * K);
                    #pragma omp parallel for schedule(static)
                    for (int n = 0; n < N; n++) {
                        float * col = f32e.data() + (size_t) n * K;
                        if (type == GGML_TYPE_F32) memcpy(col, x + (size_t) n*src0->nb[1], (size_t) K*sizeof(float));
                        else                        to_float(x + (size_t) n*src0->nb[1], col, K);
                        for (int o = 0; o < K; o += fb) ork_fwht_norm(col + o, fb);
                        float mx = 1e-9f; for (int k = 0; k < K; k++) { float v = fabsf(col[k]); if (v > mx) mx = v; }
                        float s = mx / 7.0f; ow.bscale[n] = s;
                        for (int k = 0; k < K; k++) { int q = (int) lrintf(col[k] / s);
                            bi[(size_t) k*N + n] = (int8_t) (q > 7 ? 7 : q < -8 ? -8 : q); }
                    }
                }
                // MULTI-DOMAIN resident via IMPORT (bcreate faults at scale on the big int4 set — the load must
                // import). NOTE (#54, open): the int4 RUN reads imported weights in NON-0 domains wrong on the
                // 2nd+ buffer (int8's run reads imported non-0 weights bit-exact — this is an int4-run-specific
                // integration bug still to fix). Per-expert import also saturates IOMMU mappings (~2340/dom) at
                // full scale. ORK_I4_ARENA consolidates to fewer chunks but hit a chunk-boundary read bug.
                int _dom = ork_weight_domain(ctx, (size_t) K * N / 2, ork_layer_of(src0->name));
                ork_npu_set_pack_domain(ctx->npu, _dom);
                auto mk = [&]() -> ork_w * {
                    if (!pe) return ork_i4_mm_pack(ctx->npu, K, N, bi.data());
                    // #54 RESIDENT MULTI-DOMAIN: zero-copy import (page-cache-mapped) across MANY light domains —
                    // domains are cheap (mapped to the page cache, count limit >> 8), so keep weights zero-copy
                    // (NOT bcreate/pinned-copy). The wall is the bcreate run-SCRATCH (pinned pool), fixed separately.
                    return ork_i4_mm_load_import(ctx->npu, K, N, (const char *) ctx->persist_map + pe->blob_off, pe->blob_size);
                };
                ow.w = mk();
                while (!ow.w && (_dom = ork_domain_advance(ctx)) >= 0) ow.w = mk();
                if (ow.w) {
                    if (pe && pe->bscale_n) memcpy(ow.bscale.data(), (const char *) ctx->persist_map + pe->bscale_off, (size_t) pe->bscale_n * sizeof(float));
                    ow.bytes = ork_w_bytes(ow.w); ctx->wcache_bytes += ow.bytes;
                    if (ctx->n_domains > 1 && _dom < 64) ctx->domain_bytes[_dom] += ow.bytes;
                    it = ctx->wcache.emplace(key, std::move(ow)).first;
                } else {                                          // load failed: CPU f32 GEMV fallback (dequant from src0)
                    std::vector<float> wcol(K);
                    for (auto & r : rows) {
                        const int t = r.first, jj = r.second;
                        const float * y = (const float *)((const char *) src1->data + (size_t)(bcast?0:jj)*src1->nb[1] + (size_t) t*src1->nb[2]);
                        float * dr = (float *)(dbase + (size_t) jj * dst->nb[1] + (size_t) t * dst->nb[2]);
                        for (int n = 0; n < N; n++) {
                            const char * wsrc = x + (size_t) n * src0->nb[1];
                            if (type == GGML_TYPE_F32) memcpy(wcol.data(), wsrc, (size_t) K*sizeof(float));
                            else to_float(wsrc, wcol.data(), K);
                            float acc = 0.f; for (int k = 0; k < K; k++) acc += wcol[k] * y[k]; dr[n] = acc;
                        }
                    }
                    continue;
                }
            }
            const ork_weight & ow = it->second;
            const int Mr = (int) rows.size();                     // Mr>=2 here (Mr==1 went to CPU above)
            // #54 M-pad granularity: was hard 32 ("like the dense path"), but MoE experts avg ~16 routed rows,
            // so pad-to-32 DOUBLES the BCHAIN M-groups (NG=ceil(Mp/H), H=8 for gate/up K=2048) → ~1.6x wasted
            // programs. BCHAIN handles any M (partial last group), so pad only to 16 (aligned to both H=8/H=16;
            // halves NG for the common Mr~16). MEASURED (35B pp512, 2026-08-11): 32→5.57, 16→5.38, 8→4.03 t/s
            // — reducing the pad does NOT help end-to-end (routed Mr mostly rounds to 32; the prefill wall isn't
            // the expert-matmul NG count). Pad fixed at the 32 baseline.
            static const int mpad = 32;
            const int Mp = (mpad <= 1) ? Mr : ((Mr + mpad - 1) / mpad) * mpad;
            co_ai.emplace_back((size_t) Mp * K); co_ci.emplace_back((size_t) Mp * N); co_as.emplace_back(Mp);
            std::vector<int8_t> & aiE = co_ai.back(); std::vector<float> & asE = co_as.back();
            #pragma omp parallel for if (Mp >= 16)
            for (int r = 0; r < Mp; r++) {                         // rotate + int4-quant each routed activation
                if (r < Mr) {
                    const int t = rows[r].first, jj = rows[r].second;
                    const float * y = (const float *)((const char *) src1->data + (size_t)(bcast?0:jj)*src1->nb[1] + (size_t) t*src1->nb[2]);
                    float arow[K]; memcpy(arow, y, (size_t) K*sizeof(float));
                    for (int o = 0; o < K; o += fb) ork_fwht_norm(arow + o, fb);
                    float mx = 1e-9f; for (int k = 0; k < K; k++) { float v = fabsf(arow[k]); if (v > mx) mx = v; }
                    float s = mx / 7.0f; asE[r] = s;
                    for (int k = 0; k < K; k++) { int q = (int) lrintf(arow[k] / s);
                        aiE[(size_t) r*K + k] = (int8_t) (q > 7 ? 7 : q < -8 ? -8 : q); }
                } else { memset(aiE.data() + (size_t) r*K, 0, K); asE[r] = 0.0f; }
            }
            ork_mm_task_i4 tk4; tk4.w = ow.w; tk4.M = Mp; tk4.A = aiE.data(); tk4.C = co_ci.back().data();
            co_tasks.push_back(tk4);                               // gather for the coalesced doorbell
            co_rows.push_back(rows); co_bs.push_back(ow.bscale.data()); co_Mr.push_back(Mr);
        }
        // #54 pass 2: ONE coalesced doorbell for ALL this tensor's M>=2 experts (chained across cores), then
        // per-expert fp32 dequant scatter. Replaces the per-expert submit storm (~1152/layer -> ~nc/layer).
        if (!co_tasks.empty()) {
            // #54 DEFAULT = per-expert via ork_i4_mm_run -> run_i4_bchain_db, which rides the PROVEN doorbell
            // recover (mc_recover_resubmit: RESET + re-seed + resubmit missed cores, domain-safe — the same path
            // Per-expert int4 submit (rides mc_recover_resubmit; multi-domain-safe). co_ci[q] feeds the dequant below.
            for (size_t q = 0; q < co_tasks.size(); q++)
                if (ork_i4_mm_run(ctx->npu, co_tasks[q].w, co_tasks[q].M, co_ai[q].data(), co_ci[q].data())) return false;
            for (size_t q = 0; q < co_tasks.size(); q++) {
                const int32_t * ciq = co_ci[q].data(); const float * asq = co_as[q].data(); const float * bs = co_bs[q];
                for (int r = 0; r < co_Mr[q]; r++) {
                    const int t = co_rows[q][r].first, jj = co_rows[q][r].second;
                    float * dr = (float *)(dbase + (size_t) jj * dst->nb[1] + (size_t) t * dst->nb[2]);
                    const float rs = asq[r]; const int32_t * cr = ciq + (size_t) r * N;
                    for (int n = 0; n < N; n++) dr[n] = rs * bs[n] * (float) cr[n];
                }
            }
        }
        if (ctx->profile) { ctx->t_run += ork_now_us() - t0; ctx->n_mm++; }
        return true;
    }

    // pre-quantize each token's input once (per-row int8); MoE broadcasts one input across a token's
    // experts (n_b1==1). Non-broadcast quantizes per (token,slot) in the gather below.
    const double mp0 = ctx->profile ? ork_now_us() : 0;
    std::vector<int8_t> A_all; std::vector<float> as_all;
    if (bcast) {
        A_all.resize((size_t) n_tokens * K); as_all.resize(n_tokens);
        for (int t = 0; t < n_tokens; t++) {
            const float * y = (const float *)((const char *) src1->data + (size_t) t * src1->nb[2]);
            int8_t * ar = A_all.data() + (size_t) t * K;
            float mx=1e-9f; for (int k=0;k<K;k++){ float v=fabsf(y[k]); if(v>mx)mx=v; }
            as_all[t]=mx/127.0f; float ainv=127.0f/mx;
            for (int k=0;k<K;k++){ int qi=(int)lrintf(y[k]*ainv); ar[k]=(int8_t)(qi>127?127:qi<-127?-127:qi); }
        }
    }
    if (ctx->profile) ctx->moe_prequant += ork_now_us() - mp0;

    // GROUP TOKENS BY EXPERT: bucket every (token t, slot j) under expert ids[j,t].
    std::unordered_map<int, std::vector<std::pair<int,int>>> buckets;
    for (int t = 0; t < n_tokens; t++) {
        const int32_t * idp = (const int32_t *)((const char *) ids->data + (size_t) t * ids->nb[1]);
        for (int j = 0; j < n_used; j++) buckets[idp[j]].push_back(std::make_pair(t, j));
    }
    // EXPERIMENT: routing-frequency profile (ORK_MOE_PROFILE_FREQ) — count expert selections per _exps
    // tensor so the mixed-orkpack tier map can be built hottest-first. Cheap; gated to the profile run.
    static const bool freq_prof = getenv("ORK_MOE_PROFILE_FREQ") != nullptr;
    if (freq_prof) {
        const int n_expert = (int) src0->ne[2];
        auto & fv = ctx->moe_freq[src0->name];
        if ((int) fv.size() < n_expert) fv.resize(n_expert, 0);
        for (auto & kv : buckets) if (kv.first >= 0 && kv.first < n_expert) fv[kv.first] += (long) kv.second.size();
    }
    // ===== M1b/M3 STATIC HOT-EXPERT PARTITION =====
    // Per LAYER-TENSOR cap K (ORK_MOE_HOT, default 8). The hottest K experts of THIS `_exps` tensor stay
    // resident on the NPU (LRU within the cap, IOVA freed on evict); every other (cold) expert this step
    // is computed on the CPU (dequant + f32 GEMV, the same math the convert path uses). Residency is thus
    // bounded to K * (#_exps tensors) * per-proj bytes — for LFM2.5 top-8 = ~1.9 GiB, well under 4 GiB —
    // so the IOMMU never EFAULTs (the live-path soft-reset) regardless of how many experts the router hits.
    static const size_t hot_K = getenv("ORK_MOE_HOT") ? (size_t) atoi(getenv("ORK_MOE_HOT")) : 8;
    static const size_t hot_budget = (size_t)((getenv("ORK_MOE_HOT_GIB") ? atof(getenv("ORK_MOE_HOT_GIB")) : 2.5) * 1024.0*1024.0*1024.0);
    // ===== STAGE 1: M-gated ALL-ACTIVE-to-NPU (batched prefill / verify) =====
    // The standalone M-sweep probe (ork-driver tools/moe_batched_probe, commit 3a44272) found the
    // optimal split for M>=8 is ALL active experts to the NPU: each per-expert GEMM amortizes the M=1
    // submit floor over its M_e routed rows while the CPU fused job grows linearly with the batch. So at
    // batch M (a per-expert row count M_e >= batch_minM, default 8) we admit the expert to the NPU
    // REGARDLESS of the hot_K count cap — bounded only by the IOVA budget (hot_budget). This is the
    // "load-cold-once-per-batch, amortized over M" config (a). Below batch_minM (M=1 decode) we keep the
    // hot_K LRU cap (the M=1 regime where per-expert serial submits lose to CPU's fused MoE). When the
    // budget can't hold every active expert, the overflow falls back to CPU (config (b): static-resident
    // hot set on NPU || cold on CPU) — so the SAME code measures both configs, gated by hot_budget.
    static const int    batch_minM = getenv("ORK_MOE_BATCH_MINM") ? atoi(getenv("ORK_MOE_BATCH_MINM")) : 8;
    static const bool   all_active = !(getenv("ORK_MOE_ALL_ACTIVE") && atoi(getenv("ORK_MOE_ALL_ACTIVE")) == 0);
    const void * tbase = src0->data;                          // unique per (layer, projection)
    auto & hotmap = ctx->moe_hot[tbase];

    // COLD-EXPERT CPU GEMV — reuse ggml's NEON-optimized per-row vec_dot kernel (the same one its native
    // CPU MUL_MAT_ID uses) instead of a scalar dequant+triple-loop. For a quantized expert type T, ggml's
    // ggml_get_type_traits_cpu(T)->vec_dot dots a RAW quantized weight row (no full f32 dequant) against an
    // activation pre-quantized to vec_dot_type (e.g. Q8_K for Q4_K), using SDOT/asimddp. We (a) quantize
    // each distinct token's activation once to vec_dot_type, then (b) fan the (expert × output-row) dot
    // products across a thread pool. This is bit-compatible with ggml's own CPU expert GEMV — so cold
    // experts match the CPU MoE reference exactly (no longer the f32-dequant path, but ggml's quant path).
    const struct ggml_type_traits_cpu * tcpu = ggml_get_type_traits_cpu(type);
    const enum ggml_type vdt = (type == GGML_TYPE_F32) ? GGML_TYPE_F32 : tcpu->vec_dot_type;
    const ggml_vec_dot_t vec_dot = tcpu->vec_dot;
    const size_t row_bytes = src0->nb[1];                         // bytes per weight output-row (type T)
    const size_t vdt_row   = ggml_row_size(vdt, K);               // bytes of one K-length activation in vec_dot_type
    const auto * vdt_tt = ggml_get_type_traits_cpu(vdt);
    // Quantize every distinct token's f32 activation to vec_dot_type ONCE (broadcast => one per token).
    std::unordered_map<int, size_t> tok_q;                        // token -> offset (in vdt-rows) into qact
    std::vector<char> qact;
    // Returns the BYTE OFFSET of the quantized activation within qact (not a pointer — qact is resized as
    // more tokens are quantized, which can relocate the buffer; resolve qact.data()+off only after the
    // single-threaded quantize phase is complete).
    auto quant_tok = [&](int t, int jslot) -> size_t {
        const int key = bcast ? t : (t * 100000 + jslot);
        auto qit = tok_q.find(key);
        if (qit != tok_q.end()) return qit->second;
        const size_t off = qact.size(); qact.resize(off + vdt_row);
        const float * y = (const float *)((const char *) src1->data + (size_t)(bcast?0:jslot)*src1->nb[1] + (size_t) t*src1->nb[2]);
        if (vdt == GGML_TYPE_F32) memcpy(qact.data()+off, y, (size_t)K*sizeof(float));
        else vdt_tt->from_float(y, qact.data()+off, K);
        tok_q[key] = off;
        return off;
    };

    // Collect cold (expert, entries) pairs; run them all in one threaded fan-out after the split.
    std::vector<std::pair<int, std::vector<std::pair<int,int>> *>> cold;
    auto cpu_expert = [&](int e, std::vector<std::pair<int,int>> & ent) { cold.push_back({e, &ent}); };

    // #12 FEATURE-DETECT (no env flag): if the loaded orkpack stores THIS tensor's experts as int4/NF4,
    // compute cold experts from the ork-native Bi4 (the single shared format, ork_native_cpu.h) instead of
    // ggml Q4_K. Driven purely by the orkpack (persist_mode==1) + each expert's O4N1 entry (20B hdr
    // {magic,ver,K,N,quant_kind} + bscale[N] f32 + Bi4[K*N/2] channel-contiguous). Activations -> int8.
    const int n_expert_c = (int) src0->ne[2];
    std::vector<const uint8_t*> onib(n_expert_c, nullptr);
    std::vector<const float*>   obsc(n_expert_c, nullptr);
    ork_cpu_fmt ofmt = ORK_CPU_I4; bool ork_native = false;
    static int8_t ork_nf4_lut[16]; static int ork_lut_done = 0;
    if (!ork_lut_done) { for (int i=0;i<16;i++) ork_nf4_lut[i]=(int8_t)lrintf(ORK_NF4_LVL[i]*127.0f); ork_lut_done=1; }
    int8x16_t ork_lutv = vld1q_s8(ork_nf4_lut);
    if (ctx->persist_mode == 1 && ctx->persist_map) {
        auto p0 = ctx->persist_idx.find(ork_expert_key(src0->name, 0));
        if (p0 != ctx->persist_idx.end() && p0->second.dtype == ORKPACK_DT_I4) {
            ork_native = true;
            for (int e = 0; e < n_expert_c; e++) {
                auto pe = ctx->persist_idx.find(ork_expert_key(src0->name, e));
                if (pe == ctx->persist_idx.end()) { ork_native = false; break; }
                const char * blob = (const char *) ctx->persist_map + pe->second.blob_off;
                uint32_t qk; memcpy(&qk, blob + 16, 4);                 // O4N1 hdr: [magic,ver,K,N,quant_kind]
                if (e == 0) ofmt = (qk == 1u) ? ORK_CPU_NF4 : ORK_CPU_I4;   // 1=CODEBOOK_NF4, 0=UNIFORM
                obsc[e] = (const float *)(blob + 20);
                onib[e] = (const uint8_t *)(blob + 20 + (size_t) N * sizeof(float));
            }
        }
    }
    // int8 activation cache for the ork-native path (absmax/127 per token), keyed like tok_q. Built
    // single-threaded in the dispatch loop below (ai8 stable — no resize — before the threaded run).
    std::unordered_map<int,int> tok_i8; std::vector<int8_t> ai8; std::vector<float> ai8sc;
    auto quant_tok_i8 = [&](int t, int jslot) -> int {
        const int key = bcast ? t : (t * 100000 + jslot);
        auto it = tok_i8.find(key); if (it != tok_i8.end()) return it->second;
        const int idx = (int) ai8sc.size(); ai8.resize((size_t)(idx + 1) * K); ai8sc.resize(idx + 1);
        const float * y = (const float *)((const char *) src1->data + (size_t)(bcast?0:jslot)*src1->nb[1] + (size_t) t*src1->nb[2]);
        float mx = 1e-9f; for (int k=0;k<K;k++){ float v=fabsf(y[k]); if(v>mx)mx=v; } float inv = 127.0f/mx;
        int8_t * a = ai8.data() + (size_t) idx * K;
        for (int k=0;k<K;k++){ int q=(int)lrintf(y[k]*inv); a[k]=(int8_t)(q>127?127:q<-127?-127:q); }
        ai8sc[idx] = mx/127.0f; tok_i8[key] = idx; return idx;
    };

    // One cold work-item: output rows [n0,n1) of expert e for (token t, slot j). ork-native path uses the
    // shared Bi4 (int8 activation); else ggml vec_dot on the raw Q4_K row.
    struct cold_item { int e, t, j, n0, n1; size_t qoff; int i8idx; };
    auto run_cold_item = [&](const cold_item & ci) {
        float * dr = (float *)(dbase + (size_t) ci.j * dst->nb[1] + (size_t) ci.t * dst->nb[2]);
        if (ork_native) {
            ork_cpu_w w; w.fmt = ofmt; w.nibble = onib[ci.e]; w.bit4 = nullptr; w.bit5 = nullptr; w.bit6 = nullptr;
            w.i8 = nullptr; w.bscale = obsc[ci.e]; w.nf4_lut = ork_lutv; w.K = K; w.N = N;
            ork_cpu_gemv_m1(&w, ai8.data() + (size_t) ci.i8idx * K, ai8sc[ci.i8idx], dr, ci.n0, ci.n1);
            return;
        }
        const char * xw = (const char *) src0->data + (size_t) ci.e * src0->nb[2];
        const void * qa = qact.data() + ci.qoff;
        for (int n = ci.n0; n < ci.n1; n++)
            vec_dot(K, dr + n, 0, xw + (size_t) n * row_bytes, 0, qa, 0, 1);
    };

    // Make expert e resident in this tensor's hot pool, packing it live on first touch (or loading from
    // the .orkpack if present). Evicts the LRU resident expert (freeing IOVA) when the pool is at cap.
    // Returns the slot, or nullptr if it could not be made resident (caller falls back to CPU).
    // EXPERIMENT #2 (ORK_MOE_INT8_ONLY_RESIDENT): precision-threshold admission. Only experts STORED as
    // int8 (ORKPACK_DT_I8) are eligible for the NPU IOVA hot cache; any sub-int8 tier (int4/NF4/etc.) must
    // inflate to int8 to run on the NPU anyway — it would cost the inflate AND occupy the full int8
    // footprint once resident, earning nothing from the scarce IOVA. So sub-int8 experts ALWAYS route to
    // the CPU GEMV (NF4/int4 LUT-dequant vec_dot) and never compete for the window. Default OFF: when off
    // get_hot keeps the greedy precision-agnostic budget fill (admit int8 first, then inflate lower tiers
    // until ORK_MOE_HOT_GIB is full) so a scarce-int8 model still fills the reclaimed window.
    static const int int8_only = env_enabled("ORK_MOE_INT8_ONLY_RESIDENT");
    // eff_cap = the per-tensor resident count cap for THIS call: hot_K at decode (M<batch_minM), or a
    // large value in batched all-active mode (M>=batch_minM) so the IOVA budget (hot_budget) is the only
    // admission limiter — every active expert that fits the window goes resident on the NPU.
    auto get_hot = [&](int e, size_t eff_cap) -> ggml_backend_ork_context::ork_hot_slot * {
        const void * x = (const char *) src0->data + (size_t) e * src0->nb[2];
        auto it = hotmap.find(x);
        if (it != hotmap.end() && it->second.w) { it->second.last_use = ++ctx->moe_hot_tick; ctx->moe_hot_hits++; return &it->second; }
        // #2 precision gate (BEFORE any eviction/budget commit): if int8-only mode and this expert is
        // stored sub-int8 in the orkpack, refuse residency (caller -> CPU). No-op in live-pack mode (no
        // orkpack: everything packs to int8) and in greedy mode (int8_only off).
        if (int8_only && ctx->persist_mode == 1) {
            auto pit = ctx->persist_idx.find(ork_expert_key(src0->name, e));
            if (pit != ctx->persist_idx.end() && pit->second.dtype != ORKPACK_DT_I8)
                return (ggml_backend_ork_context::ork_hot_slot *) nullptr;
        }
        // Global IOVA budget gate FIRST: the backbone wcache + hot experts share the 4 GiB window. If this
        // expert won't fit under the weight-budget AND evicting our own LRU wouldn't free enough, refuse
        // (route to CPU) — without freeing a resident hot expert or EFAULTing the MEM_CREATE ioctl. (When
        // at per-tensor cap an eviction below frees exactly K*N, so a same-shape swap always fits.)
        const size_t need = (size_t) K * N;
        const bool will_evict = hotmap.size() >= eff_cap;
        const size_t after_evict = ctx->moe_hot_bytes - (will_evict ? need : 0);
        if (after_evict + need > hot_budget) return (ggml_backend_ork_context::ork_hot_slot *) nullptr;
        // admitted: evict LRU within THIS tensor's pool until under the per-tensor count cap
        while (hotmap.size() >= eff_cap) {
            auto lru = hotmap.end();
            for (auto p = hotmap.begin(); p != hotmap.end(); ++p)
                if (lru == hotmap.end() || p->second.last_use < lru->second.last_use) lru = p;
            if (lru == hotmap.end()) break;
            if (lru->second.w) { ork_mm_free(ctx->npu, lru->second.w); ctx->moe_hot_bytes -= (size_t) K * N; }
            hotmap.erase(lru);
        }
        ggml_backend_ork_context::ork_hot_slot slot;
        // .orkpack hit (persist_mode==1): load pre-tiled bytes; else pack live from the dequantized plane.
        const orkpack_entry * pe = nullptr;
        if (ctx->persist_mode == 1) {
            auto pit = ctx->persist_idx.find(ork_expert_key(src0->name, e));
            if (pit != ctx->persist_idx.end() && pit->second.K==(uint32_t)K && pit->second.N==(uint32_t)N &&
                (pit->second.dtype==ORKPACK_DT_I8 || pit->second.dtype==ORKPACK_DT_I4)) pe = &pit->second;
        }
        std::vector<float> bsc(N);
        const double pk0 = ctx->profile ? ork_now_us() : 0;   // [VERIFY] time first-touch pack/load
        if (pe) {
            const char * blob = (const char *) ctx->persist_map + pe->blob_off;
            slot.w = (pe->dtype==ORKPACK_DT_I4) ? ork_i4a8_mm_load(ctx->npu,K,N,blob,pe->blob_size)
                                                : ork_i8_mm_load  (ctx->npu,K,N,blob,pe->blob_size);
            if (!slot.w) return (ggml_backend_ork_context::ork_hot_slot *) nullptr;
            if (pe->dtype==ORKPACK_DT_I4){ const float*b=ork_w_bscale(slot.w); if(b) memcpy(bsc.data(),b,N*sizeof(float)); }
            else memcpy(bsc.data(), (const char*)ctx->persist_map + pe->bscale_off, N*sizeof(float));
            ctx->persist_hits++;
        } else {
            if (ctx->persist_mode==1) ctx->persist_misses++;
            ork_moe_deq_ctx dq = { (const char*)x, (size_t) src0->nb[1], to_float, type==GGML_TYPE_F32 };
            slot.w = ork_i8_mm_pack_dequant(ctx->npu, K, N, ork_moe_deq_row, &dq, bsc.data());
            if (!slot.w) return (ggml_backend_ork_context::ork_hot_slot *) nullptr;
        }
        if (ctx->profile) { ctx->moe_pack += ork_now_us() - pk0; ctx->moe_pack_calls++; }   // [VERIFY]
        slot.bscale = std::move(bsc); slot.key = x; slot.last_use = ++ctx->moe_hot_tick;
        ctx->moe_hot_bytes += (size_t) K * N;
        if (ctx->moe_hot_bytes > ctx->moe_hot_peak) ctx->moe_hot_peak = ctx->moe_hot_bytes;
        ctx->moe_hot_hits++;
        auto res = hotmap.emplace(x, std::move(slot));
        return &res.first->second;
    };

    // M2 change #2 (conformance gate): only this projection's K conforms to the cross-core stream / full-K
    // envelope (K%512==0 && K<=4096) get NPU-resident hot experts; non-conforming shapes (e.g. LFM
    // ffn_down K=1792) route ENTIRELY to the CPU GEMV. The per-task run_i8 K-split fallback on a *loaded*
    // (orkpack) non-conforming weight returns rc=-1 on this SoC and can soft-wedge the NPU, so we never
    // submit those — the conforming gate/up (K=2048) carry the NPU win, down stays on CPU. Opt back in to
    // the old all-K NPU path with ORK_MOE_NPU_ALLK=1 (then the run_chain/run_i8 fallback handles non-conf K).
    const bool conforming_k = (K % 512 == 0 && K <= 4096);
    static const bool allk = getenv("ORK_MOE_NPU_ALLK") != nullptr;
    // PATH B (M>1): a NON-conforming-K expert (e.g. LFM ffn_down K=1792) runs CORRECTLY on the NPU via the
    // per-task run_i8 K-split path when it carries M_e>1 routed rows — the rc=-1 / soft-wedge was M_e==1
    // specific (verified packed AND loaded by tools/moe_m1_probe: down beats CPU at M_e>=8, ~6.7x@8). The
    // submit + weight read amortize across M_e rows, so the M>1 down-proj GEMM itself wins on the NPU.
    // BUT end-to-end (LFM2.5 prefill pp64/128, board 10.3.0.236) this LOSES: admitting down-proj DOUBLES the
    // resident expert footprint, and the ~1.2 GiB usable IOVA already caps the gate/up hot set to a ~16%
    // hit-rate — so adding down EFAULTs (over-admission) and the 84% cold-CPU experts dominate regardless.
    // PATH B verdict: the M>1 GEMM wins in isolation, but the IOVA hit-rate wall (NOT the GEMM) decides
    // end-to-end, and prefill WIDENS the expert footprint vs decode. So default OFF (INT_MAX): admit
    // non-conforming K to the NPU only when M_e >= ORK_MOE_NPU_DOWN_MINM (set e.g. =8 to opt into the
    // validated down-on-NPU path on wider-IOVA HW / fewer-expert models). Conforming K (gate/up) unchanged.
    static const int down_minM = getenv("ORK_MOE_NPU_DOWN_MINM") ? atoi(getenv("ORK_MOE_NPU_DOWN_MINM")) : INT_MAX;

    // Split active experts: those already resident, or that fit/evict into the per-tensor cap -> NPU.
    // The rest this step -> CPU. Prefer keeping already-resident experts (the hot set) on the NPU.
    // STAGE 1: this tensor's max per-expert row count decides the regime. If ANY active expert carries
    // >= batch_minM routed rows we're in the batched (prefill / verify) regime — admit ALL active experts
    // that fit the IOVA budget (eff_cap large), the probe's all-active-to-NPU optimum. Otherwise (decode,
    // M_e<batch_minM) keep the hot_K LRU cap. Default ON when MoE-on-NPU is enabled (ORK_MOE_ALL_ACTIVE=0
    // reverts to the pure hot_K policy for A/B).
    size_t max_Me = 0; for (auto & kv : buckets) if (kv.second.size() > max_Me) max_Me = kv.second.size();
    const bool batched = all_active && ((int) max_Me >= batch_minM);

    // ============================ PATH (b) ============================================
    // Fusion-preserving + concurrent split (ORK_MOE_PATHB=1, default OFF). At the batched (prefill/
    // verify) regime we split this tensor's active experts into an NPU share (run on a DEDICATED thread
    // via the resident-int8 stream, overlapped) and a CPU share computed through ggml's REAL FUSED
    // batched MUL_MAT_ID kernel (a compacted sub-graph on a cached CPU backend — NOT the per-expert
    // vec_dot loop). This fixes BOTH Stage-1 losses: lost CPU fusion + serial NPU submit. Below the
    // threshold (decode), fall through to the existing all-CPU cold path.
    //   ORK_MOE_PATHB_FRAC = fraction of ACTIVE experts to route to the NPU (0..1; 0=all CPU-fused,
    //                        1=all NPU). The f*S experts with the MOST routed rows go to the NPU (best
    //                        submit-floor amortization); the rest stay on the CPU-fused sub-graph.
    static const bool pathb = env_enabled("ORK_MOE_PATHB");
    if (pathb && batched && conforming_k) {
        const double pb_t0 = ork_now_us();
        // --- partition active experts: largest-M_e first to NPU up to frac ---
        std::vector<std::pair<int,int>> act;   // (M_e, expert)
        for (auto & kv : buckets) act.push_back({(int) kv.second.size(), kv.first});
        std::sort(act.begin(), act.end(), [](auto&a, auto&b){ return a.first > b.first; });
        static const double frac = getenv("ORK_MOE_PATHB_FRAC") ? atof(getenv("ORK_MOE_PATHB_FRAC")) : 0.5;
        const int S_act = (int) act.size();
        int n_npu = (int) lrint(frac * S_act);
        if (n_npu < 0) n_npu = 0; if (n_npu > S_act) n_npu = S_act;
        // NPU experts: try to make resident (budget-limited); any that fail to go resident fall to CPU.
        std::vector<int> npu_e; std::vector<ggml_backend_ork_context::ork_hot_slot *> npu_s;
        std::vector<int> cpu_e;
        for (int i = 0; i < S_act; i++) {
            const int e = act[i].second;
            ggml_backend_ork_context::ork_hot_slot * s =
                (i < n_npu) ? get_hot(e, (size_t) -1) : nullptr;
            if (s) { npu_e.push_back(e); npu_s.push_back(s); }
            else   { cpu_e.push_back(e); }
        }
        const int Snpu = (int) npu_e.size();

        // --- NPU side: gather routed rows into bigA, prepare run_stream_i8 tasks (run on a thread) ---
        size_t npu_rows = 0; for (int e : npu_e) npu_rows += buckets[e].size();
        std::vector<int8_t>  bigA((size_t) npu_rows * K);
        std::vector<int32_t> bigC((size_t) npu_rows * N);
        std::vector<float>   as_row(npu_rows ? npu_rows : 1);
        std::vector<ork_mm_task_i8> tasks(Snpu);
        std::vector<size_t> offs(Snpu);
        { size_t off = 0;
          for (int x = 0; x < Snpu; x++) {
            const int e = npu_e[x]; auto & ent = buckets[e]; const int cnt = (int) ent.size();
            int8_t * Ae = bigA.data() + off * K;
            for (int i = 0; i < cnt; i++) {
                const int t = ent[i].first, j = ent[i].second;
                int8_t * ar = Ae + (size_t) i * K;
                if (bcast) { memcpy(ar, A_all.data() + (size_t) t * K, (size_t) K); as_row[off+i] = as_all[t]; }
                else {
                    const float * y = (const float *)((const char *) src1->data + (size_t) j*src1->nb[1] + (size_t) t*src1->nb[2]);
                    float mx=1e-9f; for (int k=0;k<K;k++){ float v=fabsf(y[k]); if(v>mx)mx=v; }
                    as_row[off+i]=mx/127.0f; float ainv=127.0f/mx;
                    for (int k=0;k<K;k++){ int qi=(int)lrintf(y[k]*ainv); ar[k]=(int8_t)(qi>127?127:qi<-127?-127:qi); }
                }
            }
            tasks[x].w = npu_s[x]->w; tasks[x].M = cnt; tasks[x].A = Ae; tasks[x].C = bigC.data() + off * N;
            offs[x] = off; off += cnt;
          }
        }
        // launch the NPU stream on a dedicated thread so it OVERLAPS the CPU sub-graph compute below.
        // run_stream_i8 is blocking (kernel wait); the join is a full memory barrier -> bigC is visible
        // to the combine after t_npu.join() (host memory, no DMA-coherency dance — not a DMA buffer).
        std::atomic<int> npu_rc(0);
        double npu_dt = 0;
        std::thread t_npu;
        if (Snpu > 0) {
            t_npu = std::thread([&](){
                const double n0 = ork_now_us();
                int rc = ork_i8_mm_run_stream(ctx->npu, Snpu, tasks.data());
                if (rc) { rc = 0; for (int x = 0; x < Snpu && rc == 0; x++) rc = ork_i8_mm_run(ctx->npu, tasks[x].w, tasks[x].M, tasks[x].A, tasks[x].C); }
                npu_rc.store(rc);
                npu_dt = ork_now_us() - n0;
            });
        }

        // --- CPU side: ggml's REAL FUSED batched MUL_MAT_ID on a compacted sub-graph ---
        // Build a sub-graph node MUL_MAT_ID(as'=src0, b'=[K,1,P], ids'=[1,P]) where P = total CPU-routed
        // (token,slot) pairs. Column p maps 1:1 to a CPU pair; ids'[0,p]=expert(p). The fused kernel
        // still groups the P columns by expert -> a real batched GEMM per CPU expert. We then scatter
        // dst'[:,0,p] back to dst[:, slot(p), token(p)]. Zero wasted rows (no park-expert redirection).
        // PARK layout (ORK_MOE_PATHB_PARK=1): instead of compacting CPU pairs into P single-expert columns,
        // keep the NATIVE-efficient structure — b' = src1 (n_tokens cols, NO per-slot duplication), ids' =
        // a copy of the real ids [n_used,n_tokens] where NPU-routed (slot,token) entries are redirected to a
        // PARK expert (the most-loaded CPU expert) so the fused kernel batches exactly like native; we then
        // scatter only the CPU-routed dst' slots into dst (the park rows for NPU slots are discarded). Waste
        // = NPU-slot rows computed on the park expert, but the per-token batching matches native (fast).
        static const bool park = env_enabled("ORK_MOE_PATHB_PARK");
        double cpu_dt = 0;
        std::unordered_set<int> cpu_set(cpu_e.begin(), cpu_e.end());
        if (park && !cpu_e.empty()) {
            const double c0 = ork_now_us();
            if (!ctx->cpu_backend) { ctx->cpu_backend = ggml_backend_cpu_init();
                ctx->pathb_cpu_threads = getenv("ORK_MOE_CPU_THREADS") ? atoi(getenv("ORK_MOE_CPU_THREADS")) : 4; }
            ggml_backend_cpu_set_n_threads(ctx->cpu_backend, ctx->pathb_cpu_threads);
            // park expert = most-loaded CPU expert (already gets the most rows -> least relative waste).
            int park_e = cpu_e[0]; { size_t best = 0; for (int e : cpu_e) if (buckets[e].size() > best) { best = buckets[e].size(); park_e = e; } }
            size_t mem = 32 * ggml_tensor_overhead() + 2 * ggml_graph_overhead() + 4096;
            struct ggml_init_params ip = { mem, nullptr, true }; struct ggml_context * gctx = ggml_init(ip);
            // b' = src1 (alias, full n_tokens). ids' = remapped [n_used, n_tokens]. dst' [N, n_used, n_tokens].
            std::vector<int32_t> idP((size_t) n_used * n_tokens);
            for (int t = 0; t < n_tokens; t++) {
                const int32_t * idp = (const int32_t *)((const char *) ids->data + (size_t) t * ids->nb[1]);
                for (int jj = 0; jj < n_used; jj++) { int e = idp[jj]; idP[(size_t) t*n_used + jj] = cpu_set.count(e) ? e : park_e; }
            }
            std::vector<float> dstP((size_t) N * n_used * n_tokens);
            struct ggml_tensor * as_t = nullptr;
            { static const bool use_repack = env_enabled("ORK_MOE_PATHB_REPACK");
              if (use_repack) { auto & rp = ctx->pathb_repack[src0->data];
                if (!rp.t) { size_t mo = ggml_tensor_overhead()+256; struct ggml_init_params rip={mo,nullptr,true};
                    rp.gctx=ggml_init(rip); struct ggml_tensor*rt=ggml_new_tensor_3d(rp.gctx,type,K,N,src0->ne[2]); ggml_set_name(rt,src0->name);
                    rp.buf=ggml_backend_alloc_ctx_tensors_from_buft(rp.gctx,ggml_backend_cpu_repack_buffer_type());
                    if(rp.buf){ggml_backend_tensor_set(rt,src0->data,0,ggml_nbytes(rt));rp.t=rt;}else{ggml_free(rp.gctx);rp.gctx=nullptr;} }
                as_t = rp.t; }
              if (!as_t) { as_t = ggml_new_tensor_3d(gctx, type, K, N, src0->ne[2]);
                as_t->data=src0->data; as_t->nb[0]=src0->nb[0];as_t->nb[1]=src0->nb[1];as_t->nb[2]=src0->nb[2];as_t->nb[3]=src0->nb[3];
                as_t->buffer=src0->buffer; as_t->extra=src0->extra; }
            }
            struct ggml_tensor * b_t = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, K, 1, n_tokens);
            b_t->data = src1->data; b_t->nb[0]=src1->nb[0]; b_t->nb[1]=src1->nb[1]; b_t->nb[2]=src1->nb[2]; b_t->nb[3]=src1->nb[3];
            struct ggml_tensor * id_t = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, n_used, n_tokens);
            id_t->data = idP.data();
            struct ggml_tensor * out = ggml_mul_mat_id(gctx, as_t, b_t, id_t);
            out->data = dstP.data();
            struct ggml_cgraph * gf = ggml_new_graph(gctx); ggml_build_forward_expand(gf, out);
            ggml_backend_graph_compute(ctx->cpu_backend, gf);
            ggml_free(gctx);
            cpu_dt = ork_now_us() - c0;
            // join NPU + combine, then return (park layout has its own combine path)
            if (Snpu > 0) t_npu.join();
            const double cb0 = ork_now_us();
            if (npu_rc.load()) { if (getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK] PATH-B NPU rc=%d\n", npu_rc.load()); return false; }
            for (int x = 0; x < Snpu; x++) { auto & ent = buckets[npu_e[x]]; const int cnt=(int)ent.size(); const size_t o=offs[x];
                const float * bs = npu_s[x]->bscale.data();
                for (int i = 0; i < cnt; i++) { const int t=ent[i].first,j=ent[i].second; const int32_t*cr=bigC.data()+(o+i)*N;
                    float*dr=(float*)(dbase+(size_t)j*dst->nb[1]+(size_t)t*dst->nb[2]); const float as=as_row[o+i];
                    for(int n=0;n<N;n++) dr[n]=as*bs[n]*(float)cr[n]; } }
            // CPU partial: scatter ONLY the CPU-routed slots from dst'[N,n_used,n_tokens] (park rows ignored).
            for (int t = 0; t < n_tokens; t++) { const int32_t*idp=(const int32_t*)((const char*)ids->data+(size_t)t*ids->nb[1]);
                for (int jj = 0; jj < n_used; jj++) { if (!cpu_set.count(idp[jj])) continue;
                    const float*sp=dstP.data()+((size_t)t*n_used+jj)*N;
                    float*dr=(float*)(dbase+(size_t)jj*dst->nb[1]+(size_t)t*dst->nb[2]);
                    memcpy(dr, sp, (size_t)N*sizeof(float)); } }
            const double cb1 = ork_now_us();
            ctx->pathb_calls++; ctx->pathb_npu_t+=npu_dt; ctx->pathb_cpu_t+=cpu_dt;
            ctx->pathb_combine_t+=cb1-cb0; ctx->pathb_wall_t+=cb1-pb_t0;
            ctx->pathb_npu_experts+=Snpu; ctx->pathb_cpu_experts+=(long)cpu_e.size();
            ctx->moe_hot_hits+=Snpu; ctx->moe_cold_cpu+=(long)cpu_e.size();
            if (ctx->profile && ctx->moe_calls < 4)
                fprintf(stderr,"[ORK PATH-B/PARK] K=%d N=%d Snpu=%d cpuE=%d park=%d npu=%.0fus cpu=%.0fus comb=%.0fus wall=%.0fus eff=%.2fx\n",
                    K,N,Snpu,(int)cpu_e.size(),park_e,npu_dt,cpu_dt,cb1-cb0,cb1-pb_t0,(npu_dt+cpu_dt)>0?(npu_dt+cpu_dt)/(cb1-pb_t0):0.0);
            if (ctx->profile){ctx->t_run+=ork_now_us()-t0;ctx->n_mm++;ctx->moe_calls++;}
            return true;
        }

        struct pair_loc { int t, j; };
        std::vector<pair_loc> cpu_pairs;
        if (!cpu_e.empty()) {
            for (int e : cpu_e) { auto & ent = buckets[e]; for (auto & pr : ent) cpu_pairs.push_back({pr.first, pr.second}); }
        }
        const int P = (int) cpu_pairs.size();
        std::vector<float> dstp;        // dst' [N, P] f32
        if (P > 0) {
            const double c0 = ork_now_us();
            if (!ctx->cpu_backend) {
                ctx->cpu_backend = ggml_backend_cpu_init();
                ctx->pathb_cpu_threads = getenv("ORK_MOE_CPU_THREADS") ? atoi(getenv("ORK_MOE_CPU_THREADS")) : 4;
            }
            ggml_backend_cpu_set_n_threads(ctx->cpu_backend, ctx->pathb_cpu_threads);
            // scratch ggml context: metadata only (no_alloc); we wire ->data to our own buffers.
            size_t mem = 32 * ggml_tensor_overhead() + 2 * ggml_graph_overhead() + 4096;
            struct ggml_init_params ip = { mem, nullptr, true };
            struct ggml_context * gctx = ggml_init(ip);
            // b' [K,1,P] f32 activations (gather token columns); ids' [1,P] i32; dst' [N,1,P] f32.
            std::vector<float>   bP((size_t) K * P);
            std::vector<int32_t> idP(P);
            dstp.assign((size_t) N * P, 0.f);
            for (int p = 0; p < P; p++) {
                const int t = cpu_pairs[p].t, j = cpu_pairs[p].j;
                const float * y = (const float *)((const char *) src1->data + (size_t)(bcast?0:j)*src1->nb[1] + (size_t) t*src1->nb[2]);
                memcpy(bP.data() + (size_t) p * K, y, (size_t) K * sizeof(float));
            }
            { int p = 0; for (int e : cpu_e) { auto & ent = buckets[e]; for (size_t i = 0; i < ent.size(); i++) idP[p++] = e; } }
            // as' choice: by default alias src0 directly (standard fused kernel). With ORK_MOE_PATHB_REPACK=1
            // use a cached repack-buffer copy of the full experts tensor so the sub-graph dispatches the SAME
            // REPACKED batched MUL_MAT_ID the native fused-CPU baseline uses — a fair CPU-side fight.
            static const bool use_repack = env_enabled("ORK_MOE_PATHB_REPACK");
            struct ggml_tensor * as_t = nullptr;
            if (use_repack) {
                auto & rp = ctx->pathb_repack[src0->data];
                if (!rp.t) {
                    size_t mo = ggml_tensor_overhead() + 256;
                    struct ggml_init_params rip = { mo, nullptr, true };
                    rp.gctx = ggml_init(rip);
                    struct ggml_tensor * rt = ggml_new_tensor_3d(rp.gctx, type, K, N, src0->ne[2]);
                    ggml_set_name(rt, src0->name);
                    rp.buf = ggml_backend_alloc_ctx_tensors_from_buft(rp.gctx, ggml_backend_cpu_repack_buffer_type());
                    if (rp.buf) {
                        // repack: set_tensor on a repack-buffer tensor triggers the tiling repack from src0.
                        ggml_backend_tensor_set(rt, src0->data, 0, ggml_nbytes(rt));
                        rp.t = rt;
                    } else { ggml_free(rp.gctx); rp.gctx = nullptr; }   // type not repackable -> fall back
                }
                as_t = rp.t;
            }
            if (!as_t) {   // standard path: alias src0 (carry buffer/extra so it dispatches like native).
                as_t = ggml_new_tensor_3d(gctx, type, K, N, src0->ne[2]);
                as_t->data = src0->data; as_t->nb[0]=src0->nb[0]; as_t->nb[1]=src0->nb[1]; as_t->nb[2]=src0->nb[2]; as_t->nb[3]=src0->nb[3];
                as_t->buffer = src0->buffer; as_t->extra = src0->extra;
            }
            struct ggml_tensor * b_t  = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, K, 1, P);
            b_t->data = bP.data();
            struct ggml_tensor * id_t = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, 1, P);
            id_t->data = idP.data();
            struct ggml_tensor * out  = ggml_mul_mat_id(gctx, as_t, b_t, id_t);
            out->data = dstp.data();
            struct ggml_cgraph * gf = ggml_new_graph(gctx);
            ggml_build_forward_expand(gf, out);
            ggml_backend_graph_compute(ctx->cpu_backend, gf);
            ggml_free(gctx);
            cpu_dt = ork_now_us() - c0;
        }

        // --- join NPU + combine (scatter-add both partials into dst) ---
        if (Snpu > 0) t_npu.join();
        const double cb0 = ork_now_us();
        if (npu_rc.load()) { if (getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK] PATH-B NPU rc=%d\n", npu_rc.load()); return false; }
        // NPU partial: bigC[int32] * as_row * bscale -> dst[:, slot, token]
        for (int x = 0; x < Snpu; x++) {
            auto & ent = buckets[npu_e[x]]; const int cnt = (int) ent.size(); const size_t o = offs[x];
            const float * bs = npu_s[x]->bscale.data();
            for (int i = 0; i < cnt; i++) {
                const int t = ent[i].first, j = ent[i].second;
                const int32_t * cr = bigC.data() + (o + i) * N;
                float * dr = (float *)(dbase + (size_t) j * dst->nb[1] + (size_t) t * dst->nb[2]);
                const float as = as_row[o + i];
                for (int n = 0; n < N; n++) dr[n] = as * bs[n] * (float) cr[n];
            }
        }
        // CPU partial: dst'[:,p] -> dst[:, slot(p), token(p)]
        for (int p = 0; p < P; p++) {
            const int t = cpu_pairs[p].t, j = cpu_pairs[p].j;
            const float * sp = dstp.data() + (size_t) p * N;
            float * dr = (float *)(dbase + (size_t) j * dst->nb[1] + (size_t) t * dst->nb[2]);
            memcpy(dr, sp, (size_t) N * sizeof(float));
        }
        const double cb1 = ork_now_us();
        ctx->pathb_calls++; ctx->pathb_npu_t += npu_dt; ctx->pathb_cpu_t += cpu_dt;
        ctx->pathb_combine_t += cb1 - cb0; ctx->pathb_wall_t += cb1 - pb_t0;
        ctx->pathb_npu_experts += Snpu; ctx->pathb_cpu_experts += (long) cpu_e.size();
        ctx->moe_hot_hits += Snpu; ctx->moe_cold_cpu += (long) cpu_e.size();
        if (ctx->profile && ctx->moe_calls < 4)
            fprintf(stderr, "[ORK PATH-B] K=%d N=%d S_act=%d Snpu=%d P=%d npu=%.0fus cpu=%.0fus comb=%.0fus wall=%.0fus eff=%.2fx\n",
                    K, N, S_act, Snpu, P, npu_dt, cpu_dt, cb1-cb0, cb1-pb_t0,
                    (npu_dt+cpu_dt)>0 ? (npu_dt+cpu_dt)/(cb1-pb_t0) : 0.0);
        if (ctx->profile) { ctx->t_run += ork_now_us() - t0; ctx->n_mm++; ctx->moe_calls++; }
        return true;
    }
    // ========================== END PATH (b) ==========================================

    // (get_hot's per-tensor eff_cap removed from this path — the footprint-derived residence mode governs
    //  eviction now, not a per-tensor cap; RESIDENT resolves via ork_resolve_weight_i8, STREAM -> CPU.)
    std::vector<int> hot_e; std::vector<ork_weight *> hot_s;   // ork_weight (wcache) — same .w/.bscale fields as the old hot_slot
    // STEP B — DECODE CPU/NPU SPLIT. At decode (max_Me < batch_minM) route a FRACTION of the active experts
    // to the NON-BLOCKING async doorbell (ork_submit_async, which overlaps run_cold) instead of all-CPU. The
    // blocking #14 rendezvous lost net (profiled); the thread-free doorbell should overlap for free. Knob:
    //   ORK_SPLIT_FRAC (default 0.0) = CPU-only baseline;  0.5 = half the experts to the NPU async share.
    // Prefill (max_Me >= batch_minM) admits all as before; ORK_M1_NPU forces all-decode-on-NPU (old A/B).
    static const bool  m1_npu     = env_enabled("ORK_M1_NPU");
    // ORK_MOE_CPU (task #54, NF4 route): force ALL experts to the CPU cold path (the batched int4/NF4 NEON
    // GEMM) instead of the NPU — the "prefill using only int4 from orkpack on CPU" A/B. The int4 weights stay
    // resident once (the mmap'd orkpack blob the ork-native cold path reads); the NPU IOVA copy is unused.
    const bool         cpu_only   = env_enabled("ORK_MOE_CPU") || (ork_moe_auto() && !env_enabled("ORK_MOE_NPU"));
    static const float split_frac = getenv("ORK_SPLIT_FRAC") ? (float) atof(getenv("ORK_SPLIT_FRAC")) : 0.0f;
    const bool prefill_phase = ((int) max_Me >= batch_minM);
    const int  n_active_e    = (int) buckets.size();
    const int  npu_budget    = prefill_phase ? n_active_e
                             : (m1_npu ? n_active_e : (int) lrintf(split_frac * (float) n_active_e));
    int npu_admitted = 0;
    // STREAM residence (task #54): before resolving THIS _exps tensor's experts, evict prior layers' experts
    // (LRU, experts-only; dense stays resident) so the streamed working set stays bounded (~ORK_MOE_STREAM_MB,
    // default 2 GiB ≈ a couple of layers). The 35B's ~30 GB int8-inflate can't fit resident -> it streams.
    // RESIDENT mode skips this (the whole footprint fits; experts accumulate resident, zero churn).
    if (ctx->residence_stream) {
        static const size_t stream_bytes = (getenv("ORK_MOE_STREAM_MB") ? (size_t) atoll(getenv("ORK_MOE_STREAM_MB")) : 2048) * (size_t) 1024 * 1024;
        ork_wcache_evict_experts(ctx, stream_bytes);
    }
    for (auto & kv : buckets) {
        const int e = kv.first;
        // conforming K always NPU-eligible; non-conforming K only when this expert has enough routed rows
        // (M_e) to amortize the submit (the prefill / batched-verify regime). allk forces all.
        const int M_e = (int) kv.second.size();
        const bool npu_ok = conforming_k || allk || (M_e >= down_minM);
        const bool npu_phase_ok = (npu_admitted < npu_budget);   // split budget (decode: split_frac share; prefill: all)
        // FOOTPRINT-DERIVED RESIDENCE (task #54). RESIDENT mode: resolve the expert through the SHARED
        // streaming loader (ork_resolve_weight_i8 expert-overload -> zero-copy import + ork_weight_domain +
        // multi-domain + shared wcache), CO-RESIDENT for the chain (allow_evict=false: a chained submit must
        // not evict its own in-flight weights -> UAF; the residence mode, not this call, is the evict policy).
        // STREAM mode (footprint > RAM budget) or a resolve-miss: expert falls to the CPU NF4 path below.
        // get_hot (the old per-expert single-domain hot cache) is no longer on this path (it was the S0 churn).
        ork_weight * s = nullptr;
        if (npu_ok && npu_phase_ok && !cpu_only) {
            auto rit = ork_resolve_weight_i8(ctx, src0, K, N, src0->nb[1], type, to_float, /*allow_evict*/false, /*expert*/e);
            if (rit != ctx->wcache.end() && rit->second.w) { rit->second.is_expert = true; s = &rit->second; }
        }
        if (s) { hot_e.push_back(e); hot_s.push_back(s); npu_admitted++; }
        else   { cpu_expert(e, kv.second); }   // STREAM / non-conforming / resolve-miss -> CPU (NF4 decode kernel)
    }

    // Pre-quantize all needed token activations single-threaded (qact grows; not thread-safe), then build
    // a flat list of (expert, token, slot, row-block) work-items and fan THOSE across the pool. Splitting
    // each expert's N output rows into ROW_BLK-sized blocks keeps every thread busy even when only one or
    // two experts went cold this step (the decode case) — the old per-expert fan-out left N-row parallelism
    // on the table and serialized a lone cold expert's 1792 vec_dot calls onto a single core.
    // #14 swap-hidden pipeline: wrap the CPU cold bulk (ork-native #12 / vec_dot) as a callable so it can
    // run CONCURRENTLY with the NPU hot submit below (the NPU int8 chain hides behind the CPU window).
    auto run_cold = [&]() {
        if (cold.empty()) return;
        const double cd0 = ctx->profile ? ork_now_us() : 0;
        const int n_cold = (int) cold.size();

        // BATCHED int4 cold path (ORK_MOE_COLD_BATCH, default ON for the ork-native I4 orkpack): instead of a
        // per-(token,slot) M=1 GEMV (which re-reads+re-unpacks each expert weight per routed row), gather each
        // cold expert's M_e rows and run ONE ork_cpu_gemm_i4 (4x4-blocked, weight/unpack amortized across
        // rows — the prefill lever; ~2.35x on the expert GEMM, bit-exact per test_i4_gemm). Threads over
        // EXPERTS (one batched call each). Activations pre-quantized single-threaded (quant_tok_i8 resizes ai8,
        // not thread-safe), then the workers only read the stable ai8. NF4/i8/non-native fall to the loop below.
        static const bool cold_batch = !getenv("ORK_MOE_COLD_BATCH") || atoi(getenv("ORK_MOE_COLD_BATCH")) != 0;
        const bool batch_fmt = (ofmt == ORK_CPU_I4 || ofmt == ORK_CPU_NF4);   // batched kernels cover I4 + NF4
        if (getenv("ORK_VERBOSE")) { static int g=0; if(!g){ g=1;
            fprintf(stderr, "[ork COLD-GATE] ork_native=%d ofmt=%d(0=I4,1=NF4) cold_experts=%d cold_batch=%d -> batched=%d\n",
                (int)ork_native, (int)ofmt, (int)cold.size(), (int)cold_batch, (int)(ork_native && batch_fmt && cold_batch)); } }
        // PURPOSE-BUILT DECODE KERNEL (ORK_MOE_DECODE_FAST, default on): at M=1 decode the batched path's
        // per-expert gather (Ag memcpy) + outg alloc/zero + gemm-setup + scatter memcpy is pure overhead with
        // no rows to amortize. Instead: quantize each token's activation ONCE (bcast-cached in quant_tok_i8)
        // and GEMV each expert's NF4/int4 weight DIRECTLY into dst (ork_cpu_gemv_m1, full N). Threads over
        // experts on the A76 cluster. Removes the wrapper; keeps NF4's cheap vqtbl unpack.
        static const bool decode_fast = !getenv("ORK_MOE_DECODE_FAST") || atoi(getenv("ORK_MOE_DECODE_FAST")) != 0;
        if (ork_native && batch_fmt && decode_fast && max_Me == 1) {
            std::vector<int> ci8(cold.size(), 0);                 // one int8-quant per token (bcast-cached)
            for (size_t ce = 0; ce < cold.size(); ce++) { auto & ent = *cold[ce].second;
                if (!ent.empty()) ci8[ce] = quant_tok_i8(ent[0].first, ent[0].second); }
            const int8_t * ai8base = ai8.data();                 // stable now (no more resizes)
            unsigned hw = std::thread::hardware_concurrency();
            int nthr = (int) (getenv("ORK_MOE_COLD_THREADS") ? atoi(getenv("ORK_MOE_COLD_THREADS")) : (hw ? hw/2 : 4));
            if (nthr < 1) nthr = 1; if (nthr > (int) cold.size()) nthr = (int) cold.size();
            const int ncold_i = (int) cold.size();
            // PERSISTENT POOL: OpenMP reuses its thread pool across ops (no per-op std::thread create/join —
            // the overhead that made the earlier std::thread fast-path == the batched path). Process affinity
            // (taskset -c 4-7) already pins these to the A76 cluster. Each iter: direct NF4/int4 GEMV -> dst.
            #pragma omp parallel for schedule(dynamic,1) num_threads(nthr)
            for (int ce = 0; ce < ncold_i; ce++) {
                auto & ent = *cold[ce].second; if (ent.empty()) continue;
                const int e = cold[ce].first; const int t = ent[0].first, j = ent[0].second; const int idx = ci8[ce];
                ork_cpu_w w; memset(&w, 0, sizeof w); w.fmt = ofmt; w.nf4_lut = ork_lutv; w.K = K; w.N = N;
                w.nibble = onib[e]; w.bscale = obsc[e];
                float * dr = (float *)(dbase + (size_t) j * dst->nb[1] + (size_t) t * dst->nb[2]);
                ork_cpu_gemv_m1(&w, ai8base + (size_t) idx * K, ai8sc[idx], dr, 0, N);   // NF4/int4 GEMV -> dst, no gather/scatter
            }
            ctx->moe_cold_cpu += n_cold;
            { static bool df1=false; if(!df1 && getenv("ORK_VERBOSE")){ df1=true; fprintf(stderr, "[ork MoE DECODE-FAST] FIRED (omp pool): %d experts, direct gemv->dst (M=1, one quant/token)\n", n_cold); } }
            if (ctx->profile) { ctx->moe_cold += ork_now_us() - cd0; ctx->moe_cold_calls += n_cold; }
            return;
        }
        if (ork_native && batch_fmt && cold_batch) {
            // pre-quantize every cold (token,slot) activation single-threaded + record its ai8 row per expert.
            std::vector<std::vector<int>> eidx(cold.size());   // eidx[ce] = i8 row index per pair of that expert
            for (size_t ce = 0; ce < cold.size(); ce++) {
                auto & ent = *cold[ce].second; eidx[ce].resize(ent.size());
                for (size_t m = 0; m < ent.size(); m++) eidx[ce][m] = quant_tok_i8(ent[m].first, ent[m].second);
            }
            const int8_t * ai8base = ai8.data();   // stable now (no more resizes)
            unsigned hw = std::thread::hardware_concurrency();
            int nthr = (int) (getenv("ORK_MOE_COLD_THREADS") ? atoi(getenv("ORK_MOE_COLD_THREADS")) : (hw ? hw/2 : 4));
            if (nthr < 1) nthr = 1; if (nthr > (int) cold.size()) nthr = (int) cold.size();
            static const bool cbpin = !env_enabled("ORK_NO_AFFINITY");
            std::atomic<int> nexte(0);
            auto bworker = [&]() {
                if (cbpin) { cpu_set_t s; CPU_ZERO(&s); for (int c = 4; c < 8; c++) CPU_SET(c, &s); sched_setaffinity(0, sizeof s, &s); }
                std::vector<int8_t> Ag; std::vector<float> ascg, outg;
                int ce; while ((ce = nexte.fetch_add(1)) < (int) cold.size()) {
                    const int e = cold[ce].first; auto & ent = *cold[ce].second; const int Me = (int) ent.size();
                    if (Me == 0) continue;
                    ork_cpu_w w; memset(&w, 0, sizeof w);
                    w.fmt = ofmt; w.nibble = onib[e]; w.bscale = obsc[e]; w.nf4_lut = ork_lutv; w.K = K; w.N = N;
                    Ag.resize((size_t) Me * K); ascg.resize(Me); outg.assign((size_t) Me * N, 0.f);
                    for (int m = 0; m < Me; m++) { const int idx = eidx[ce][m];
                        memcpy(Ag.data() + (size_t) m * K, ai8base + (size_t) idx * K, (size_t) K); ascg[m] = ai8sc[idx]; }
                    if (ofmt == ORK_CPU_NF4) ork_cpu_gemm_nf4(&w, Ag.data(), K, ascg.data(), outg.data(), N, Me, 0, N);
                    else                     ork_cpu_gemm_i4 (&w, Ag.data(), K, ascg.data(), outg.data(), N, Me, 0, N);
                    for (int m = 0; m < Me; m++) { const int t = ent[m].first, j = ent[m].second;
                        float * dr = (float *)(dbase + (size_t) j * dst->nb[1] + (size_t) t * dst->nb[2]);
                        memcpy(dr, outg.data() + (size_t) m * N, (size_t) N * sizeof(float)); }
                }
            };
            if (nthr <= 1) { bworker(); }
            else { cpu_set_t saved; bool hs = cbpin && sched_getaffinity(0, sizeof saved, &saved) == 0;
                std::vector<std::thread> th; th.reserve(nthr - 1);
                for (int wi = 0; wi < nthr - 1; wi++) th.emplace_back(bworker);
                bworker(); for (auto & t : th) t.join();
                if (hs) sched_setaffinity(0, sizeof saved, &saved); }
            ctx->moe_cold_cpu += n_cold;
            if (ctx->profile) { ctx->moe_cold += ork_now_us() - cd0; ctx->moe_cold_calls += n_cold;
                if (ctx->moe_calls < 4) fprintf(stderr, "[ork MoE COLD-BATCH i4] %d experts, gemm_i4 batched (weight amortized)\n", n_cold); }
            return;
        }

        std::vector<cold_item> items;
        const int ROW_BLK = 256;
        for (auto & ce : cold) {
            const int e = ce.first;
            for (auto & pr : *ce.second) {
                // ork-native path uses int8 acts (quant_tok_i8); else Q8_K for ggml vec_dot. Single-threaded.
                const size_t qoff  = ork_native ? 0 : quant_tok(pr.first, pr.second);
                const int    i8idx = ork_native ? quant_tok_i8(pr.first, pr.second) : 0;
                for (int n0 = 0; n0 < N; n0 += ROW_BLK)
                    items.push_back(cold_item{ e, pr.first, pr.second, n0, n0+ROW_BLK<N ? n0+ROW_BLK : N, qoff, i8idx });
            }
        }
        const int n_items = (int) items.size();
        unsigned hw = std::thread::hardware_concurrency();
        int nthr = (int) (getenv("ORK_MOE_COLD_THREADS") ? atoi(getenv("ORK_MOE_COLD_THREADS")) : (hw ? hw/2 : 4));
        if (nthr < 1) nthr = 1; if (nthr > n_items) nthr = n_items;
        // (a) FUSED CPU MoE: pin the cold-expert workers to the A76 BIG cluster (RK3588 cores 4-7) so the
        // ork-native CPU GEMV runs on the fast cores like ggml's -t4, instead of scattering onto the slow A55
        // little cores (the measured ork-CPU < ggml-CPU gap). Gated by ORK_NO_AFFINITY (default: pin).
        static const bool cold_pin = !env_enabled("ORK_NO_AFFINITY");
        if (nthr <= 1) {
            for (auto & ci : items) run_cold_item(ci);
        } else {
            cpu_set_t saved; bool have_saved = cold_pin && sched_getaffinity(0, sizeof saved, &saved) == 0;
            std::atomic<int> next(0);
            auto worker = [&]() {
                if (cold_pin) { cpu_set_t s; CPU_ZERO(&s); for (int c = 4; c < 8; c++) CPU_SET(c, &s); sched_setaffinity(0, sizeof s, &s); }
                int i; while ((i = next.fetch_add(1)) < n_items) run_cold_item(items[i]);
            };
            std::vector<std::thread> th; th.reserve(nthr-1);
            for (int w = 0; w < nthr-1; w++) th.emplace_back(worker);
            worker();
            for (auto & t : th) t.join();
            if (have_saved) sched_setaffinity(0, sizeof saved, &saved);   // restore main-thread affinity
        }
        ctx->moe_cold_cpu += n_cold;
        if (ctx->profile) { ctx->moe_cold += ork_now_us() - cd0; ctx->moe_cold_calls += n_cold; }
    };

    if (ctx->profile && ctx->moe_calls < 4) fprintf(stderr, "[ork MoE-DIM] K=%d N=%d S=%d type=%d (chain-envelope K%%512==0&&K<=4096: %s)\n", K, N, (int)hot_e.size(), (int)type, (K%512==0&&K<=4096)?"YES":"no");
    const int S = (int) hot_e.size();
    if (S == 0) { run_cold(); if (ctx->profile) { ctx->t_run += ork_now_us() - t0; ctx->n_mm++; } return true; }

    // Pack the hot experts' routed rows into one chained submit (run_chain_i8; per-task run_i8 fallback
    // for the K=1792 down-proj that sits outside the chain envelope).
    size_t total_rows = 0; for (int e : hot_e) total_rows += buckets[e].size();
    std::vector<int8_t>  bigA((size_t) total_rows * K);
    std::vector<int32_t> bigC((size_t) total_rows * N);
    std::vector<float>   as_row(total_rows);
    std::vector<ork_mm_task_i8> tasks(S);
    std::vector<size_t> offs(S);
    // #14 multi-core NONBLOCK path (ORK_DYN_MC): if every hot expert has exactly one routed row (M==1, the
    // decode case) and K conforms (K%512==0 && K<=4096), fire the int8 share across all 3 cores NONBLOCK via
    // ork_dyn_begin_mc (doorbell rendezvous, thread-free — beats the blocking stream 1.38-2.3x) and overlap
    // run_cold. That requires the output resident, so point tasks[x].C into a reused DMA buffer. Else fall
    // back to the std::thread + blocking-stream path (handles M>1 / non-conforming K via per-task run_i8).
    static const int dyn_mc = getenv("ORK_DYN_MC") ? atoi(getenv("ORK_DYN_MC")) : 0;
    bool dyn_ok = dyn_mc && (K % 512 == 0 && K <= 4096) && (total_rows == (size_t) S);
    // Multi-domain-safe: ork_dyn_begin_mc outputs to its own in-domain per-core scratch and copies back, and
    // requires all experts in one domain (a MoE node's experts share the layer's domain) — it returns NULL
    // (=> fall back) if that's ever violated. So no n_domains gate is needed here.
    int32_t * Cbuf = bigC.data();
    if (dyn_ok) {
        // Lever 1: place the output buffer in the HOT EXPERTS' domain (a node's experts share one — layer-based
        // residence) so begin_mc direct-writes it in place (no copy-back) even in a non-0 domain. ork_dma_alloc
        // uses the pack domain, so set it to the node's domain for the (re)alloc, then RESTORE it (else a later
        // weight resolve/pack would land in the wrong domain).
        int nd = ork_w_domain(hot_s[0]->w); if (nd < 0) nd = 0; if (nd > 15) nd = 15;
        int saved_pd = ork_npu_pack_domain(ctx->npu);
        ork_npu_activate_domain(ctx->npu, nd);   // a non-0 domain buffer must be allocated with THAT domain active
        ork_npu_set_pack_domain(ctx->npu, nd);
        int32_t * d = (int32_t *) ork_dma_grow(ctx->npu, &ctx->dma_moeC[nd], &ctx->dma_moeC_sz[nd], (size_t) total_rows * N * 4);
        ork_npu_set_pack_domain(ctx->npu, saved_pd);
        if (d) Cbuf = d; else dyn_ok = false;
    }
    if (getenv("ORK_VERBOSE")) fprintf(stderr, "[#14] S=%d K=%d N=%d total_rows=%zu dyn_mc=%d Kconform=%d oneRow=%d dyn_ok=%d\n",
        S, K, N, total_rows, dyn_mc, (int)(K%512==0 && K<=4096), (int)(total_rows==(size_t)S), (int)dyn_ok);
    size_t off = 0;
    for (int x = 0; x < S; x++) {
        const int e = hot_e[x]; std::vector<std::pair<int,int>> & ent = buckets[e];
        const int cnt = (int) ent.size();
        int8_t * Ae = bigA.data() + off * K;
        for (int i = 0; i < cnt; i++) {
            const int t = ent[i].first, j = ent[i].second;
            int8_t * ar = Ae + (size_t) i * K;
            if (bcast) { memcpy(ar, A_all.data() + (size_t) t * K, (size_t) K); as_row[off+i] = as_all[t]; }
            else {
                const float * y = (const float *)((const char *) src1->data + (size_t) j*src1->nb[1] + (size_t) t*src1->nb[2]);
                float mx=1e-9f; for (int k=0;k<K;k++){ float v=fabsf(y[k]); if(v>mx)mx=v; }
                as_row[off+i]=mx/127.0f; float ainv=127.0f/mx;
                for (int k=0;k<K;k++){ int qi=(int)lrintf(y[k]*ainv); ar[k]=(int8_t)(qi>127?127:qi<-127?-127:qi); }
            }
        }
        tasks[x].w = hot_s[x]->w; tasks[x].M = cnt; tasks[x].A = Ae; tasks[x].C = Cbuf + off * N;
        offs[x] = off; off += cnt;
    }
    // M2 change #2: dispatch the S independent hot experts via the ASYNC ROUND-ROBIN STREAM
    // (run_stream_i8, ~2.5x cross-core even when weights are resident) instead of the single-core
    // PC-chain. run_stream_i8 rejects any task whose K is outside the full-K envelope (K%512==0 &&
    // K<=4096) with rc=-3, so non-conforming shapes (e.g. LFM down-proj K=1792) fall through to the
    // per-task run_i8 path below — same correctness, just no cross-core dispatch for those.
    // ORK_MOE_STREAM=0 reverts to the chain path (A/B comparison).
    static const bool use_stream = !(getenv("ORK_MOE_STREAM") && atoi(getenv("ORK_MOE_STREAM")) == 0);
    // #14 SWAP-HIDDEN PIPELINE: fire the NPU hot chain on a dedicated thread (its kernel-wait parks the
    // thread, freeing the core) CONCURRENTLY with the CPU cold bulk (run_cold), then join — the NPU int8
    // share hides behind the CPU int4/NF4 window (validated: tools/hybrid_decode_probe ~1.07x @M=1). Uses
    // the chained/stream submit (NOT the batched mcworker path that errno=110's at M=1).
    const double ch0 = ctx->profile ? ork_now_us() : 0;
    int crc;
    if (dyn_ok) {
        if (getenv("ORK_VERBOSE")) fprintf(stderr, "[#14-mc] begin_mc S=%d K=%d N=%d\n", S, K, N);
        // multi-core NONBLOCK: fire the int8 share across 3 cores, run the CPU bulk in parallel, rendezvous
        // on the output doorbells (thread-free). Cbuf is the resident DMA output ork_dyn_end writes back.
        ork_dyn_chain * h = ork_submit_async(ctx, tasks);     // NON-BLOCKING doorbell (NULL if ineligible)
        run_cold();                                            // CPU cold bulk ‖ NPU
        crc = h ? ork_submit_end(h) : ork_submit_sync(ctx, tasks);   // rendezvous, or sync fallback AFTER cold
    } else {
        // #14 SWAP-HIDDEN (default): NPU hot chain on a dedicated thread ‖ CPU cold bulk, then join.
        std::atomic<int> crc_a(0);
        auto npu_submit = [&]() {
            int rc = use_stream ? ork_i8_mm_run_stream(ctx->npu, S, tasks.data())
                                : ork_i8_mm_run_chain (ctx->npu, S, tasks.data());
            if (rc) { rc = 0; for (int x = 0; x < S && rc == 0; x++) rc = ork_i8_mm_run(ctx->npu, tasks[x].w, tasks[x].M, tasks[x].A, tasks[x].C); }
            crc_a.store(rc);
        };
        std::thread t_npu(npu_submit);
        run_cold();                   // CPU cold bulk (ork-native #12) — overlaps the NPU submit
        t_npu.join();
        crc = crc_a.load();
    }
    if (ctx->profile) { ctx->moe_chain += ork_now_us() - ch0; ctx->moe_calls++; ctx->moe_chain_S_sum += S; }
    if (crc) { if(getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK] mul_mat_id partition run FAIL rc=%d S=%d K=%d N=%d\n", crc, S, K, N); return false; }

    const double sc0 = ctx->profile ? ork_now_us() : 0;
    for (int x = 0; x < S; x++) {
        std::vector<std::pair<int,int>> & ent = buckets[hot_e[x]];
        const int cnt = (int) ent.size(); const size_t o = offs[x];
        const float * bs = hot_s[x]->bscale.data();
        for (int i = 0; i < cnt; i++) {
            const int t = ent[i].first, j = ent[i].second;
            const int32_t * cr = Cbuf + (o + i) * N;
            float * dr = (float *)(dbase + (size_t) j * dst->nb[1] + (size_t) t * dst->nb[2]);
            const float as = as_row[o + i];
            for (int n = 0; n < N; n++) dr[n] = as * bs[n] * (float) cr[n];
        }
    }
    if (ctx->profile) { ctx->moe_scatter += ork_now_us() - sc0; ctx->t_run += ork_now_us() - t0; ctx->n_mm++; }
    return true;
}

// ============================================================================
// PPU element-wise / activation ops (ork-driver v0.6.39 on-NPU ops)
// ----------------------------------------------------------------------------
// Standalone on-NPU SDP ops: MUL (SwiGLU silu(gate)*up), ADD (residual),
// SILU/GELU (unary activations). EXPERIMENTAL, default-OFF — this is a
// buffer-less backend, so every offloaded op pays a fp32->(quant)->NPU-submit
// ->(dequant) round-trip plus a scheduler backend-crossing; for memory-bound
// element-wise work that can lose to inline CPU/NEON (the standalone-activation
// finding). Gated so it can be measured and enabled where it wins (large-M
// prefill). Each handler ALWAYS produces a correct result: if the NPU call
// fails it falls back to computing the op on the CPU in-place (never aborts the
// graph). Master gate ORK_PPU_OPS enables the bit-exact fp16 SwiGLU MUL; the
// quality-lossier paths (fp16 residual ADD, int8 SILU/GELU) are opt-in per-op.
//   ORK_PPU_OPS=1   -> fp16 element-wise MUL (SwiGLU multiply; bit-exact fp16)
//   ORK_PPU_ADD=1   -> fp16 residual ADD          (fp16 accumulation caveat)
//   ORK_PPU_SILU=1  -> int8 SILU unary            (int8 activation-quant caveat)
//   ORK_PPU_GELU=1  -> int8 GELU unary            (int8 activation-quant caveat)
//   ORK_PPU_MINM=<n> minimum rows (M) to offload (default 32; keeps M=1 decode
//                    on CPU where the ~365us submit floor dominates).
// ============================================================================
static inline bool ork_ppu_ops_on()  { static const int e = env_enabled("ORK_PPU_OPS");  return e; }
static inline bool ork_ppu_add_on()  { static const int e = env_enabled("ORK_PPU_ADD");   return e; }
static inline bool ork_ppu_silu_on() { static const int e = env_enabled("ORK_PPU_SILU");  return e; }
static inline bool ork_ppu_gelu_on() { static const int e = env_enabled("ORK_PPU_GELU");  return e; }
static inline int  ork_ppu_minm()    { static const int m = getenv("ORK_PPU_MINM") ? atoi(getenv("ORK_PPU_MINM")) : 32; return m; }
// One-time loud warning: measured on RK3588 (Qwen3-1.7B-Q8_0, 2026-07-04) that standalone
// element-wise/activation offload is a NET LOSS and NPU-unstable — it exists as a research/measurement
// hook, NOT a recommended path. Kept default-off; the real on-NPU-activation win is FUSION (activation in
// the matmul output stage), tracked in the ork-driver RE-roadmap (M4.6).
static inline void ork_ppu_warn_once() {
    static bool warned = false;
    if (warned) return; warned = true;
    fprintf(stderr, "[ork] WARNING: ORK_PPU_* standalone element-wise/activation offload is EXPERIMENTAL and "
        "a measured NET LOSS on RK3588 (residual ADD: pp128 157->19 t/s, ~8x slower) — element-wise ops have "
        "no arithmetic intensity to amortize the NPU submit-floor + DMA round-trip vs inline NEON, and per-op "
        "DMA-buffer churn can fragment/exhaust the ~4 GiB IOVA window (IOMMU-wedge risk; mitigated by the "
        "ork-driver IOVA guard -> CPU fallback). Also, modern llama.cpp fuses SwiGLU/GEGLU into GGML_OP_GLU, "
        "so standalone SILU/MUL never even fire on the FFN. The productionizable path is matmul-output-stage "
        "FUSION (ork-driver RE-roadmap M4.6), not standalone offload.\n");
}

// SwiGLU element-wise multiply: dst = src0 * src1 (same-shape, contiguous, f32),
// computed on the NPU in fp16 via ork_f16_npu_ewmul. fp16 is bit-exact for this
// op (no lossy quant). Falls back to CPU on any NPU error.
static bool ggml_backend_ork_mul_f16(ggml_backend_ork_context * ctx, struct ggml_tensor * dst) {
    ork_ppu_warn_once();
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const int64_t ne  = ggml_nelements(dst);
    const int     N   = (int) dst->ne[0];
    const int     M   = (int) (ne / N);
    const float * a   = (const float *) src0->data;
    const float * b   = (const float *) src1->data;
    float       * o   = (float *)       dst->data;

    std::vector<ork_f16> ha((size_t) ne), hb((size_t) ne), ho((size_t) ne);
    for (int64_t i = 0; i < ne; i++) { ha[i] = (ork_f16) a[i]; hb[i] = (ork_f16) b[i]; }
    double us = 0;
    int rc = ork_f16_npu_ewmul(ctx->npu, ha.data(), hb.data(), M, N, ho.data(), &us);
    if (rc == 0) {
        for (int64_t i = 0; i < ne; i++) o[i] = (float) ho[i];
        return true;
    }
    // CPU fallback (fp32, exact) — keeps the graph correct if the NPU declined/failed.
    for (int64_t i = 0; i < ne; i++) o[i] = a[i] * b[i];
    return true;
}

// Residual ADD: dst = src0 + src1 (same-shape, contiguous, f32) via ork_f16_npu_add.
static bool ggml_backend_ork_add_f16(ggml_backend_ork_context * ctx, struct ggml_tensor * dst) {
    ork_ppu_warn_once();
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const int64_t ne  = ggml_nelements(dst);
    const int     N   = (int) dst->ne[0];
    const int     M   = (int) (ne / N);
    const float * a   = (const float *) src0->data;
    const float * b   = (const float *) src1->data;
    float       * o   = (float *)       dst->data;

    std::vector<ork_f16> ha((size_t) ne), hb((size_t) ne), ho((size_t) ne);
    for (int64_t i = 0; i < ne; i++) { ha[i] = (ork_f16) a[i]; hb[i] = (ork_f16) b[i]; }
    double us = 0;
    int rc = ork_f16_npu_add(ctx->npu, ha.data(), hb.data(), M, N, ho.data(), &us);
    if (rc == 0) {
        for (int64_t i = 0; i < ne; i++) o[i] = (float) ho[i];
        return true;
    }
    for (int64_t i = 0; i < ne; i++) o[i] = a[i] + b[i];
    return true;
}

// int8 unary activation (SILU or GELU). Dynamic per-tensor abs-max quant; the
// output range of both curves is bounded by ~|in| (silu/gelu(x)->x for large x,
// small bounded undershoot for x<0), so out_scale = in_scale covers it. int8
// (256 levels) is a quality trade — gated, opt-in, CPU fallback on NPU error.
static bool ggml_backend_ork_unary_i8(ggml_backend_ork_context * ctx, struct ggml_tensor * dst, bool gelu) {
    ork_ppu_warn_once();
    const struct ggml_tensor * src0 = dst->src[0];
    const int64_t ne  = ggml_nelements(dst);
    const int     N   = (int) dst->ne[0];
    const int     M   = (int) (ne / N);
    const float * x   = (const float *) src0->data;
    float       * o   = (float *)       dst->data;

    float mx = 1e-9f;
    for (int64_t i = 0; i < ne; i++) { float v = fabsf(x[i]); if (v > mx) mx = v; }
    const double in_scale  = mx / 127.0;
    const double out_scale = in_scale;   // |act(x)| <= ~|x|
    const float  inv = 127.0f / mx;

    std::vector<int8_t> qi((size_t) ne), qo((size_t) ne);
    for (int64_t i = 0; i < ne; i++) {
        float q = x[i] * inv;
        int v = (int) (q + copysignf(0.5f, q));
        qi[i] = (int8_t) (v > 127 ? 127 : v < -127 ? -127 : v);
    }
    double us = 0;
    int rc = gelu ? ork_i8_npu_gelu(ctx->npu, qi.data(), M, N, in_scale, out_scale, qo.data(), &us)
                  : ork_i8_npu_silu(ctx->npu, qi.data(), M, N, in_scale, out_scale, qo.data(), &us);
    if (rc == 0) {
        for (int64_t i = 0; i < ne; i++) o[i] = (float) (qo[i] * out_scale);
        return true;
    }
    // CPU fallback (fp32 curve, exact).
    for (int64_t i = 0; i < ne; i++) {
        float v = x[i];
        o[i] = gelu ? 0.5f * v * (1.0f + tanhf(0.79788456f * (v + 0.044715f * v * v * v)))
                    : v / (1.0f + expf(-v));
    }
    return true;
}

static inline bool ork_ppu_glu_on() { static const int e = env_enabled("ORK_PPU_GLU"); return e; }
// ORK_FFN_CHAIN: round-trip-free on-NPU SwiGLU FFN inner (gate+SiLU int8-out -> up int8-out -> ewmul int8
// -> down), int8 intermediates never touching fp32. Requires PER-TENSOR activation+weight scales (the fused
// SiLU output stage applies a single scalar R; it cannot express the model's per-row/per-channel scales) —
// an accuracy concession, gated off by default. Also flips GLU support on so the 4 nodes land on ork together.
static inline bool ork_ffn_chain_on() { static const int e = env_enabled("ORK_FFN_CHAIN"); return e; }

// Skip ggml no-op view/reshape nodes when scanning for the next compute node in a pattern.
static inline bool ork_is_noop_node(const struct ggml_tensor * n) {
    return n->op == GGML_OP_NONE || n->op == GGML_OP_RESHAPE || n->op == GGML_OP_VIEW ||
           n->op == GGML_OP_PERMUTE || n->op == GGML_OP_TRANSPOSE;
}

// Detect the fused-SwiGLU FFN inner starting at cgraph->nodes[i]:
//   gate = MUL_MAT(ffn_gate.*, x) ; up = MUL_MAT(ffn_up.*, x) ; glu = GLU_SWIGLU(gate, up) ; down = MUL_MAT(ffn_down.*, glu)
// within a small window (tolerating interleaved no-op nodes). On match fills the four node ptrs and *last_idx
// = the highest cgraph index consumed; returns true. Pure structural check — no compute.
static bool ork_ffn_chain_match(struct ggml_cgraph * cg, int i,
                                struct ggml_tensor ** g, struct ggml_tensor ** u,
                                struct ggml_tensor ** gl, struct ggml_tensor ** dn, int * last_idx) {
    struct ggml_tensor * gate = cg->nodes[i];
    if (gate->op != GGML_OP_MUL_MAT || gate->ne[2] != 1 || gate->ne[3] != 1) return false;
    if (!gate->src[0] || !strstr(gate->src[0]->name, "ffn_gate")) return false;
    // llama.cpp's build_ffn emits ffn_up BEFORE ffn_gate (llama-graph.cpp: `tmp = build_lora_mm(up, cur)`
    // precedes the gate matmul), so a forward-only scan from the gate never finds up and the whole match
    // fails. Instead anchor forward on the SWIGLU GLU node (created after BOTH gate and up) and take up
    // straight from glu->src[1]: ggml_swiglu_split(gate, up) => GLU src0=gate, src1=up (not swapped,
    // verified in ggml.c). Position-independent, so it matches regardless of gate/up node ordering.
    struct ggml_tensor *glu = nullptr, *down = nullptr; int id=-1;
    const int WIN = 12;
    for (int j = i + 1; j < cg->n_nodes && j <= i + WIN; j++) {
        struct ggml_tensor * n = cg->nodes[j];
        if (ork_is_noop_node(n)) continue;
        if (!glu && n->op == GGML_OP_GLU && ggml_get_glu_op(n) == GGML_GLU_OP_SWIGLU &&
            n->src[0] == gate) { glu = n; continue; }
        if (glu && !down && n->op == GGML_OP_MUL_MAT && n->src[0] && strstr(n->src[0]->name, "ffn_down") &&
            n->src[1] == glu) { down = n; id = j; break; }
    }
    if (!glu || !down) return false;
    struct ggml_tensor * up = glu->src[1];               // the up MUL_MAT feeding the SwiGLU
    if (!up || up->op != GGML_OP_MUL_MAT || !up->src[0] || !strstr(up->src[0]->name, "ffn_up")) return false;
    if (up->src[1] != gate->src[1]) return false;        // gate and up must share the same ffn input x
    *g=gate; *u=up; *gl=glu; *dn=down; *last_idx=id; return true;
}

// ORK_FFN_CHAIN: resolve a PER-TENSOR (single scalar scale) packed int8 weight for `src0` [K,N], cached in
// ptcache. Dequant each output channel to f32, take the GLOBAL abs-max -> one scale, quantize int8 in the
// bi[k*N+n] layout ork_i8_mm_pack expects. Simpler than ork_resolve_weight_i8 (no stream/orkpack/domain
// machinery) — this is an experimental gated path. Returns nullptr on failure.
static ggml_backend_ork_context::ork_pt_weight *
ork_resolve_pt_weight(ggml_backend_ork_context * ctx, const struct ggml_tensor * src0) {
    const void * key = src0->data;
    auto it = ctx->ptcache.find(key);
    if (it != ctx->ptcache.end()) return &it->second;
    const int K = (int) src0->ne[0], N = (int) src0->ne[1];
    const size_t nb01 = src0->nb[1];
    const enum ggml_type type = src0->type;
    const auto * tt = ggml_get_type_traits(type);
    ggml_to_float_t to_float = tt->to_float;
    std::vector<float> f32((size_t) N * K);
    float mx = 1e-9f;
    for (int n = 0; n < N; n++) {
        float * frow = f32.data() + (size_t) n * K;
        if (type == GGML_TYPE_F32) memcpy(frow, (const char *) src0->data + n*nb01, (size_t) K*sizeof(float));
        else                       to_float((const char *) src0->data + n*nb01, frow, K);
        for (int k = 0; k < K; k++) { float v = fabsf(frow[k]); if (v > mx) mx = v; }
    }
    const float scale = mx / 127.0f, inv = 127.0f / mx;
    std::vector<int8_t> bi((size_t) K * N);
    for (int n = 0; n < N; n++) { const float * frow = f32.data() + (size_t) n * K;
        for (int k = 0; k < K; k++) { int q = (int) lrintf(frow[k] * inv);
            bi[(size_t) k*N + n] = (int8_t) (q > 127 ? 127 : q < -127 ? -127 : q); } }
    int dom = ork_weight_domain(ctx, (size_t) K * N, ork_layer_of(src0->name));
    ork_npu_set_pack_domain(ctx->npu, dom);
    ork_w * w = ork_i8_mm_pack(ctx->npu, K, N, bi.data());
    while (!w && (dom = ork_domain_advance(ctx)) >= 0) w = ork_i8_mm_pack(ctx->npu, K, N, bi.data());
    if (!w) return nullptr;
    auto & e = ctx->ptcache[key]; e.w = w; e.scale = scale;
    if (getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK FFN-CHAIN] pt-pack %s K=%d N=%d scale=%.3e\n", src0->name, K, N, scale);
    return &ctx->ptcache[key];
}

// Pack a PER-TENSOR int8 weight from an fp32 [N*K] plane (row n = K contiguous), with an optional per-input
// -channel smoothing vector s[K] folded in (W'[n,k] = W[n,k]*s[k]). Returns the ork_w* and the scalar scale.
static ork_w * ork_pack_pt_f32(ggml_backend_ork_context * ctx, const float * f32, int K, int N,
                               const float * s, float * out_scale, int layer) {
    float mx = 1e-9f;
    for (int n = 0; n < N; n++) { const float * r = f32 + (size_t) n * K;
        for (int k = 0; k < K; k++) { float v = fabsf(r[k] * (s ? s[k] : 1.0f)); if (v > mx) mx = v; } }
    const float scale = mx / 127.0f, inv = 127.0f / mx;
    std::vector<int8_t> bi((size_t) K * N);
    for (int n = 0; n < N; n++) { const float * r = f32 + (size_t) n * K;
        for (int k = 0; k < K; k++) { int q = (int) lrintf(r[k] * (s ? s[k] : 1.0f) * inv);
            bi[(size_t) k*N + n] = (int8_t) (q > 127 ? 127 : q < -127 ? -127 : q); } }
    // fc.wg REPLACES this layer's per-channel gate (never packed in fused mode) — so it co-resides in the
    // SAME layer's domain via the ordinary weight placement, and is packed as an IMPORT (uniform chunks, like
    // every other weight) so it doesn't fragment the domain's 32-bit IOVA with a native-alloc outlier. No
    // dedicated fc.wg domain, no extra volume, and gate/up/down land in one domain (no per-layer switch).
    const size_t fcwg_bytes = (size_t) 2 * K * N;   // Bb + full-K Bf
    int dom = ork_weight_domain(ctx, fcwg_bytes, layer); ork_npu_set_pack_domain(ctx->npu, dom);
    ork_w * w = ork_i8_mm_pack_import(ctx->npu, K, N, bi.data());
    while (!w && (dom = ork_domain_advance(ctx)) >= 0) { ork_npu_set_pack_domain(ctx->npu, dom);
        w = ork_i8_mm_pack_import(ctx->npu, K, N, bi.data()); }
    if (!w) w = ork_i8_mm_pack(ctx->npu, K, N, bi.data());   // last-resort native (single-domain / import unavailable)
    if (w && dom >= 0 && dom < 64) ctx->domain_bytes[dom] += fcwg_bytes;
    if (w && out_scale) *out_scale = scale;
    return w;
}

// Dequant a ggml weight tensor [K,N] (row n = K) into a f32 [N*K] plane.
static void ork_deq_weight_f32(const struct ggml_tensor * W, int K, int N, std::vector<float> & out) {
    out.resize((size_t) N * K);
    const auto * tt = ggml_get_type_traits(W->type); ggml_to_float_t to_f = tt->to_float;
    for (int n = 0; n < N; n++) {
        if (W->type == GGML_TYPE_F32) memcpy(out.data() + (size_t) n*K, (const char *) W->data + n*W->nb[1], (size_t) K*sizeof(float));
        else                          to_f((const char *) W->data + n*W->nb[1], out.data() + (size_t) n*K, K);
    }
}

// ORK_FFN_CHAIN one-time per-layer prep: SmoothQuant smoothing (migrate x's per-channel outliers into the
// gate/up weights), pack smoothed per-tensor weights, calibrate STATIC per-tensor scales (s_x, s_silu, s_up
// via int32 matmuls on the calibration batch), build the fused-SiLU LUT. Fills fc. Returns false on failure.
// ORK_FFN_F16_JIT helpers ------------------------------------------------------------------------
// Per-output-channel symmetric int8 quant of a row-major [K][N] fp16 weight -> int8[K*N] + bscale[N],
// matching ork_i8_mm_inflate_to_f16 (wf16[k,n] ~= i8[k*N+n]*bscale[n]). Same convention as pack_i8_f32.
static void ork_quant_f16_i8_perchan(const ork_f16 * w, int K, int N,
                                      std::vector<int8_t> & i8, std::vector<float> & bs) {
    i8.resize((size_t) K * N); bs.resize(N);
    for (int n = 0; n < N; n++) {
        float mx = 1e-9f;
        for (int k = 0; k < K; k++) { float v = fabsf((float) w[(size_t) k*N + n]); if (v > mx) mx = v; }
        float scale = mx / 127.0f, inv = 127.0f / mx; bs[n] = scale;
        for (int k = 0; k < K; k++) { int q = (int) lrintf((float) w[(size_t) k*N + n] * inv);
            i8[(size_t) k*N + n] = (int8_t) (q > 127 ? 127 : q < -127 ? -127 : q); }
    }
}
// Fetch (allocate once, then reuse) a shared fp16 scratch weight of shape (K,N). Allocated in a weight
// domain like the resident packs; cached on ctx keyed by (K<<32|N). Returns NULL if every domain is full.
// CO-DOMAIN placement (the ORK_ATTN submit-timeout fix). A scratch is TRANSIENT: unlike a resident weight
// it has no natural home domain, so it must be placed where it costs the LEAST — and the cost that matters
// is not bytes, it is IOMMU DOMAIN SWITCHES. Placing it by byte-balancing (ork_weight_domain) puts it in a
// domain OTHER than the layer's resident int8 weights, so every bmm forces two `switch iommu domain` events
// (into the scratch's domain and back). A switch waits for the outgoing domain's job refcount to reach 0,
// and a nonblock doorbell job whose completion IRQ never fired holds that refcount forever — so the switch
// stalls the NEXT submit for the full 60 s submit timeout (kernel: "commit elapse time: 61.2s" then "job
// timeout"), then ork self-heals with a reset. That is the whole ORK_ATTN collapse: 3 such events x 60 s
// == the entire 186 s pp64 wall (measured). ork-driver already has the reap-at-boundary machinery for this
// (dom_dirty / ork_dom_flush_if_dirty) but it is armed only for int4 doorbell drops.
// So: allocate the scratch in the domain that is ALREADY ACTIVE, and key the cache by (dom,K,N). The bmm
// sits between the same layer's int8 attn_q/k/v and attn_output, which are co-domain by the layer-aligned
// residence rule, so the active domain IS that layer's domain -> the bmm adds ZERO switches and the run
// switches exactly as often as the (0-failure) ORK_ATTN=0 baseline. Cost: one small scratch per (dom,K,N)
// instead of per (K,N) — the attention/GDN shapes are tens of KB, so this is noise.
// Falls back to the old byte-balanced placement + domain_advance only if the co-domain alloc fails.
static ork_w * ork_get_f16_scratch(ggml_backend_ork_context * ctx, int K, int N) {
    const int adom = ork_npu_active_domain(ctx->npu);
    uint64_t key = ((uint64_t) (uint32_t) adom << 48) | ((uint64_t) (uint32_t) (K & 0xffffff) << 24) | (uint32_t) (N & 0xffffff);
    auto it = ctx->f16_scratch.find(key);
    if (it != ctx->f16_scratch.end()) return it->second;
    const int saved = ork_npu_pack_domain(ctx->npu);
    ork_npu_set_pack_domain(ctx->npu, adom);
    ork_w * s = ork_f16_mm_scratch(ctx->npu, K, N);
    if (!s) {   // co-domain alloc failed (that domain's IOVA is full) — fall back to the byte-balanced cursor
        int dom = ork_weight_domain(ctx, (size_t) K * N * 2, -1); ork_npu_set_pack_domain(ctx->npu, dom);
        s = ork_f16_mm_scratch(ctx->npu, K, N);
        while (!s && (dom = ork_domain_advance(ctx)) >= 0) s = ork_f16_mm_scratch(ctx->npu, K, N);
    }
    ork_npu_set_pack_domain(ctx->npu, saved);   // don't leave the pack cursor pointing at the scratch domain
    if (s) ctx->f16_scratch[key] = s;   // cache only success, so a later domain-free retries
    if (getenv("ORK_ATTN_TRACE"))
        fprintf(stderr, "[f16scratch] K=%d N=%d active_dom=%d -> %s dom=%d (cache=%zu)\n", K, N, adom,
                s ? "ok" : "FAIL", s ? ork_w_domain(s) : -1, ctx->f16_scratch.size());
    return s;
}

static bool ork_ffn_prep(ggml_backend_ork_context * ctx, ggml_backend_ork_context::ork_ffn_smooth & fc,
        const struct ggml_tensor * Wg, const struct ggml_tensor * Wu, const struct ggml_tensor * Wd,
        const float * xf, int M, int K, int Nff, int Kd) {
    // HYBRID chain: only the GATE is forced per-tensor (its fused SiLU has a scalar SDP output scale).
    // up + down run the STANDARD per-channel host-dequant path (near-baseline quality) — the RE finding was
    // that RKNN itself does matmul per-channel requant on the host, not in the SDP regcmd. So prep only needs
    // the gate: SmoothQuant the gate input (x outliers -> gate weight), pack the smoothed gate per-tensor,
    // calibrate static s_x + s_silu, build the fused-SiLU LUT.
    // NOTE: do NOT evict the per-channel gate here. Freeing it mid-eval (ork_mm_free) churns the 32-bit
    // domain's IOVA into <chunk-size holes -> a later 16MB chunk import fails to find a CONTIGUOUS range
    // (PRIME_FD_TO_HANDLE ENOMEM at low domain fill, ~904MB/4GB, 24GB RAM free -> NOT memory, fragmentation)
    // -> partial down-proj -> errno 110 -> wedge, intermittently. Instead the domains are sized for the FULL
    // fused footprint (per-channel gate + up + down + per-tensor fc.wg) via the higher ORK_FFN_CHAIN inflation
    // (see the n_domains auto-calc) so both gates coexist with NO eviction/churn -> deterministic fusion.
    std::vector<float> wgf;
    ork_deq_weight_f32(Wg, K, Nff, wgf);
    std::vector<float> cmax(K, 1e-9f), wmax(K, 1e-9f);
    for (int m = 0; m < M; m++) { const float * xr = xf + (size_t) m * K;
        for (int k = 0; k < K; k++) { float v = fabsf(xr[k]); if (v > cmax[k]) cmax[k] = v; } }
    for (int n = 0; n < Nff; n++) { const float * gr = wgf.data() + (size_t) n*K;
        for (int k = 0; k < K; k++) { float a = fabsf(gr[k]); if (a > wmax[k]) wmax[k] = a; } }
    fc.s.resize(K);   // SmoothQuant s[k] = sqrt(cmax)/sqrt(wmax); x'=x/s, Wg'=s*Wg (equivalent)
    for (int k = 0; k < K; k++) { double s = sqrt((double) cmax[k]) / sqrt((double) wmax[k]);
        if (!(s > 1e-4)) s = 1e-4; if (s > 1e4) s = 1e4; fc.s[k] = (float) s; }
    // fc.wg (per-tensor int8 fused gate) feeds ONLY the normal int8-fused-silu handler branch. In GATE_F16
    // mode the handler uses fc.wg_f16 and a chain failure falls through to per-node, so fc.wg is never read —
    // skip packing it (saves ~12MB IOVA per layer, the difference between fitting 28 layers in one 4GiB domain
    // or hitting the IOVA ceiling: fp16 gate + orkpack int8 gate + fc.wg was 3 resident gate copies per layer).
    if (!getenv("ORK_FFN_GATE_F16")) {
        fc.wg = ork_pack_pt_f32(ctx, wgf.data(), K, Nff, fc.s.data(), &fc.sg, ork_layer_of(Wg->name));
        if (!fc.wg) { if (getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK FFN-CHAIN] prep FAIL %s: fc.wg pack (K=%d N=%d dom=%d)\n", Wg->name, K, Nff, ork_weight_domain(ctx,(size_t)K*Nff,ork_layer_of(Wg->name))); return false; }
    }
    float xmx = 1e-9f;   // static s_x from smoothed activations x'=x/s
    for (int m = 0; m < M; m++) { const float * xr = xf + (size_t) m * K;
        for (int k = 0; k < K; k++) { float v = fabsf(xr[k] / fc.s[k]); if (v > xmx) xmx = v; } }
    fc.s_x = xmx / 127.0;
    double gmax_gate = 1e-9;
    // ORK_FFN_SILU_PCT: option-3 (output-scale outlier clamp). Default = max-based s_silu (unchanged). When set
    // (e.g. 99.9), s_silu = that PERCENTILE of |silu| instead of the max, so a few large-gate outlier neurons
    // don't blow up the per-tensor output scale — the bulk small silu values then get much finer int8 (outliers
    // saturate at 127, which is fine since silu(large)≈large is ~linear). One-line lever vs the CPU sprinkle.
    { const int NS = M < 32 ? M : 32; double smax = 1e-9, gm = 1e-9;   // calibrate s_silu + max|gate| (subset)
      const char* pe = getenv("ORK_FFN_SILU_PCT"); double pct = pe ? atof(pe) : 0.0;
      std::vector<float> sabs; if (pct > 0 && pct < 100) sabs.resize((size_t) Nff * NS);
      #pragma omp parallel for reduction(max:smax,gm) if (Nff >= 64)
      for (int n = 0; n < Nff; n++) { const float * gr = wgf.data() + (size_t) n*K;
          for (int m = 0; m < NS; m++) { const float * xr = xf + (size_t) m * K;
              double ag = 0; for (int k = 0; k < K; k++) ag += (double) gr[k]*xr[k];
              double sv = fabs(ag / (1.0 + exp(-ag))); if (sv > smax) smax = sv;
              if (!sabs.empty()) sabs[(size_t) n*NS + m] = (float) sv;
              double av = fabs(ag); if (av > gm) gm = av; } }
      double sref = smax;
      if (!sabs.empty()) { size_t idx = (size_t)(pct/100.0 * (double)(sabs.size()-1));
          std::nth_element(sabs.begin(), sabs.begin()+idx, sabs.end()); sref = sabs[idx]; if (sref < 1e-9) sref = smax; }
      fc.s_silu = sref * 1.15 / 127.0; if (!(fc.s_silu > 0)) fc.s_silu = 1e-6; gmax_gate = gm * 1.2; }
    fc.gmax = gmax_gate;
    ctx->gmax_profile.emplace_back(Wg->name, (float) gmax_gate);   // (a) auto-capture the model's gmax profile
    // DEFAULT (shipped 2026-07-11): CPU fp32 silu for ALL layers. The int8 FUSED silu LUT is broadly
    // incoherent (PPL 55.45 vs 18.39 all-CPU-silu on Qwen3-1.7B; even fixing only the gmax=131 outlier
    // layer recovers 55->24), and CPU silu is ~FREE — prefill is matmul-bound (flat ~90 tok/s from 0 to 9+
    // CPU-silu layers) and decode never uses the chain (M=1 -> per-node int8). So default all-CPU-silu:
    // coherent and model-agnostic (no hand-set threshold). Overrides:
    //   ORK_FFN_SILU_CPU_GMAX=<thr> : CPU silu only for layers with gmax>thr, rest fused [selective A/B]
    //   ORK_FFN_FUSED_SILU=1        : force the old all-fused path (incoherent; A/B only)
    // The fp16-gate modes (ORK_FFN_GATE_F16 / ORK_FFN_F16) drive their own wg_f16 gate path and need
    // silu_cpu OFF so the handler reaches that branch (see the gate dispatch).
    fc.silu_cpu = !(getenv("ORK_FFN_GATE_F16") || getenv("ORK_FFN_F16"));
    if (const char * t = getenv("ORK_FFN_SILU_CPU_GMAX")) fc.silu_cpu = (gmax_gate > atof(t));
    if (getenv("ORK_FFN_FUSED_SILU")) fc.silu_cpu = false;
    // ORK_FFN_SILU_I16[_GMAX]: MIXED-PRECISION gmax gate. With a threshold, high-gmax OUTLIER layers get the
    // on-NPU int16 silu (coherent, all-NPU) and the rest get the int8 FUSED silu (1 submit, fast) — int8
    // mostly + int16 on the worst offenders. Without a threshold, ALL layers -> int16. The int8-fused silu
    // is broadly lossy (PPL 55 all-fused), but the OUTLIERS dominate (fixing blk.2 alone 55->24; top-9
    // ->19.5), so the threshold buys most of the coherence while keeping most layers on the fast fused path.
    if (getenv("ORK_FFN_SILU_I16")) { const char * t = getenv("ORK_FFN_SILU_I16_GMAX");
        if (!t || gmax_gate > atof(t)) { fc.silu_cpu = true; fc.silu_i16 = true; }   // outlier -> int16 (all-NPU, coherent)
        else { fc.silu_cpu = false; fc.silu_i16 = false; } }                          // low-gmax -> int8 FUSED silu (fast)
    // ORK_FFN_SILU_I16_FUSED scale-bridge (used by the fused int16 handler branch): fc.wg int8-mm -> int16-out
    // via set_i16_out feeding the int16 SiLU resident. gate g = acc*s_x*sg; int16 SiLU wants gate_i16 = g/is
    // with is = gmax/32000. So set_i16_out mult/shift encodes s_x*sg/is = s_x*sg*32000/gmax (fixed point,
    // mult<=32767, shift<=31), scaled up to ~[16384,32767] for precision.
    { double is = fc.gmax / 32000.0; if (!(is > 0)) is = 1e-9; fc.gate16_is = is;
      double r = (fc.s_x * (double) fc.sg) / is; int sh = 0; double m = r > 0 ? r : 1e-9;
      while (m < 16384.0 && sh < 31) { m *= 2.0; sh++; }
      long mult = lround(m); if (mult > 32767) mult = 32767; if (mult < 1) mult = 1;
      fc.gate16_mult = (int) mult; fc.gate16_shift = sh; }
    // ORK_FFN_F16_GMAX: PER-LAYER all-fp16 selection. all-fp16 removes the CPU int8 quant/dequant but fp16
    // weights are 2x int8 RAM — ALL layers fp16 overflows one 4GiB domain -> ORK_DOMAINS=2 -> per-submit
    // overhead. Selecting only the MOST-SENSITIVE layers (gate gmax > threshold) keeps the fp16 footprint
    // small enough to fit ONE domain: those layers get precise fp16 (each submit ~3x int8, acceptable) +
    // coherence, the rest stay full-speed int8. ORK_FFN_F16 with no _GMAX = all layers (backward compat).
    const bool f16_env = getenv("ORK_FFN_F16") != nullptr;
    const bool use_f16 = f16_env && (!getenv("ORK_FFN_F16_GMAX") || gmax_gate > atof(getenv("ORK_FFN_F16_GMAX")));
    if (getenv("ORK_VERBOSE") && f16_env) fprintf(stderr, "[ORK FFN-F16] layer %s gmax=%.2f -> %s\n", Wg->name, gmax_gate, use_f16 ? "FP16" : "int8");
    // ORK_FFN_GATE_F16: precise fp16 gate — build the fp16 SiLU LUT for this layer's gate range and pack the
    // gate weight as -S*Wg (fp16). silu output = C*f16_out at fp16 precision (no int8 activation quant).
    if (getenv("ORK_FFN_GATE_F16") || use_f16) {
        fc.lut_f16.resize(1030); double S = 0, R = 0, os = 0;
        if (ork_f16_mm_build_silu_lut(ctx->npu, gmax_gate, fc.lut_f16.data(), &S, &R, &os)) return false;
        fc.f16_out = os;
        // ORK_FFN_F16_JIT: keep weights host-side as int8+bscale (no resident fp16), inflate at run time.
        const bool jit = use_f16 && getenv("ORK_FFN_F16_JIT");
        // gate = raw-Wg fp16 matmul + EXACT CPU silu. Works under full ORK_FFN_F16 (up/down also fp16) AND
        // under gate-only ORK_FFN_GATE_F16 (up/down stay int8 -> fast prefill), since the fused fp16 SiLU LUT
        // is broken (PPL 16742) but the fp16 gate MATMUL is coherent (PPL 18.13). The hybrid handler's
        // wg_f16 block honors fc.f16_cpusilu to pick ork_f16_mm_run (raw fp16) + CPU silu over the fused LUT.
        const bool cpusilu = (use_f16 || getenv("ORK_FFN_GATE_F16")) && getenv("ORK_FFN_F16_CPUSILU");
        fc.f16_cpusilu = cpusilu;
        // N-chunk the fp16 gate: a single [K,Nff] fp16 buffer (Nff*K*2 = 24MB for Nff=6144,K=2048) fails
        // MEM_CREATE (failed to allocate IOVA:-12) when the domain's IOVA is fragmented by the resident
        // orkpack — only <=~12MB contiguous IOVA ranges survive (12.6MB int8 tiles allocate; 16MB+ fail).
        // Cap each chunk at wg_f16_cn cols so K*cn*2 <= ~12.6MB. Each chunk is its own single-tile ork_w.
        int cn = 3072; if (const char * e = getenv("ORK_FFN_GATE_F16_CN")) { cn = atoi(e); }
        cn -= cn % 16; if (cn < 16) cn = 16;                  // fp16 N must be a multiple of 16
        fc.wg_f16_cn = cn;
        std::vector<ork_f16> wgh((size_t) K * cn);            // reused per chunk: -S*Wg cols [n0:n0+cw] in [K][cw] layout
        for (int n0 = 0; n0 < Nff; n0 += cn) {
            int cw = (Nff - n0 < cn) ? (Nff - n0) : cn;
            if (cw % 16) { if (getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK GATE-F16] Nff=%d chunk unaligned at %d (cw=%d)\n", Nff, n0, cw); return false; }
            for (int n = 0; n < cw; n++) for (int k = 0; k < K; k++)
                wgh[(size_t) k*cw + n] = cpusilu ? (ork_f16) wgf[(size_t) (n0+n)*K + k]           // raw Wg (CPU silu)
                                                 : (ork_f16) (-(double) S * wgf[(size_t) (n0+n)*K + k]);  // -S*Wg (dead fused-LUT run path)
            if (jit) {   // host int8 store (no resident fp16); inflated into a shared scratch at run
                fc.jg.emplace_back(); auto & j = fc.jg.back(); j.N = cw;
                ork_quant_f16_i8_perchan(wgh.data(), K, cw, j.i8, j.bs); continue; }
            int dom = ork_weight_domain(ctx, (size_t) K * cw * 2, ork_layer_of(Wg->name)); ork_npu_set_pack_domain(ctx->npu, dom);
            ork_w * ch = ork_f16_mm_pack(ctx->npu, K, cw, wgh.data());
            while (!ch && (dom = ork_domain_advance(ctx)) >= 0) ch = ork_f16_mm_pack(ctx->npu, K, cw, wgh.data());
            if (!ch) { if (getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK GATE-F16] chunk pack FAIL K=%d cw=%d n0=%d\n", K, cw, n0); return false; }
            fc.wg_f16.push_back(ch);
        }
        if (getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK GATE-F16] prep %s: gmax=%.2f S=%.2f out=%.3e chunks=%zu cn=%d\n", Wg->name, gmax_gate, S, os, fc.wg_f16.size(), cn);
        // ORK_FFN_F16: ALL-fp16 path — also pack up (N-chunked, same cn) + down (single) as RAW fp16 so the
        // whole FFN inner runs fp16 with NO int8 activation quant and NO int32->fp32 dequant.
        if (use_f16) {
            std::vector<float> wuf, wdf; ork_deq_weight_f32(Wu, K, Nff, wuf); ork_deq_weight_f32(Wd, Nff, Kd, wdf);
            std::vector<ork_f16> h;
            for (int n0 = 0; n0 < Nff; n0 += cn) {              // up: raw fp16 [K][cw] chunks (plain ork_f16_mm_run)
                int cw = (Nff - n0 < cn) ? (Nff - n0) : cn; h.resize((size_t) K * cw);
                for (int n = 0; n < cw; n++) for (int k = 0; k < K; k++) h[(size_t) k*cw + n] = (ork_f16) wuf[(size_t)(n0+n)*K + k];
                if (jit) { fc.ju.emplace_back(); auto & j = fc.ju.back(); j.N = cw;
                    ork_quant_f16_i8_perchan(h.data(), K, cw, j.i8, j.bs); continue; }
                int dom = ork_weight_domain(ctx, (size_t) K * cw * 2, ork_layer_of(Wu->name)); ork_npu_set_pack_domain(ctx->npu, dom);
                ork_w * ch = ork_f16_mm_pack(ctx->npu, K, cw, h.data());
                while (!ch && (dom = ork_domain_advance(ctx)) >= 0) ch = ork_f16_mm_pack(ctx->npu, K, cw, h.data());
                if (!ch) { if (getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK FFN-F16] up chunk pack FAIL n0=%d\n", n0); return false; }
                fc.wu_f16.push_back(ch);
            }
            h.resize((size_t) Nff * Kd);                        // down: raw fp16 [K=Nff][N=Kd] (deq [Kd][Nff] transposed), K-sliced by pack
            for (int k = 0; k < Nff; k++) for (int n = 0; n < Kd; n++) h[(size_t) k*Kd + n] = (ork_f16) wdf[(size_t) n*Nff + k];
            if (jit) { fc.jd.N = Kd; ork_quant_f16_i8_perchan(h.data(), Nff, Kd, fc.jd.i8, fc.jd.bs); }
            else { int dom = ork_weight_domain(ctx, (size_t) Nff * Kd * 2, ork_layer_of(Wd->name)); ork_npu_set_pack_domain(ctx->npu, dom);
              fc.wd_f16 = ork_f16_mm_pack(ctx->npu, Nff, Kd, h.data());
              while (!fc.wd_f16 && (dom = ork_domain_advance(ctx)) >= 0) fc.wd_f16 = ork_f16_mm_pack(ctx->npu, Nff, Kd, h.data());
              if (!fc.wd_f16) { if (getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK FFN-F16] down pack FAIL\n"); return false; } }
            fc.f16_all = !jit; fc.f16_jit = jit;
            if (getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK FFN-F16] prep %s: %s up chunks=%zu + down ok\n", Wg->name, jit?"JIT-inflate":"all-fp16", jit?fc.ju.size():fc.wu_f16.size());
        }
    } else {
        fc.lut.resize(1030);   // int8 fused-SiLU LUT for (in_g = s_x*s_Wg, s_silu)
        if (ork_mm_silu_build_lut(ctx->npu, fc.s_x * fc.sg, fc.s_silu, 0x4000, 0x10, 0x56391300u, fc.lut.data())) {
            if (getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK FFN-CHAIN] prep FAIL %s: silu LUT build\n", Wg->name); return false; }
    }
    fc.ready = true;
    if (getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK FFN-CHAIN] prep(hybrid) %s: s_x=%.3e s_silu=%.3e sg=%.3e gmax=%.2f silu=%s\n",
        Wg->name, fc.s_x, fc.s_silu, fc.sg, fc.gmax, fc.silu_cpu ? "CPU-fp32" : "int8-fused");
    return true;
    (void) Wu; (void) Wd;   // up/down resolved per-channel in the handler
}

// ORK_FFN_CHAIN handler: run the whole SwiGLU FFN inner as one round-trip-free on-NPU int8 chain.
//   gate:  ork_i8_mm_run_silu (gate matmul + fused SiLU int8-out, per-layer LUT)  -> silu_i8[M,Nff]
//   up:    ork_i8_mm_run_out8 (up matmul int8-out)                                -> up_i8[M,Nff]
//   glu:   ork_i8_npu_ewmul   (silu_i8 * up_i8 / 128)                             -> glu_i8[M,Nff]
//   down:  ork_i8_mm_run      (down matmul, int8-in)                              -> down_i32[M,Kd]
// then dequant down -> dst fp32. PER-TENSOR scales throughout (the fused SiLU stage's scalar R can't carry
// per-row/per-channel). CPU fp32 fallback (recompute the whole inner) on ANY error, so output is always valid.
// Grow a cached NPU-coherent DMA scratch buffer to at least `need` bytes (alloc-once / realloc-on-growth).
// Returns the buffer (registered in ork-driver's zero-copy table so dma_find resolves it), or nullptr.
static void * ork_dma_grow(ork_npu * npu, void ** buf, size_t * sz, size_t need) {
    if (*buf && *sz >= need) return *buf;
    if (*buf) ork_dma_free(npu, *buf);
    *buf = ork_dma_alloc(npu, need);
    *sz  = *buf ? need : 0;
    return *buf;
}

// ratio r -> fixed-point (mult>>shift), mult kept precise in [2^13, 2^15).
static inline void ork_ratio_to_ms(double r, int * mult, int * shift) {
    if (r <= 0) { *mult = 1; *shift = 14; return; }
    int sh = 14; double m = r * (double)(1 << sh);
    while (m >= 32767.0 && sh > 0)  { sh--; m = r * (double)(1 << sh); }
    while (m <  8192.0  && sh < 30) { sh++; m = r * (double)(1 << sh); }
    int mi = (int) lrint(m); if (mi > 32767) mi = 32767; if (mi < 1) mi = 1;
    *mult = mi; *shift = sh;
}

// ORK_FFN_DEC: run the DECODE (M==1) SwiGLU FFN inner on the NPU against the orkpack's ALREADY-RESIDENT
// PER-CHANNEL int8 weights (ork_resolve_weight_i8 — NO second per-tensor copy, so no double-pack, no IOVA
// wedge, and full per-channel quality; the weights are SHARED with the prefill path). gate+up batch into ONE
// run_chain_i8 submit (shared input x, K<=4096); SiLU/GLU run on the host (fp32, per-channel dequant via
// bscale[n]); down is one wide-K ork_i8_mm_run submit. 2 NPU submits/layer. Falls through on any miss.
static bool ggml_backend_ork_ffn_decode_orkd(ggml_backend_ork_context * ctx,
        struct ggml_tensor * gate_n, struct ggml_tensor * up_n,
        struct ggml_tensor * glu_n, struct ggml_tensor * down_n) {
    (void) glu_n;
    const struct ggml_tensor * Wg = gate_n->src[0], * Wu = up_n->src[0], * Wd = down_n->src[0];
    const struct ggml_tensor * x  = gate_n->src[1];
    const int K = (int) Wg->ne[0], Nff = (int) Wg->ne[1], Kd = (int) Wd->ne[1], M = (int) x->ne[1];
    if (M != 1 || x->type != GGML_TYPE_F32) return false;              // decode only, fp32 activation
    if (K % 32 || Nff % 32 || Kd % 16 || K > 4096) return false;       // gate/up chain tile (K<=4096); down wide-K via run_i8
    // resolve the 3 per-channel weights from the orkpack wcache (resident is_orkd; shared with prefill -> no
    // extra IOVA). A stream-pool tier entry (w==null) can't feed the chain -> fall through to per-node.
    auto tof = [](const struct ggml_tensor * W){ return ggml_get_type_traits(W->type)->to_float; };
    auto itg = ork_resolve_weight_i8(ctx, Wg, K, Nff, Wg->nb[1], Wg->type, tof(Wg), true);
    auto itu = ork_resolve_weight_i8(ctx, Wu, K, Nff, Wu->nb[1], Wu->type, tof(Wu), true);
    auto itd = ork_resolve_weight_i8(ctx, Wd, Nff, Kd, Wd->nb[1], Wd->type, tof(Wd), true);
    if (itg==ctx->wcache.end() || itu==ctx->wcache.end() || itd==ctx->wcache.end()) return false;
    ork_w *wg=itg->second.w, *wu=itu->second.w, *wd=itd->second.w;
    if (!wg || !wu || !wd) return false;
    const float *bsg=itg->second.bscale.data(), *bsu=itu->second.bscale.data(), *bsd=itd->second.bscale.data();
    // ORK_FFN_PROF: TRUE per-hop timers + bytes moved -> achieved weight-DMA GB/s. The decode wall is
    // bandwidth: gb/s here vs the LPDDR4X peak (~30) says whether we're at the ceiling or moving data
    // suboptimally (K-slice re-reads, socket copies). run() time = socket + submit(5us) + HW + return.
    static int fprof=-1; if(fprof<0) fprof=getenv("ORK_FFN_PROF")?1:0;
    static double p_q=0,p_gu=0,p_host=0,p_dn=0,p_deq=0; static long p_n=0; double _t;
    const float * xf = (const float *) x->data;
    float amx=1e-9f; for (int k=0;k<K;k++){ float a=fabsf(xf[k]); if(a>amx)amx=a; }
    const float a_scale=amx/127.0f, ainv=127.0f/amx;
    int8_t  *xi  =(int8_t*) malloc((size_t)K);
    int32_t *gi  =(int32_t*)malloc((size_t)Nff*4), *ui=(int32_t*)malloc((size_t)Nff*4);
    int8_t  *glu8=(int8_t*) malloc((size_t)Nff);
    int32_t *di  =(int32_t*)malloc((size_t)Kd*4);
    float   *glf =(float*)  malloc((size_t)Nff*4);
    bool ok=false;
    if (!xi||!gi||!ui||!glu8||!di||!glf) goto done;
    _t=fprof?ork_now_us():0;
    for (int k=0;k<K;k++){ int q=(int)lrintf(xf[k]*ainv); xi[k]=(int8_t)(q>127?127:q<-127?-127:q); }
    if(fprof){ p_q+=ork_now_us()-_t; _t=ork_now_us(); }
    { ork_mm_task_i8 t[2]={{wg,1,xi,gi},{wu,1,xi,ui}}; if (ork_i8_mm_run_chain(ctx->npu,2,t)) goto done; }  // gate+up: ONE submit
    if(fprof){ p_gu+=ork_now_us()-_t; _t=ork_now_us(); }
    { float gmax=1e-9f;                                                                                   // host: per-channel dequant + SiLU*up
      for (int n=0;n<Nff;n++){ float g=(float)gi[n]*a_scale*bsg[n], u=(float)ui[n]*a_scale*bsu[n];
          float v=(g/(1.0f+expf(-g)))*u; glf[n]=v; float av=fabsf(v); if(av>gmax)gmax=av; }
      const float gs=gmax/127.0f, ginv=127.0f/gmax;
      for (int n=0;n<Nff;n++){ int q=(int)lrintf(glf[n]*ginv); glu8[n]=(int8_t)(q>127?127:q<-127?-127:q); }
      if(fprof){ p_host+=ork_now_us()-_t; _t=ork_now_us(); }
      if (ork_i8_mm_run(ctx->npu, wd, 1, glu8, di)) goto done;                                            // down: ONE submit (wide-K)
      if(fprof){ p_dn+=ork_now_us()-_t; _t=ork_now_us(); }
      float * dst=(float*)down_n->data;
      for (int j=0;j<Kd;j++) dst[j]=(float)((double)di[j]*gs*bsd[j]);
      if(fprof) p_deq+=ork_now_us()-_t;
    }
    ok=true;
    if(fprof){ p_n++; if(p_n%64==0){ double T=p_q+p_gu+p_host+p_dn+p_deq; if(T<1)T=1;
        double gub=(double)2*K*Nff, dnb=(double)Nff*Kd;   // resident weight bytes streamed (int8)
        fprintf(stderr,"[ork-ffn-PROF] calls=%ld | Qquant %.0fus Gate+Up %.0fus(%.1fGB/s) host-silu %.0fus Down %.0fus(%.1fGB/s) deq %.0fus | %.0fus/layer\n",
            p_n, p_q/p_n, p_gu/p_n, gub/(p_gu/p_n*1e3), p_host/p_n, p_dn/p_n, dnb/(p_dn/p_n*1e3), p_deq/p_n, T/p_n); } }
    { static long n=0; if((n++%256)==0) fprintf(stderr,"[ork-ffn-dec] per-channel decode FFN on NPU (call %ld): K=%d Nff=%d Kd=%d\n", n,K,Nff,Kd); }
done:
    free(xi);free(gi);free(ui);free(glu8);free(di);free(glf);
    return ok;
}

// ==== Inc 1: work-stealing CPU pull-queue for the FFN silu/glu lane ===================================
// Persistent workers pull row-tiles via an atomic counter (NO per-tile barrier -> little A55 cores contribute
// their ~1/3 share without dragging, unlike a barrier threadpool; a slow puller just pulls fewer tiles).
// Workers SLEEP on a CV between jobs (no idle spin to fight the doorbell); only the final join spins briefly.
// Core set from ORK_FFN_POOL_CORES ("4,5,6"=3 big; "0,1,2,3,4,5,6"=big+little, reserve one big for the caller +
// doorbell drain). Unset => serial (pool disabled). The NPU keeps its doorbell spinner (on the calling thread).
struct OrkFfnPool {
    std::vector<std::thread> ths;
    std::mutex m; std::condition_variable cv;
    std::deque<std::function<void()>> q; std::atomic<int> pending{0}; bool stop = false;
    void worker() {
        for (;;) {
            std::function<void()> job;
            { std::unique_lock<std::mutex> lk(m); cv.wait(lk, [&]{ return stop || !q.empty(); }); if (stop) return; job = std::move(q.front()); q.pop_front(); }
            job(); pending.fetch_sub(1, std::memory_order_release);
        }
    }
    void start(const std::vector<int> & cores) {
        for (int c : cores) {
            ths.emplace_back([this]{ worker(); });
            cpu_set_t s; CPU_ZERO(&s); CPU_SET(c, &s);
            pthread_setaffinity_np(ths.back().native_handle(), sizeof s, &s);
        }
    }
    bool empty() const { return ths.empty(); }
    // async: enqueue one task; workers pull it. Non-blocking (used by the scheduler while the caller polls the doorbell).
    void submit(std::function<void()> f) { pending.fetch_add(1, std::memory_order_relaxed); { std::lock_guard<std::mutex> lk(m); q.push_back(std::move(f)); } cv.notify_one(); }
    void wait() { while (pending.load(std::memory_order_acquire) > 0) sched_yield(); }
    // blocking parallel-for over [0,nt); the calling thread participates (drains the deque too).
    void run(int nt, const std::function<void(int)> & f) {
        if (ths.empty() || nt <= 1) { for (int t = 0; t < nt; t++) f(t); return; }
        for (int t = 0; t < nt; t++) submit([&f, t]{ f(t); });
        for (;;) { std::function<void()> job;   // calling thread participates
            { std::lock_guard<std::mutex> lk(m); if (q.empty()) break; job = std::move(q.front()); q.pop_front(); }
            job(); pending.fetch_sub(1, std::memory_order_release); }
        wait();
    }
};
static OrkFfnPool * ork_ffn_pool() {
    static OrkFfnPool * p = nullptr; static bool init = false;
    if (!init) { init = true;
        if (const char * e = getenv("ORK_FFN_POOL_CORES")) {
            std::vector<int> cores; for (const char * s = e; *s; ) { cores.push_back(atoi(s)); while (*s && *s != ',') s++; if (*s == ',') s++; }
            if (!cores.empty()) { p = new OrkFfnPool(); p->start(cores);
                if (getenv("ORK_VERBOSE")) { fprintf(stderr, "[FFN POOL] started %zu workers on cores:", cores.size()); for (int c : cores) fprintf(stderr, " %d", c); fprintf(stderr, "\n"); } }
        }
    }
    return p;
}

// Dedicated NPU-DRIVER (spinner) thread: owns the single-stream NPU. Pulls a doorbell chain, arms it, and
// SPINS in ork_dyn_end driving/collecting it — so the NPU stays continuously driven even while OTHER cores do
// silu. Fixes the ORK_FFN_SCHED starvation (there the calling thread left the doorbell to do silu, so the
// chain wasn't advanced). Pinned to a big core (ORK_FFN_DRIVER_CORE, default 7). Only THIS thread touches the
// NPU during a chain (single-stream). begin_mc NULL => per-task blocking fallback (still on the driver thread).
struct OrkNpuDriver {
    std::thread th; std::mutex m; std::condition_variable cv;
    ork_npu * npu = nullptr;
    const ork_mm_task_i8 * tasks = nullptr; int ntasks = 0; int K = 0; int nc = 0;   // nc: doorbell colsplit core count (0=all)
    std::atomic<long> job{0}, done{0}; std::atomic<int> rc{0}; bool stop = false;
    void driver() {
        long seen = 0;
        for (;;) {
            const ork_mm_task_i8 * tk; int n, kk;
            { std::unique_lock<std::mutex> lk(m); cv.wait(lk, [&]{ return stop || job.load() != seen; }); if (stop) return; seen = job.load(); tk = tasks; n = ntasks; kk = K; }
            int r = 0;
            ork_dyn_chain * h = ork_dyn_begin_mc(npu, n, tk, nc);
            if (h) { if (ork_dyn_end(h) < 0) r = -1; }
            else for (int i = 0; i < n && r == 0; i++) if (ork_i8_mm_run(npu, tk[i].w, tk[i].M, tk[i].A, tk[i].C)) r = -1;   // fallback (Bf missing / ineligible)
            (void) kk; rc.store(r); done.store(seen, std::memory_order_release);
        }
    }
    void start(ork_npu * n, int core) { npu = n; th = std::thread([this]{ driver(); });
        cpu_set_t s; CPU_ZERO(&s); CPU_SET(core, &s); pthread_setaffinity_np(th.native_handle(), sizeof s, &s); }
    void run_async(const ork_mm_task_i8 * tk, int n) { { std::lock_guard<std::mutex> lk(m); tasks = tk; ntasks = n; job.fetch_add(1); } cv.notify_one(); }
    int wait() { long g = job.load(); while (done.load(std::memory_order_acquire) != g) sched_yield(); return rc.load(); }
};
static OrkNpuDriver * ork_npu_driver(ork_npu * npu) {
    static OrkNpuDriver * d = nullptr; static bool init = false;
    if (!init) { init = true;
        if (getenv("ORK_FFN_DRIVER")) { d = new OrkNpuDriver();
            int core = getenv("ORK_FFN_DRIVER_CORE") ? atoi(getenv("ORK_FFN_DRIVER_CORE")) : 7;
            d->nc = getenv("ORK_FFN_DRIVER_NC") ? atoi(getenv("ORK_FFN_DRIVER_NC")) : 0;   // doorbell colsplit cores (0=all -> greedy; 1..3 leaves big cores for silu)
            d->start(npu, core);
            if (getenv("ORK_VERBOSE")) fprintf(stderr, "[FFN DRIVER] dedicated NPU spinner on core %d, doorbell nc=%d\n", core, d->nc);
        }
    }
    return d;
}

// ==== Generalized ork_spine FFN scheduler (ORK_FFN_SPINE) ============================================
// The dependency-ordered dual-priority DAG — the general form of the bespoke pool+driver. Units: 1 NPU
// (blocking ork_i8_mm_run = kernel-IRQ, frees its core during the matmul, apipe's mechanism) + N CPU units
// (silu/glu), driven by ork_spine_run. Per M-tile ops: gate,up (NPU) -> silu (CPU, dep gate) -> glu (CPU,
// dep silu+up) -> down (NPU, dep glu). Placement tags route ops; the scheduler overlaps + self-balances.
// Op fns are C-style (ork_spine_op.fn = long(*)(void*)); the per-tile context rides in OrkSpineArg.
struct OrkSpineArg {
    ggml_backend_ork_context * ctx;
    ork_w * wg; ork_w * wu; ork_w * wd;
    const int8_t * xr_i8; const float * as_x; const float * bsg; const float * bsu; const float * bsd;
    int32_t * g32; int32_t * u32; int32_t * d32;
    float * silu_f; float * up_f; int8_t * glu_i8; float * as_g; float * dst;
    int K; int Nff; int Kd; int m0; int mc;
};
static long ork_sp_gate(void * a){ OrkSpineArg * s = (OrkSpineArg *) a;                 // NPU: gate matmul -> g32
    return ork_i8_mm_run(s->ctx->npu, s->wg, s->mc, s->xr_i8 + (size_t) s->m0*s->K, s->g32 + (size_t) s->m0*s->Nff); }
static long ork_sp_up(void * a){ OrkSpineArg * s = (OrkSpineArg *) a;                   // NPU: up matmul -> u32
    return ork_i8_mm_run(s->ctx->npu, s->wu, s->mc, s->xr_i8 + (size_t) s->m0*s->K, s->u32 + (size_t) s->m0*s->Nff); }
static long ork_sp_silu(void * a){ OrkSpineArg * s = (OrkSpineArg *) a;                 // CPU: gate int32 -> exact fp32 silu
    for (int r = 0; r < s->mc; r++) { const float rs = s->as_x[s->m0+r];
        float * so = s->silu_f + (size_t)(s->m0+r)*s->Nff; const int32_t * gr = s->g32 + (size_t)(s->m0+r)*s->Nff;
        for (int n = 0; n < s->Nff; n++) { float g = (float) gr[n] * rs * s->bsg[n]; so[n] = g / (1.0f + expf(-g)); } }
    return 0; }
static long ork_sp_glu(void * a){ OrkSpineArg * s = (OrkSpineArg *) a;                  // CPU: up dequant + glu=silu*up + per-row int8 quant
    std::vector<float> g(s->Nff);
    for (int r = 0; r < s->mc; r++) { const int m = s->m0+r; const float rs = s->as_x[m];
        const int32_t * ur = s->u32 + (size_t) m*s->Nff; float * uo = s->up_f + (size_t) m*s->Nff; const float * so = s->silu_f + (size_t) m*s->Nff;
        float mx = 1e-9f;
        for (int n = 0; n < s->Nff; n++) { float u = rs * s->bsu[n] * (float) ur[n]; uo[n] = u; float gg = so[n]*u; g[n]=gg; float av=fabsf(gg); if(av>mx)mx=av; }
        float sc = mx/127.0f; s->as_g[m] = sc; float iv = 127.0f/mx; int8_t * gi = s->glu_i8 + (size_t) m*s->Nff;
        for (int n = 0; n < s->Nff; n++) { int q=(int)lrintf(g[n]*iv); gi[n]=(int8_t)(q>127?127:q<-127?-127:q); } }
    return 0; }
static long ork_sp_down(void * a){ OrkSpineArg * s = (OrkSpineArg *) a;                 // NPU: down matmul + dequant -> dst
    if (ork_i8_mm_run(s->ctx->npu, s->wd, s->mc, s->glu_i8 + (size_t) s->m0*s->Nff, s->d32 + (size_t) s->m0*s->Kd)) return -1;
    for (int r = 0; r < s->mc; r++) { const int m = s->m0+r; const float rs = s->as_g[m];
        const int32_t * dr = s->d32 + (size_t) m*s->Kd; float * do_ = s->dst + (size_t) m*s->Kd;
        for (int n = 0; n < s->Kd; n++) do_[n] = rs * s->bsd[n] * (float) dr[n]; }
    return 0; }

struct OrkSpine { ork_spine_unit * U = nullptr; int nu = 0; };
static OrkSpine * ork_spine_get() {
    static OrkSpine * s = nullptr; static bool init = false;
    if (!init) { init = true;
        if (getenv("ORK_FFN_SPINE")) {
            std::vector<int> cc; if (const char * e = getenv("ORK_FFN_SPINE_CPU_CORES")) { for (const char * p=e; *p; ) { cc.push_back(atoi(p)); while(*p&&*p!=',')p++; if(*p==',')p++; } }
            if (cc.empty()) cc = {4,5,6};
            int npu_core = getenv("ORK_FFN_SPINE_NPU_CORE") ? atoi(getenv("ORK_FFN_SPINE_NPU_CORE")) : 7;
            s = new OrkSpine(); s->nu = 1 + (int) cc.size(); s->U = new ork_spine_unit[s->nu];
            ork_spine_unit_start(&s->U[0], ORK_UNIT_NPU, npu_core);
            for (size_t i = 0; i < cc.size(); i++) ork_spine_unit_start(&s->U[1+i], ORK_UNIT_CPU, cc[i]);
            if (getenv("ORK_VERBOSE")) { fprintf(stderr, "[FFN SPINE] NPU unit core %d + %zu CPU units cores:", npu_core, cc.size()); for(int c:cc)fprintf(stderr," %d",c); fprintf(stderr,"\n"); }
        }
    }
    return s;
}

static bool ggml_backend_ork_ffn_swiglu_chain(ggml_backend_ork_context * ctx,
        struct ggml_tensor * gate_n, struct ggml_tensor * up_n,
        struct ggml_tensor * glu_n, struct ggml_tensor * down_n) {
    const struct ggml_tensor * Wg = gate_n->src[0];   // [K, Nff]
    const struct ggml_tensor * Wu = up_n->src[0];      // [K, Nff]
    const struct ggml_tensor * Wd = down_n->src[0];    // [Nff, Kd]
    const struct ggml_tensor * x  = gate_n->src[1];    // [K, M] (ffn_norm out)
    const int K   = (int) Wg->ne[0];
    const int Nff = (int) Wg->ne[1];
    const int Kd  = (int) Wd->ne[1];                   // down output width (= K, the hidden size)
    const int M   = (int) x->ne[1];
    // guards: shapes the fused gate/up primitives require. K (hidden) must be K%512 && K<=4096 (the
    // fused-SiLU / out8 full-K Bf envelope). Nff is the gate/up OUTPUT width (Sn-tiled at NMAX=8192 by
    // the primitives) AND the down projection's CONTRACTION dim (run() K-slices it, Sk=ceil(Nff/1024),
    // host-accumulate — the exact path the default 7B ffn_down K=18944 already uses). So Nff only needs
    // Nff%512==0 (down K-slice bit-exactness) && Nff%32==0 (gate/up N). No Nff<=8192 cap: the streamed
    // 7B (Nff=18944) tiles fine — validated the primitives handle it, perf may degrade (acceptable).
    if (getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK FFN-CHAIN] swiglu_chain reached %s K=%d Nff=%d Kd=%d M=%d\n", Wg->name, K, Nff, Kd, M);
    if (K % 512 || K > 4096 || Nff % 512 || Nff % 32 || Kd % 16 || M < 1) {
        if (getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK FFN-CHAIN] shape-guard REJECT %s K=%d Nff=%d Kd=%d M=%d\n", Wg->name, K, Nff, Kd, M);
        return false; }
    // ork_ffn_prep calibrates STATIC per-tensor scales (s_x, s_silu, SmoothQuant s[]) on the batch it FIRST
    // sees and caches them (fc.ready). llama.cpp's warmup runs at M=2 (unrepresentative) -> the cached scales
    // saturate/clip real activations -> garbage output (measured PPL 15.6k @1.7B / 1.2M @7B). Skip fusion for
    // tiny M so prep calibrates on a representative prefill-sized batch; per-node handles small-M / decode.
    // The M<32 skip guards the INT8 static-scale calibration (unrepresentative on a tiny warmup batch). The
    // all-fp16 path (ORK_FFN_F16) has NO static int8 calibration, so it is exempt — it engages at any M
    // (including decode M=1 and perplexity's M=2), which also lets perplexity actually measure its coherence.
    if (M < 32 && !getenv("ORK_FFN_F16")) { if (getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK FFN-CHAIN] skip small M=%d (calibration guard)\n", M); return false; }

    const float * xf = (const float *) x->data;
    // one-time per-layer SmoothQuant prep (smoothing + smoothed-weight pack + static-scale calibration + LUT)
    auto & fc = ctx->ffncache[Wg->data];
    if (!fc.ready) { if (!ork_ffn_prep(ctx, fc, Wg, Wu, Wd, xf, M, K, Nff, Kd)) return false; }

    // ---- ORK_FFN_F16: ALL-FP16 inner (no int8 quant, no int32 dequant, no int8 software-chain machinery).
    // x -> fp16 cast (shared by gate+up); gate = fp16 matmul + fused SiLU (silu_f fp32); up = fp16 matmul (up_f
    // fp32); glu = silu_f*up_f (fp32, cheap); down = fp16 matmul(glu->fp16) -> dst fp32. Each matmul is native
    // fp16 (M-tile = the f16_mtile ceiling after the M-tile fix). Falls through to per-node on any error. ----
    if (fc.f16_all) {
        const int cn = fc.wg_f16_cn;
        std::vector<ork_f16> xh((size_t) M * K); for (size_t i = 0; i < (size_t) M*K; i++) xh[i] = (ork_f16) xf[i];
        std::vector<float> siluf((size_t) M * Nff), upf((size_t) M * Nff), ctmp((size_t) M * cn);
        bool ok2 = true;
        if (fc.f16_cpusilu) ork_npu_set_core_budget(ctx->npu, 1);   // single-core gate: plain fp16 EINVALs cold multi-core (see f16_jit)
        for (size_t ci = 0; ci < fc.wg_f16.size() && ok2; ci++) {   // gate: fp16 matmul + SiLU
            int n0 = (int) ci * cn, cw = (Nff - n0 < cn) ? (Nff - n0) : cn;
            if (fc.f16_cpusilu ? ork_f16_mm_run(ctx->npu, fc.wg_f16[ci], M, xh.data(), ctmp.data())
                               : ork_f16_mm_run_silu(ctx->npu, fc.wg_f16[ci], M, xh.data(), ctmp.data(), 0, 0xffffc000u, 0x56391100u, fc.lut_f16.data(), 1030)) { ok2 = false; break; }
            for (int m = 0; m < M; m++) { float * so = siluf.data() + (size_t) m*Nff + n0; const float * cr = ctmp.data() + (size_t) m*cw;
                if (fc.f16_cpusilu) for (int n = 0; n < cw; n++) { float g = cr[n]; so[n] = g / (1.0f + expf(-g)); }   // exact CPU silu on the raw fp16 gate
                else                for (int n = 0; n < cw; n++) so[n] = cr[n] * (float) fc.f16_out; }
        }
        if (fc.f16_cpusilu) ork_npu_set_core_budget(ctx->npu, ork_npu_cores(ctx->npu));   // restore MC for up/down
        for (size_t ci = 0; ci < fc.wu_f16.size() && ok2; ci++) {   // up: fp16 matmul
            int n0 = (int) ci * cn, cw = (Nff - n0 < cn) ? (Nff - n0) : cn;
            if (ork_f16_mm_run(ctx->npu, fc.wu_f16[ci], M, xh.data(), ctmp.data())) { ok2 = false; break; }
            for (int m = 0; m < M; m++) { float * uo = upf.data() + (size_t) m*Nff + n0; const float * cr = ctmp.data() + (size_t) m*cw;
                for (int n = 0; n < cw; n++) uo[n] = cr[n]; }
        }
        if (ok2) {                                                  // glu = silu*up (fp32) -> fp16; down: fp16 matmul -> dst
            std::vector<ork_f16> gluh((size_t) M * Nff);
            for (size_t i = 0; i < (size_t) M*Nff; i++) gluh[i] = (ork_f16) (siluf[i] * upf[i]);
            if (ork_f16_mm_run(ctx->npu, fc.wd_f16, M, gluh.data(), (float *) down_n->data)) ok2 = false;
        }
        if (getenv("ORK_VERBOSE")) { static int o=0; if(!o++) fprintf(stderr,"[ORK FFN-F16] all-fp16 inner %s ok=%d M=%d\n", Wg->name, ok2, M); }
        return ok2;   // false -> dispatch falls through to per-node (weights already packed; harmless)
    }

    // ---- ORK_FFN_F16_JIT: same fp16 inner as f16_all, but each weight is inflated from a host int8 store
    // into a SHARED fp16 scratch right before its matmul (no resident fp16). IOVA cost = a few shared
    // scratches reused across every JIT layer, so gmax layer count is decoupled from the 4GiB domain. ----
    if (fc.f16_jit) {
        const int cn = fc.wg_f16_cn;
        // ORK_VERBOSE timing split: accumulate CPU inflate-us vs NPU run-us (+ other=cast/scatter) across ALL
        // handler calls, to find where the fp16 route spends time (is it the JIT inflate, or the fp16 matmuls?).
        const bool tv = getenv("ORK_VERBOSE") != nullptr;
        static double t_infl = 0, t_run = 0; static long t_calls = 0;
        double t_h0 = tv ? (double) ggml_time_us() : 0;
        std::vector<ork_f16> xh((size_t) M * K); for (size_t i = 0; i < (size_t) M*K; i++) xh[i] = (ork_f16) xf[i];
        std::vector<float> siluf((size_t) M * Nff), upf((size_t) M * Nff), ctmp((size_t) M * cn);
        bool ok2 = true;
        // CPU-silu gate uses the PLAIN fp16 matmul (ork_f16_mm_run), which would take the multi-core mcworker path
        // and EINVAL when hit COLD (before fp16 mode is warmed). Force single-core for the gate (it's a tiny
        // single-tile weight anyway); the single-core run warms fp16 mode so up/down can go multi-core after.
        if (fc.f16_cpusilu) ork_npu_set_core_budget(ctx->npu, 1);
        for (size_t ci = 0; ci < fc.jg.size() && ok2; ci++) {       // gate: inflate -> fp16 matmul + SiLU
            int n0 = (int) ci * cn, cw = fc.jg[ci].N;
            ork_w * sc = ork_get_f16_scratch(ctx, K, cw);
            double ta = tv ? (double) ggml_time_us() : 0;
            if (!sc || ork_i8_mm_inflate_to_f16(ctx->npu, sc, fc.jg[ci].i8.data(), fc.jg[ci].bs.data(), K, cw)) { ok2 = false; break; }
            double tb = tv ? (double) ggml_time_us() : 0; if (tv) t_infl += tb - ta;
            if (fc.f16_cpusilu ? ork_f16_mm_run(ctx->npu, sc, M, xh.data(), ctmp.data())
                               : ork_f16_mm_run_silu(ctx->npu, sc, M, xh.data(), ctmp.data(), 0, 0xffffc000u, 0x56391100u, fc.lut_f16.data(), 1030)) { ok2 = false; break; }
            if (tv) t_run += (double) ggml_time_us() - tb;
            for (int m = 0; m < M; m++) { float * so = siluf.data() + (size_t) m*Nff + n0; const float * cr = ctmp.data() + (size_t) m*cw;
                if (fc.f16_cpusilu) for (int n = 0; n < cw; n++) { float g = cr[n]; so[n] = g / (1.0f + expf(-g)); }   // exact CPU silu
                else                for (int n = 0; n < cw; n++) so[n] = cr[n] * (float) fc.f16_out; }
        }
        if (fc.f16_cpusilu) ork_npu_set_core_budget(ctx->npu, ork_npu_cores(ctx->npu));   // restore MC for up/down
        for (size_t ci = 0; ci < fc.ju.size() && ok2; ci++) {       // up: inflate -> fp16 matmul
            int n0 = (int) ci * cn, cw = fc.ju[ci].N;
            ork_w * sc = ork_get_f16_scratch(ctx, K, cw);           // same (K,cw) scratch the gate just used
            double ta = tv ? (double) ggml_time_us() : 0;
            if (!sc || ork_i8_mm_inflate_to_f16(ctx->npu, sc, fc.ju[ci].i8.data(), fc.ju[ci].bs.data(), K, cw)) { ok2 = false; break; }
            double tb = tv ? (double) ggml_time_us() : 0; if (tv) t_infl += tb - ta;
            if (ork_f16_mm_run(ctx->npu, sc, M, xh.data(), ctmp.data())) { ok2 = false; break; }
            if (tv) t_run += (double) ggml_time_us() - tb;
            for (int m = 0; m < M; m++) { float * uo = upf.data() + (size_t) m*Nff + n0; const float * cr = ctmp.data() + (size_t) m*cw;
                for (int n = 0; n < cw; n++) uo[n] = cr[n]; }
        }
        if (ok2) {                                                  // glu = silu*up -> fp16; down: inflate -> matmul
            std::vector<ork_f16> gluh((size_t) M * Nff);
            for (size_t i = 0; i < (size_t) M*Nff; i++) gluh[i] = (ork_f16) (siluf[i] * upf[i]);
            ork_w * sc = ork_get_f16_scratch(ctx, Nff, Kd);
            double ta = tv ? (double) ggml_time_us() : 0;
            if (!sc || ork_i8_mm_inflate_to_f16(ctx->npu, sc, fc.jd.i8.data(), fc.jd.bs.data(), Nff, Kd)) ok2 = false;
            else { double tb = tv ? (double) ggml_time_us() : 0; if (tv) t_infl += tb - ta;
                   if (ork_f16_mm_run(ctx->npu, sc, M, gluh.data(), (float *) down_n->data)) ok2 = false;
                   if (tv) t_run += (double) ggml_time_us() - tb; }
        }
        if (tv) { double t_tot = (double) ggml_time_us() - t_h0; t_calls++;
            if (t_calls == 1 || (t_calls % 56) == 0)   // ~every 2 forwards for a 28-layer model
                fprintf(stderr, "[ORK FFN-F16 JIT-PROF] %s M=%d scratches=%zu | this-call %.2fms | CUMULATIVE inflate=%.1fms run=%.1fms (inflate=%.0f%%) over %ld calls\n",
                        Wg->name, M, ctx->f16_scratch.size(), t_tot/1000.0, t_infl/1000.0, t_run/1000.0,
                        100.0*t_infl/(t_infl+t_run+1e-9), t_calls);
        }
        return ok2;   // false -> dispatch falls through to per-node
    }

    // ---- HYBRID: gate=fused-SiLU (per-tensor, silu FREE on-NPU); up/down = standard per-channel host-dequant
    // (near-baseline quality). Only silu_i8 is a per-tensor int8 intermediate; up/glu stay fp32. ----
    const size_t nx = (size_t) M * K;
    // gate activation: smoothed static per-tensor x'=x/s -> xi
    const double xinv = 1.0 / fc.s_x;
    std::vector<int8_t> xi(nx);
    for (int m = 0; m < M; m++) { const float * xr = xf + (size_t) m * K; int8_t * a = xi.data() + (size_t) m*K;
        for (int k = 0; k < K; k++) { int q = (int) lrint((xr[k] / fc.s[k]) * xinv);
            a[k] = (int8_t) (q > 127 ? 127 : q < -127 ? -127 : q); } }
    // up activation: per-ROW quant of the ORIGINAL x (baseline-style) -> xr_i8 + as_x[m]
    std::vector<int8_t> xr_i8(nx); std::vector<float> as_x(M);
    for (int m = 0; m < M; m++) { const float * xr = xf + (size_t) m * K; int8_t * a = xr_i8.data() + (size_t) m*K;
        float mx = 1e-9f; for (int k = 0; k < K; k++) { float v = fabsf(xr[k]); if (v > mx) mx = v; }
        as_x[m] = mx / 127.0f; float iv = 127.0f / mx;
        for (int k = 0; k < K; k++) { int q = (int) lrintf(xr[k]*iv); a[k] = (int8_t)(q>127?127:q<-127?-127:q); } }
    // resolve per-channel up/down weights (standard wcache path). PIN each against LRU eviction: resolving
    // `down` (and `gate` in the GU_CHAIN path) can trigger ork_wcache_evict, which under a tight budget would
    // otherwise free the just-resolved `up` — dangling the held owu/bsu references (ASAN use-after-free at the
    // phase-1 up matmul, 4421). The guard unpins on EVERY return path.
    struct WCachePinGuard { std::unordered_set<const void *> & s; ~WCachePinGuard() { s.clear(); } } _wc_pin{ ctx->wcache_pin };
    auto itu = ork_resolve_weight_i8(ctx, Wu, K, Nff, Wu->nb[1], Wu->type, ggml_get_type_traits(Wu->type)->to_float, true);
    if (itu == ctx->wcache.end()) return false;
    ctx->wcache_pin.insert(itu->first);                  // pin `up` BEFORE resolving `down` (which may evict)
    const ork_weight & owu = itu->second; const float * bsu = owu.bscale.data();
    auto itd = ork_resolve_weight_i8(ctx, Wd, Nff, Kd, Wd->nb[1], Wd->type, ggml_get_type_traits(Wd->type)->to_float, true);
    if (itd == ctx->wcache.end()) return false;
    ctx->wcache_pin.insert(itd->first);
    const ork_weight & owd = itd->second; const float * bsd = owd.bscale.data();
    // NOTE: gate/up/down do NOT need co-residence. Each is a SEPARATE ork_mm_run_* call that goes through
    // run()->dom_activate(w->domain)->a SINGLE-domain submit, so they may live in different domains (the
    // chain is a sequence of single-domain submits, dom_activate switching between; the host silu/glu
    // intermediates are domain-independent). Forcing co-residence was what jammed fc.wg into a weight-packed
    // domain and fragmented it. fc.wg now gets its OWN dedicated domain (ork_weight_domain fcwg path).

    const int RM = 0x4000, RS = 0x10; const uint32_t OB = 0, IO = 0xffffc000u, C4 = 0x56391300u;
    const int16_t * lut = fc.lut.data();
    const int MT = 128;
    // ORK_GATE_ABLATE: isolate the gate's 3 error sources (unset = normal fused hybrid).
    //   0 = exact: per-channel Wg + per-row x + fp32 silu (should ~= baseline; confirms fused-silu is the culprit)
    //   1 = +per-tensor WEIGHT (smoothed Wg, per-row smoothed x, fp32 silu)   [isolates source #2]
    //   2 = +per-tensor STATIC activation (smoothed Wg, per-tensor-static x, fp32 silu)  [+ source #1]
    //   3 = +int8 SILU output (mode 2 then round-trip silu through int8)       [+ source #3 ~= full chain]
    const int ablate = getenv("ORK_GATE_ABLATE") ? atoi(getenv("ORK_GATE_ABLATE")) : -1;
    std::vector<int8_t>  glr_i8((size_t) M * Nff);
    std::vector<int32_t> acc_i32((size_t) M * Nff);   // reused (Nff >= Kd)
    std::vector<float>   up_f((size_t) M * Nff), silu_f((size_t) M * Nff), glu_f((size_t) M * Nff), as_g(M);
    bool ok = true;
    // ORK_FFN_SPINE: generalized dependency-ordered dual-priority DAG. Runs the WHOLE FFN inner (gate/up/silu/
    // glu/down -> dst) as an ork_spine op-graph and returns; Phase 1/2/3 below are skipped. Placement tags
    // (NPU/CPU) route each op; ork_spine_run overlaps independent NPU/CPU work + self-balances free units.
    if (getenv("ORK_FFN_SPINE") && fc.silu_cpu && !fc.silu_i16 && ablate < 0) {
        OrkSpine * sp = ork_spine_get();
        if (sp) {
            auto itg = ork_resolve_weight_i8(ctx, Wg, K, Nff, Wg->nb[1], Wg->type, ggml_get_type_traits(Wg->type)->to_float, true);
            if (itg == ctx->wcache.end()) return false;
            ctx->wcache_pin.insert(itg->first);
            const ork_weight & owg = itg->second; const float * bsg = owg.bscale.data();
            // Default nt=1 (TS=M): full-M matmuls (no per-tile weight-DMA re-stream tax) with silu∥up overlap
            // in the one tile — apipe's shape. Smaller TS pipelines more tiles but re-streams weights (net loss
            // on this weight-DMA-bound matmul). ORK_FFN_SPINE_TS overrides; capped so nt*5 <= 30 (<32 DAG cap).
            int TS = getenv("ORK_FFN_SPINE_TS") ? atoi(getenv("ORK_FFN_SPINE_TS")) : M;
            if (TS < (M + 5) / 6) TS = (M + 5) / 6;   // keep nt <= 6
            if (TS < 1) TS = 1; const int nt = (M + TS - 1) / TS;
            std::vector<int32_t> g32((size_t) M*Nff), u32((size_t) M*Nff), d32((size_t) M*Kd);
            std::vector<int8_t>  glu_i8b((size_t) M*Nff);
            std::vector<OrkSpineArg> args(nt);
            std::vector<ork_spine_op> ops((size_t) nt*5);
            for (int t = 0; t < nt; t++) { int m0 = t*TS, mc = std::min(TS, M - m0);
                args[t] = OrkSpineArg{ ctx, owg.w, owu.w, owd.w, xr_i8.data(), as_x.data(), bsg, bsu, bsd,
                    g32.data(), u32.data(), d32.data(), silu_f.data(), up_f.data(), glu_i8b.data(), as_g.data(), (float *) down_n->data, K, Nff, Kd, m0, mc };
                int b = t*5;
                ops[b+0] = ork_spine_op{ 0,                       ORK_PL_NPU, ork_sp_gate, &args[t], 0, -1, 0, 0 };
                ops[b+1] = ork_spine_op{ 0,                       ORK_PL_NPU, ork_sp_up,   &args[t], 0, -1, 0, 0 };
                ops[b+2] = ork_spine_op{ 1<<(b+0),                ORK_PL_CPU, ork_sp_silu, &args[t], 0, -1, 0, 0 };   // dep gate
                ops[b+3] = ork_spine_op{ (1<<(b+2))|(1<<(b+1)),   ORK_PL_CPU, ork_sp_glu,  &args[t], 0, -1, 0, 0 };   // dep silu + up
                ops[b+4] = ork_spine_op{ 1<<(b+3),                ORK_PL_NPU, ork_sp_down, &args[t], 0, -1, 0, 0 };   // dep glu
            }
            int rc = ork_spine_run(sp->U, sp->nu, ops.data(), (int)(nt*5));
            bool oks = rc == 0; for (size_t i = 0; i < ops.size() && oks; i++) if (ops[i].ret) oks = false;
            if (getenv("ORK_VERBOSE")) { static int o = 0; if (!o++) fprintf(stderr, "[FFN SPINE] %s M=%d nt=%d TS=%d nu=%d rc=%d ok=%d\n", Wg->name, M, nt, TS, sp->nu, rc, oks); }
            return oks;   // dst written by the down ops; Phase 1/2/3 skipped
        }
    }
    bool gu_done = false;
    // ORK_FFN_PIPE: doorbell same-thread M-tile pipeline for the int8-gate + exact-CPU-SiLU path (fc.silu_cpu,
    // the shipped-default chain path). Token rows are independent, so the NPU runs tile t's gate+up (one
    // nonblocking doorbell submit) while the CPU does tile t-1's SiLU+up-dequant. Per-layer lane timing under
    // ORK_VERBOSE. Superseded by ORK_FFN_SCHED (two-chain) which tunes higher; kept for A/B.
    const bool pipe = getenv("ORK_FFN_PIPE") && fc.silu_cpu && !fc.silu_i16 && ablate < 0;
    // ORK_FFN_SCHED: doorbell-native scheduler. gate-chain (one ork_dyn_begin_mc over ALL gate tiles -> the
    // doorbell SPINS through them, no per-tile submit) -> drain (read-after-drain coherent) -> dispatch all
    // silu to the CPU pool WHILE the up-chain spins on the NPU -> drain -> glu/down. Realizes the two-priority
    // dep-ordered queue within the coherency contract (mid-chain reads are NOT coherent; per-chain drain is).
    const bool sched = getenv("ORK_FFN_SCHED") && fc.silu_cpu && !fc.silu_i16 && ablate < 0;
    // STATIC-GRAPH submit reduction (ORK_GU_CHAIN): HW-chain up+gate into ONE run_chain_i8 submit per
    // M-tile — both read xr_i8 and are independent, so the hardware walks them as task_number=2 in a single
    // ioctl (collapses 2 tiled matmul submits -> 1 for the int16/cpu-silu path). run_chain_i8 is single-core;
    // K=hidden %512==0 && <=4096 and Nff single-slice satisfy its envelope. Falls back to the separate
    // per-op path on ANY failure (cok), so output is always valid. A/B on the same binary via the env.
    if (getenv("ORK_GU_CHAIN") && fc.silu_cpu && ablate < 0) {
        auto itg = ork_resolve_weight_i8(ctx, Wg, K, Nff, Wg->nb[1], Wg->type, ggml_get_type_traits(Wg->type)->to_float, true);
        if (itg != ctx->wcache.end()) {
            ctx->wcache_pin.insert(itg->first);          // pin `gate` — GU_CHAIN uses up+gate co-resident in one submit
            const ork_weight & owg = itg->second; const float * bsg = owg.bscale.data();
            // Stage the shared int8 input + gate/up int32 outputs in cached NPU-coherent DMA scratch so
            // run_chain_i8's dma_find HITS -> no per-task bcreate/memcpy/bsync/bdestroy, and the shared
            // input syncs once (dedup). din = whole M*K input; dup/dgt = one M-tile of outputs (reused).
            int8_t  * din = (int8_t  *) ork_dma_grow(ctx->npu, &ctx->dma_in, &ctx->dma_in_sz, (size_t) M*K);
            int32_t * dup = (int32_t *) ork_dma_grow(ctx->npu, &ctx->dma_up, &ctx->dma_up_sz, (size_t) MT*Nff*4);
            int32_t * dgt = (int32_t *) ork_dma_grow(ctx->npu, &ctx->dma_gt, &ctx->dma_gt_sz, (size_t) MT*Nff*4);
            bool cok = (din && dup && dgt);
            if (cok) memcpy(din, xr_i8.data(), (size_t) M*K);
            for (int m0 = 0; m0 < M && cok; m0 += MT) { int mc = std::min(MT, M - m0);
                ork_mm_task_i8 t2[2] = {
                    { owu.w, mc, din + (size_t) m0*K, dup },
                    { owg.w, mc, din + (size_t) m0*K, dgt } };
                if (ork_i8_mm_run_chain(ctx->npu, 2, t2)) { cok = false; break; }
                for (int r = 0; r < mc; r++) { const int32_t * ur = dup + (size_t) r*Nff;                 // dequant up
                    float * uo = up_f.data() + (size_t)(m0+r)*Nff; const float rs = as_x[m0+r];
                    for (int n = 0; n < Nff; n++) uo[n] = rs * bsu[n] * (float) ur[n]; }
                if (fc.silu_i16) {                                                                        // on-NPU int16 SiLU on gate
                    std::vector<float> gr((size_t) mc*Nff); float gmx = 1e-9f;
                    for (int r = 0; r < mc; r++) { const int32_t * ar = dgt + (size_t) r*Nff; const float rs = as_x[m0+r];
                        for (int n = 0; n < Nff; n++) { float g = (float) ar[n] * rs * bsg[n]; gr[(size_t) r*Nff+n] = g; float a = fabsf(g); if (a > gmx) gmx = a; } }
                    double is = gmx/32000.0, os = gmx/32000.0; if (is<=0) is=1e-9; if (os<=0) os=1e-9;
                    std::vector<int16_t> in16((size_t) mc*Nff), out16((size_t) mc*Nff);
                    for (size_t i = 0; i < (size_t) mc*Nff; i++) { long q = lround(gr[i]/is); if (q>32767) q=32767; if (q<-32768) q=-32768; in16[i] = (int16_t) q; }
                    double us = 0;
                    if (ork_i16_npu_silu(ctx->npu, in16.data(), mc, Nff, is, os, out16.data(), &us)) { cok = false; break; }
                    for (int r = 0; r < mc; r++) { float * so = silu_f.data() + (size_t)(m0+r)*Nff;
                        for (int n = 0; n < Nff; n++) so[n] = (float) out16[(size_t) r*Nff+n] * os; }
                } else {                                                                                  // fp32 CPU SiLU on gate
                    for (int r = 0; r < mc; r++) { const int32_t * ar = dgt + (size_t) r*Nff;
                        float * so = silu_f.data() + (size_t)(m0+r)*Nff; const float rs = as_x[m0+r];
                        for (int n = 0; n < Nff; n++) { float g = (float) ar[n] * rs * bsg[n]; so[n] = g / (1.0f + expf(-g)); } }
                }
            }
            gu_done = cok;
        }
    }
    // Phase 1: up (per-channel, per-row) -> up_f fp32  [skipped if the chained fast-path already did up+gate,
    // or if pipe/sched (doorbell) fold up into their overlapped Phase-2 schedule]
    for (int m0 = 0; !gu_done && !pipe && !sched && m0 < M && ok; m0 += MT) { int mc = std::min(MT, M - m0);
        if (ork_i8_mm_run(ctx->npu, owu.w, mc, xr_i8.data() + (size_t) m0*K, acc_i32.data())) { if(getenv("ORK_VERBOSE"))fprintf(stderr,"[FFN i16] FAIL: up matmul owu m0=%d\n",m0); ok = false; break; }
        for (int r = 0; r < mc; r++) { const int32_t * ur = acc_i32.data() + (size_t) r*Nff;
            float * uo = up_f.data() + (size_t)(m0+r)*Nff; const float rs = as_x[m0+r];
            for (int n = 0; n < Nff; n++) uo[n] = rs * bsu[n] * (float) ur[n]; } }
    // Phase 2: silu_f fp32
    // FUSED int16 (ORK_FFN_SILU_I16_FUSED): fc.wg int8-mm -> int16-out (set_i16_out, on-device, internally
    // N-tiled) -> int16 SiLU (N-tiled at nmax). No host int32->fp32->int16 round-trip, no separate per-channel
    // gate matmul. Per-tensor gate (scalar out16) => PPL A/B vs the un-fused per-channel int16 path.
    static const int silu_i16_fused = getenv("ORK_FFN_SILU_I16_FUSED") != nullptr;
    if (sched && ok && !gu_done) {
        // ---- ORK_FFN_SCHED: doorbell-native two-chain scheduler (the dependency-ordered queue, within the
        // read-after-drain coherency contract). gate-chain: ONE ork_dyn_begin_mc over all gate tiles -> the
        // doorbell spins through them (no per-tile submit) -> drain (coherent). Then dispatch all silu to the
        // CPU pool (work-stealing, big+little) WHILE the up-chain spins on the NPU -> drain -> dequant up.
        // Fills silu_f + up_f; Phase 3 (glu + down) runs unchanged. down (K=Nff>4096) is not doorbell-eligible.
        auto itg = ork_resolve_weight_i8(ctx, Wg, K, Nff, Wg->nb[1], Wg->type, ggml_get_type_traits(Wg->type)->to_float, true);
        if (itg == ctx->wcache.end()) { if (getenv("ORK_VERBOSE")) fprintf(stderr, "[FFN SCHED] gate resolve miss %s\n", Wg->name); ok = false; }
        else {
            ctx->wcache_pin.insert(itg->first);
            const ork_weight & owg = itg->second; const float * bsg = owg.bscale.data();
            OrkFfnPool * pool = ork_ffn_pool();
            const int TS = 64; const int nt = (M + TS - 1) / TS;                  // doorbell tile cap (mc M<=64)
            std::vector<int32_t> g32((size_t) M*Nff), u32((size_t) M*Nff);        // gate/up int32 (read-after-drain coherent)
            std::vector<ork_mm_task_i8> gt(nt), ut(nt);
            for (int t = 0; t < nt; t++) { int m0 = t*TS, mc = std::min(TS, M - m0);
                gt[t] = { owg.w, mc, xr_i8.data() + (size_t) m0*K, g32.data() + (size_t) m0*Nff };
                ut[t] = { owu.w, mc, xr_i8.data() + (size_t) m0*K, u32.data() + (size_t) m0*Nff }; }
            static double s_g = 0, s_u = 0; static long s_l = 0; double t0 = (double) ggml_time_us();
            OrkNpuDriver * driver = ork_npu_driver(ctx->npu);   // dedicated spinner drives the NPU on its own core
            // gate-chain: driven to completion (drain => coherent). Spinner if enabled, else calling thread.
            if (driver) { driver->run_async(gt.data(), nt); if (driver->wait()) ok = false; }
            else { ork_dyn_chain * hA = ork_dyn_begin_mc(ctx->npu, nt, gt.data(), 0);
                if (hA) { if (ork_dyn_end(hA) < 0) ok = false; }
                else for (int t = 0; t < nt && ok; t++) { int m0 = t*TS, mc = std::min(TS, M - m0);
                    if (ork_i8_mm_run(ctx->npu, owg.w, mc, xr_i8.data()+(size_t) m0*K, g32.data()+(size_t) m0*Nff)) ok = false; } }
            double tg = (double) ggml_time_us();
            if (ok) {
                // up-chain: the dedicated spinner drives it on its OWN core (never leaving the doorbell) while
                // the CPU pool does silu on the rest — the fix for the sched starvation (was: caller left the
                // doorbell to do silu, so the chain stalled). Without a driver, falls back to caller-drives.
                ork_dyn_chain * hB = nullptr;
                if (driver) driver->run_async(ut.data(), nt);
                else        hB = ork_dyn_begin_mc(ctx->npu, nt, ut.data(), 0);
                for (int t = 0; t < nt; t++) {
                    auto silu = [&, t]{ int m0 = t*TS, mc = std::min(TS, M - m0);
                        for (int r = 0; r < mc; r++) { const float rs = as_x[m0+r];
                            float * so = silu_f.data() + (size_t)(m0+r)*Nff; const int32_t * gr = g32.data() + (size_t)(m0+r)*Nff;
                            for (int n = 0; n < Nff; n++) { float g = (float) gr[n] * rs * bsg[n]; so[n] = g / (1.0f + expf(-g)); } } };
                    if (pool) pool->submit(silu); else silu(); }
                if (driver) { if (driver->wait()) ok = false; }
                else if (hB) { if (ork_dyn_end(hB) < 0) ok = false; }
                else for (int t = 0; t < nt && ok; t++) { int m0 = t*TS, mc = std::min(TS, M - m0);
                    if (ork_i8_mm_run(ctx->npu, owu.w, mc, xr_i8.data()+(size_t) m0*K, u32.data()+(size_t) m0*Nff)) ok = false; }
                if (pool) pool->wait();
                double tu = (double) ggml_time_us();
                if (ok) { auto deq = [&](int t){ int m0 = t*TS, mc = std::min(TS, M - m0);   // dequant up -> up_f (coherent)
                        for (int r = 0; r < mc; r++) { const float rs = as_x[m0+r];
                            float * uo = up_f.data() + (size_t)(m0+r)*Nff; const int32_t * ur = u32.data() + (size_t)(m0+r)*Nff;
                            for (int n = 0; n < Nff; n++) uo[n] = rs * bsu[n] * (float) ur[n]; } };
                    if (pool) pool->run(nt, deq); else for (int t = 0; t < nt; t++) deq(t); }
                s_g += tg - t0; s_u += tu - tg; s_l++;
                if (getenv("ORK_VERBOSE")) fprintf(stderr, "[FFN SCHED] %s M=%d nt=%d | gate-chain=%.1fms silu∥up-chain=%.1fms || Σ L=%ld g=%.0fms u=%.0fms\n",
                    Wg->name, M, nt, (tg-t0)/1000, (tu-tg)/1000, s_l, s_g/1000, s_u/1000);
            }
        }
    } else if (pipe && ok && !gu_done) {
        // ---- ORK_FFN_PIPE: doorbell (ork_dyn_begin_mc) same-thread nonblock M-tile pipeline, tile TS<=64
        // (the envelope cap). Token rows are independent, so tile t's gate+up run on the NPU while the CPU
        // does tile t-1's SiLU+up-dequant. Measured net -3% at TS=64: halving the tile vs baseline MT=128
        // doubles the weight-DMA re-stream (NPU-bound path). Kept for the A/B + lane timing.
        auto itg = ork_resolve_weight_i8(ctx, Wg, K, Nff, Wg->nb[1], Wg->type, ggml_get_type_traits(Wg->type)->to_float, true);
        if (itg == ctx->wcache.end()) { if (getenv("ORK_VERBOSE")) fprintf(stderr, "[FFN PIPE] gate resolve miss %s\n", Wg->name); ok = false; }
        else {
            ctx->wcache_pin.insert(itg->first);
            const ork_weight & owg = itg->second; const float * bsg = owg.bscale.data();
            const int TS = 64; const int nt = (M + TS - 1) / TS;
            std::vector<int32_t> gC0((size_t) TS*Nff), gC1((size_t) TS*Nff);      // ping-pong gate int32 (CPU reads t-1 while NPU writes t)
            std::vector<int32_t> uC0((size_t) TS*Nff), uC1((size_t) TS*Nff);      // ping-pong up int32
            int32_t * gCb[2] = { gC0.data(), gC1.data() }; int32_t * uCb[2] = { uC0.data(), uC1.data() };
            static double s_cpu = 0, s_drain = 0, s_sub = 0; static long s_layers = 0;
            double l_cpu = 0, l_drain = 0, l_sub = 0;
            OrkFfnPool * pool = ork_ffn_pool();
            auto cpu_lane = [&](int t) {
                int m0 = t*TS, mc = std::min(TS, M - m0), b = t & 1;
                const int32_t * gc = gCb[b], * uc = uCb[b];
                auto row = [&](int r) { const float rs = as_x[m0+r];
                    float * so = silu_f.data() + (size_t)(m0+r)*Nff; float * uo = up_f.data() + (size_t)(m0+r)*Nff;
                    const int32_t * gr = gc + (size_t) r*Nff, * ur = uc + (size_t) r*Nff;
                    for (int n = 0; n < Nff; n++) { float g = (float) gr[n] * rs * bsg[n]; so[n] = g / (1.0f + expf(-g)); uo[n] = rs * bsu[n] * (float) ur[n]; } };
                if (pool) pool->run(mc, row); else for (int r = 0; r < mc; r++) row(r);
            };
            int prev = -1;
            for (int t = 0; t < nt && ok; t++) {
                int m0 = t*TS, mc = std::min(TS, M - m0), b = t & 1;
                ork_mm_task_i8 tk[2] = { { owg.w, mc, xr_i8.data() + (size_t) m0*K, gCb[b] },
                                         { owu.w, mc, xr_i8.data() + (size_t) m0*K, uCb[b] } };
                double ts = (double) ggml_time_us();
                ork_dyn_chain * dh = ork_dyn_begin_mc(ctx->npu, 2, tk, 0);
                double tc0 = (double) ggml_time_us(); l_sub += tc0 - ts;
                if (!dh) {
                    if (ork_i8_mm_run(ctx->npu, owg.w, mc, xr_i8.data()+(size_t) m0*K, gCb[b]) ||
                        ork_i8_mm_run(ctx->npu, owu.w, mc, xr_i8.data()+(size_t) m0*K, uCb[b])) { if (getenv("ORK_VERBOSE")) fprintf(stderr, "[FFN PIPE] blocking fallback FAIL t=%d\n", t); ok = false; break; }
                    l_drain += (double) ggml_time_us() - tc0;
                    if (prev >= 0) { double p0 = (double) ggml_time_us(); cpu_lane(prev); l_cpu += (double) ggml_time_us() - p0; }
                    prev = t; continue;
                }
                if (prev >= 0) cpu_lane(prev);
                double tc1 = (double) ggml_time_us(); l_cpu += tc1 - tc0;
                if (ork_dyn_end(dh) < 0) { if (getenv("ORK_VERBOSE")) fprintf(stderr, "[FFN PIPE] dyn_end FAIL t=%d\n", t); ok = false; break; }
                l_drain += (double) ggml_time_us() - tc1;
                prev = t;
            }
            if (ok && prev >= 0) { double p0 = (double) ggml_time_us(); cpu_lane(prev); l_cpu += (double) ggml_time_us() - p0; }
            s_cpu += l_cpu; s_drain += l_drain; s_sub += l_sub; s_layers++;
            if (getenv("ORK_VERBOSE")) fprintf(stderr, "[FFN PIPE] %s M=%d nt=%d TS=%d | cpu=%.1fms drain=%.1fms sub=%.2fms || Σ L=%ld cpu=%.0fms drain=%.0fms sub=%.0fms\n",
                Wg->name, M, nt, TS, l_cpu/1000, l_drain/1000, l_sub/1000, s_layers, s_cpu/1000, s_drain/1000, s_sub/1000);
        }
    } else if (ok && !gu_done && fc.silu_i16 && silu_i16_fused && fc.wg && ablate < 0) {
        std::vector<int16_t> gate_i16((size_t) M * Nff);
        if (ork_i8_mm_run_out16(ctx->npu, fc.wg, M, xi.data(), (short *) gate_i16.data(), fc.gate16_mult, fc.gate16_shift)) {
            if (getenv("ORK_VERBOSE")) fprintf(stderr, "[FFN i16-fused] FAIL: mm_out16 %s\n", Wg->name); ok = false;
        } else {
            const int NT16 = 8192; const double is = fc.gate16_is;
            std::vector<int16_t> cin((size_t) M*NT16), cout((size_t) M*NT16);
            for (int n0 = 0; n0 < Nff && ok; n0 += NT16) { int cw = std::min(NT16, Nff - n0);
                for (int r = 0; r < M; r++) memcpy(cin.data() + (size_t) r*cw, gate_i16.data() + (size_t) r*Nff + n0, (size_t) cw*2);
                double us = 0;
                if (ork_i16_npu_silu(ctx->npu, cin.data(), M, cw, is, is, cout.data(), &us)) { if (getenv("ORK_VERBOSE")) fprintf(stderr, "[FFN i16-fused] FAIL: silu_i16 cw=%d\n", cw); ok = false; break; }
                for (int r = 0; r < M; r++) { float * so = silu_f.data() + (size_t) r*Nff + n0;
                    for (int n = 0; n < cw; n++) so[n] = (float) cout[(size_t) r*cw+n] * is; } }
        }
    } else if (ok && !gu_done && fc.silu_cpu && ablate < 0) {
        // EXACT gate (strategic high-gmax layer): per-channel gate matmul on-NPU + fp32 silu on CPU. Same recipe
        // as the up projection (per-channel weight, per-row x), so quality matches baseline; only the fused-silu
        // stage is skipped for this layer. Reuses xr_i8 + as_x (per-row original x, already computed above).
        auto itg = ork_resolve_weight_i8(ctx, Wg, K, Nff, Wg->nb[1], Wg->type, ggml_get_type_traits(Wg->type)->to_float, true);
        if (itg == ctx->wcache.end()) { if(getenv("ORK_VERBOSE"))fprintf(stderr,"[FFN i16] FAIL: gate resolve miss %s\n",Wg->name); ok = false; }
        else { const ork_weight & owg = itg->second; const float * bsg = owg.bscale.data();
            // NOTE: an earlier single-core pin here (ork_npu_set_core_budget 1) collides with (c)'s NO_BF
            // default — it forces the gate matmul onto the single-core run_loop path which needs the full-K
            // Bf (NULL under NO_BF) -> run_loop wedges (errno 110). Keep the gate MULTI-core (mcworker/Bb).
            // The int16 silu op is single-core; the multi->single transition is handled inside the op.
            for (int m0 = 0; m0 < M && ok; m0 += MT) { int mc = std::min(MT, M - m0);
                if (ork_i8_mm_run(ctx->npu, owg.w, mc, xr_i8.data() + (size_t) m0*K, acc_i32.data())) { if(getenv("ORK_VERBOSE"))fprintf(stderr,"[FFN i16] FAIL: gate matmul owg m0=%d\n",m0); ok = false; break; }
                if (fc.silu_i16) {
                    // on-NPU int16 SiLU: dequant int32 gate -> fp32 (per-row x per-channel scale), quantize to
                    // int16 (uniform scale), run ork_i16_npu_silu, dequant. ~325x more accurate than int8-fused.
                    std::vector<float> gr((size_t) mc*Nff); float gmx = 1e-9f;
                    for (int r = 0; r < mc; r++) { const int32_t * ar = acc_i32.data() + (size_t) r*Nff; const float rs = as_x[m0+r];
                        for (int n = 0; n < Nff; n++) { float g = (float) ar[n] * rs * bsg[n]; gr[(size_t) r*Nff+n] = g; float a = fabsf(g); if (a > gmx) gmx = a; } }
                    double is = gmx/32000.0, os = gmx/32000.0; if (is<=0) is=1e-9; if (os<=0) os=1e-9;   // silu(gmax)~gmax
                    std::vector<int16_t> in16((size_t) mc*Nff), out16((size_t) mc*Nff);
                    for (size_t i = 0; i < (size_t) mc*Nff; i++) { long q = lround(gr[i]/is); if (q>32767) q=32767; if (q<-32768) q=-32768; in16[i] = (int16_t) q; }
                    // N-TILE the int16 SiLU: the standalone SDP op rejects N>nmax(8192) and does NOT tile
                    // internally, but the FFN intermediate is Nff=18944. SiLU is pointwise, so gather each
                    // contiguous ≤8192 column-chunk into [mc][cw], run it, scatter back (Nff%8==0 -> cw%8==0).
                    const int NT16 = 8192;
                    std::vector<int16_t> cin((size_t) mc*NT16), cout((size_t) mc*NT16);
                    for (int n0 = 0; n0 < Nff && ok; n0 += NT16) { int cw = std::min(NT16, Nff - n0);
                        for (int r = 0; r < mc; r++) memcpy(cin.data() + (size_t) r*cw, in16.data() + (size_t) r*Nff + n0, (size_t) cw*2);
                        double us = 0;
                        if (ork_i16_npu_silu(ctx->npu, cin.data(), mc, cw, is, os, cout.data(), &us)) { if(getenv("ORK_VERBOSE"))fprintf(stderr,"[FFN i16] FAIL: ork_i16_npu_silu mc=%d cw=%d is=%.3e\n",mc,cw,is); ok = false; break; }
                        for (int r = 0; r < mc; r++) { float * so = silu_f.data() + (size_t)(m0+r)*Nff + n0;
                            for (int n = 0; n < cw; n++) so[n] = (float) cout[(size_t) r*cw+n] * os; } }
                    if (!ok) break;
                } else {
                    for (int r = 0; r < mc; r++) { const int32_t * ar = acc_i32.data() + (size_t) r*Nff;
                        float * so = silu_f.data() + (size_t)(m0+r)*Nff; const float rs = as_x[m0+r];
                        for (int n = 0; n < Nff; n++) { float g = (float) ar[n] * rs * bsg[n]; so[n] = g / (1.0f + expf(-g)); } }
                } }
            }
    } else if (ok && !fc.wg_f16.empty() && ablate < 0) {
        // ORK_FFN_GATE_F16: precise fp16 gate matmul + fused fp16 SiLU, N-CHUNKED. x -> fp16 (cast); each chunk's
        // gate is -S*Wg (baked in prep) so acc = -S*gate spreads the fp16 LUT; silu = C_out * fc.f16_out. Each
        // chunk runs as its own single-tile ork_f16_mm_run_silu (Sn==1 per weight) into its silu_f columns.
        std::vector<ork_f16> xh((size_t) M * K);
        for (size_t i = 0; i < (size_t) M*K; i++) xh[i] = (ork_f16) xf[i];
        const int cn = fc.wg_f16_cn;
        std::vector<float> ctmp((size_t) MT * cn);        // per-(M-tile,chunk) fp16-silu output [mc][cw], scattered into silu_f
        // fc.f16_cpusilu: raw-Wg fp16 gate matmul + EXACT CPU fp32 silu (the fused fp16 SiLU LUT is broken,
        // PPL 16742; the fp16 MATMUL is coherent, 18.13). up/down stay int8 (fast prefill). Plain fp16
        // EINVALs a COLD multi-core submit, so pin the gate to single-core (mirror the f16_all cpusilu path).
        if (fc.f16_cpusilu) ork_npu_set_core_budget(ctx->npu, 1);
        for (int m0 = 0; m0 < M && ok; m0 += MT) { int mc = std::min(MT, M - m0);
            for (size_t ci = 0; ci < fc.wg_f16.size() && ok; ci++) {
                int n0 = (int) ci * cn, cw = (Nff - n0 < cn) ? (Nff - n0) : cn;
                if (fc.f16_cpusilu) {   // raw fp16 gate -> fp32 acc, then exact CPU silu
                    if (ork_f16_mm_run(ctx->npu, fc.wg_f16[ci], mc, xh.data() + (size_t) m0*K, ctmp.data())) { ok = false; break; }
                    for (int r = 0; r < mc; r++) { float * so = silu_f.data() + (size_t)(m0+r)*Nff + n0;
                        const float * cr = ctmp.data() + (size_t) r*cw;
                        for (int n = 0; n < cw; n++) { float g = cr[n]; so[n] = g / (1.0f + expf(-g)); } }
                } else {                // fused fp16 SiLU LUT (BROKEN — kept only for A/B)
                    if (ork_f16_mm_run_silu(ctx->npu, fc.wg_f16[ci], mc, xh.data() + (size_t) m0*K, ctmp.data(),
                                            0, 0xffffc000u, 0x56391100u, fc.lut_f16.data(), 1030)) { ok = false; break; }
                    for (int r = 0; r < mc; r++) { float * so = silu_f.data() + (size_t)(m0+r)*Nff + n0;
                        const float * cr = ctmp.data() + (size_t) r*cw;
                        for (int n = 0; n < cw; n++) so[n] = cr[n] * (float) fc.f16_out; }
                }
            }
        }
        if (fc.f16_cpusilu) ork_npu_set_core_budget(ctx->npu, ork_npu_cores(ctx->npu));
    } else if (ok && ablate < 0) {
        // normal fused path: silu on NPU -> int8 -> *s_silu
        std::vector<int8_t> silu_i8((size_t) M * Nff);
        for (int m0 = 0; m0 < M && ok; m0 += MT) { int mc = std::min(MT, M - m0);
            if (ork_i8_mm_run_silu(ctx->npu, fc.wg, mc, xi.data() + (size_t) m0*K, silu_i8.data() + (size_t) m0*Nff,
                                   RM, RS, OB, IO, C4, lut, 1030)) { ok = false; break; } }
        for (size_t i = 0; i < (size_t) M*Nff; i++) silu_f[i] = (float) silu_i8[i] * (float) fc.s_silu;
    } else if (ok) {
        // ablation gate: matmul (int32) + host silu at the selected precision
        const bool perchan = (ablate == 0);
        const int8_t * gx = nullptr; std::vector<int8_t> xsr; std::vector<float> as_xs;
        ork_w * gw = nullptr; const float * bsg = nullptr; double sg_pt = 0;
        if (perchan) {   // mode 0: per-channel Wg + per-row ORIGINAL x
            auto itg = ork_resolve_weight_i8(ctx, Wg, K, Nff, Wg->nb[1], Wg->type, ggml_get_type_traits(Wg->type)->to_float, true);
            if (itg == ctx->wcache.end()) return false;
            gw = itg->second.w; bsg = itg->second.bscale.data(); gx = xr_i8.data();
        } else if (ablate == 1) {   // mode 1: smoothed per-tensor Wg + per-ROW smoothed x
            xsr.resize(nx); as_xs.resize(M);
            for (int m = 0; m < M; m++) { const float * xr = xf + (size_t) m*K; int8_t * a = xsr.data() + (size_t) m*K;
                float mx = 1e-9f; for (int k = 0; k < K; k++) { float v = fabsf(xr[k]/fc.s[k]); if (v > mx) mx = v; }
                as_xs[m] = mx/127.0f; float iv = 127.0f/mx;
                for (int k = 0; k < K; k++) { int q=(int)lrintf((xr[k]/fc.s[k])*iv); a[k]=(int8_t)(q>127?127:q<-127?-127:q); } }
            gw = fc.wg; sg_pt = fc.sg; gx = xsr.data();
        } else {   // modes 2,3: smoothed per-tensor Wg + per-tensor STATIC smoothed x (xi)
            gw = fc.wg; sg_pt = fc.sg; gx = xi.data();
        }
        for (int m0 = 0; m0 < M && ok; m0 += MT) { int mc = std::min(MT, M - m0);
            if (ork_i8_mm_run(ctx->npu, gw, mc, gx + (size_t) m0*K, acc_i32.data())) { ok = false; break; }
            for (int r = 0; r < mc; r++) { const int32_t * ar = acc_i32.data() + (size_t) r*Nff;
                float * so = silu_f.data() + (size_t)(m0+r)*Nff;
                const float rrow = perchan ? as_x[m0+r] : (ablate==1 ? as_xs[m0+r] : (float) fc.s_x);
                for (int n = 0; n < Nff; n++) { float g = (float) ar[n] * rrow * (perchan ? bsg[n] : (float) sg_pt);
                    so[n] = g / (1.0f + expf(-g)); } }
        }
        if (ok && ablate == 3) {   // source #3: round-trip silu through per-tensor int8
            float mx = 1e-9f; for (size_t i = 0; i < (size_t) M*Nff; i++) { float v = fabsf(silu_f[i]); if (v > mx) mx = v; }
            float s = mx/127.0f, iv = 127.0f/mx;
            for (size_t i = 0; i < (size_t) M*Nff; i++) { int q=(int)lrintf(silu_f[i]*iv); if(q>127)q=127; if(q<-128)q=-128; silu_f[i]=(float)q*s; }
        }
    }
    // Phase 3: glu = silu_f * up_f
    if (ok) for (size_t i = 0; i < (size_t) M*Nff; i++) glu_f[i] = silu_f[i] * up_f[i];
    if (ok) {
        // down: per-row quant of glu_f -> int8, per-channel int8 matmul -> int32 -> dequant per-row*per-channel
        for (int m = 0; m < M; m++) { const float * gr = glu_f.data() + (size_t) m*Nff; int8_t * a = glr_i8.data() + (size_t) m*Nff;
            float mx = 1e-9f; for (int n = 0; n < Nff; n++) { float v = fabsf(gr[n]); if (v > mx) mx = v; }
            as_g[m] = mx / 127.0f; float iv = 127.0f / mx;
            for (int n = 0; n < Nff; n++) { int q = (int) lrintf(gr[n]*iv); a[n] = (int8_t)(q>127?127:q<-127?-127:q); } }
        std::vector<int32_t> down_i32((size_t) M * Kd);
        for (int m0 = 0; m0 < M && ok; m0 += MT) { int mc = std::min(MT, M - m0);
            if (ork_i8_mm_run(ctx->npu, owd.w, mc, glr_i8.data() + (size_t) m0*Nff, down_i32.data() + (size_t) m0*Kd)) { ok = false; break; } }
        if (ok) {
            float * d = (float *) down_n->data;
            for (int m = 0; m < M; m++) { const int32_t * cr = down_i32.data() + (size_t) m*Kd; float * dr = d + (size_t) m*Kd;
                const float rs = as_g[m]; for (int n = 0; n < Kd; n++) dr[n] = rs * bsd[n] * (float) cr[n]; }
            return true;
        }
    }

    // ---- CPU fp32 fallback: recompute the whole FFN inner exactly (output always valid) ----
    if (getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK FFN-CHAIN] NPU chain failed -> CPU fallback\n");
    // gate = Wg^T x ; up = Wu^T x ; glu = silu(gate)*up ; down = Wd^T glu   (all fp32, dequant weights on the fly)
    const auto * ttg = ggml_get_type_traits(Wg->type); ggml_to_float_t g2f = ttg->to_float;
    std::vector<float> fbglu((size_t) M * Nff);
    // dequant each gate/up output-channel plane once, accumulate ag/au, apply silu*up
    std::vector<float> wg_row(K), wu_row(K);
    for (int n = 0; n < Nff; n++) {
        g2f((const char *) Wg->data + (size_t) n * Wg->nb[1], wg_row.data(), K);
        g2f((const char *) Wu->data + (size_t) n * Wu->nb[1], wu_row.data(), K);
        for (int m = 0; m < M; m++) {
            const float * xr = xf + (size_t) m * K;
            float ag = 0, au = 0;
            for (int k = 0; k < K; k++) { ag += wg_row[k]*xr[k]; au += wu_row[k]*xr[k]; }
            fbglu[(size_t) m * Nff + n] = (ag / (1.0f + expf(-ag))) * au;
        }
    }
    const auto * ttd = ggml_get_type_traits(Wd->type); ggml_to_float_t d2f = ttd->to_float;
    std::vector<float> wd_row(Nff);
    float * d = (float *) down_n->data;
    for (int n = 0; n < Kd; n++) {
        d2f((const char *) Wd->data + (size_t) n * Wd->nb[1], wd_row.data(), Nff);
        for (int m = 0; m < M; m++) {
            const float * gr = fbglu.data() + (size_t) m * Nff;
            float acc = 0; for (int j = 0; j < Nff; j++) acc += wd_row[j]*gr[j];
            d[(size_t) m * Kd + n] = acc;
        }
    }
    return true;
}

// SwiGLU on the NPU: dst = silu(gate) (x) up  (gate=src0, up=src1, both f32 [N=ne0, rows]).
// Quantize per-tensor to int8, silu(gate) via ork_i8_npu_silu, (x)up via ork_i8_npu_ewmul (gain 1/128 so
// up_i8*silu_i8/128 fits int8), dequant with s_glu = s_up*s_silu*128. int8 (lossy) — the wiring foothold
// for the on-NPU FFN inner; the round-trip-free int8-chain (matmul int8-out feeding this) is the next step.
// CPU fp32 fallback on any NPU error, so output is always correct.
static bool ggml_backend_ork_glu(ggml_backend_ork_context * ctx, struct ggml_tensor * dst) {
    const struct ggml_tensor * g = dst->src[0];
    const struct ggml_tensor * u = dst->src[1];
    const int64_t ne = ggml_nelements(dst);
    const int N = (int) dst->ne[0], M = (int) (ne / N);
    const float * gate = (const float *) g->data;
    const float * up   = (const float *) u->data;
    float * o = (float *) dst->data;
    if (N > 8192) {   // standalone NPU ewmul_i8 is capped at N<=8192; large-N GLUs (e.g. 7B Nff=18944) are the
        for (int64_t i = 0; i < ne; i++) { float x = gate[i]; o[i] = (x/(1.0f+expf(-x))) * up[i]; }  // chain's job
        return true;  // (this standalone path is only reached if the ffn_chain matcher missed the layer)
    }
    float amg = 1e-9f, amu = 1e-9f;
    for (int64_t i = 0; i < ne; i++) { float a = fabsf(gate[i]); if (a > amg) amg = a; a = fabsf(up[i]); if (a > amu) amu = a; }
    const double s_gate = amg/127.0, s_up = amu/127.0, s_silu = s_gate;
    std::vector<int8_t> gi((size_t)ne), ui((size_t)ne), si((size_t)ne), glu((size_t)ne);
    const float invg = 127.0f/amg, invu = 127.0f/amu;
    for (int64_t i = 0; i < ne; i++) {
        int v = (int)(gate[i]*invg + copysignf(0.5f, gate[i])); gi[i] = (int8_t)(v>127?127:v<-127?-127:v);
        v     = (int)(up[i]*invu   + copysignf(0.5f, up[i]));   ui[i] = (int8_t)(v>127?127:v<-127?-127:v);
    }
    double us = 0;
    int r1 = ork_i8_npu_silu(ctx->npu, (const signed char*)gi.data(), M, N, s_gate, s_silu, (signed char*)si.data(), &us);
    int r2 = r1 ? -1 : ork_i8_npu_ewmul(ctx->npu, ui.data(), si.data(), M, N, 0x4000, 21, glu.data(), &us);
    if (r1 == 0 && r2 == 0) {
        const double s_glu = s_up * s_silu * 128.0;
        for (int64_t i = 0; i < ne; i++) o[i] = (float)(glu[i] * s_glu);
        return true;
    }
    for (int64_t i = 0; i < ne; i++) { float x = gate[i]; o[i] = (x/(1.0f+expf(-x))) * up[i]; }  // CPU fallback
    return true;
}

// Compute-dtype tier (8=W8A8, 4=W4A4) a MUL_MAT node will run under the active dispatch policy —
// used to gate same-tier group fusion (only i8-tier matmuls concatenate into mul_mat_group_i8).
static int ork_node_qbits(const ggml_backend_ork_context * ctx, const struct ggml_tensor * node) {
    if (ork_mixed_dispatch_on()) {
        static const int w4a4 = env_enabled("ORK_MIXED_W4A4");
        double b = ork_src_type_bits(node->src[0]->type);
        return (w4a4 && b >= 0.0 && b < 5.0) ? 4 : 8;
    }
    if (ctx->hybrid) {
        const char * nm = node->src[0]->name;
        if (strstr(nm, "ffn_") || ork_is_expert(nm)) return 4;
        if (strstr(nm, "attn_q") || strstr(nm, "attn_k") || strstr(nm, "attn_v") || strstr(nm, "attn_output")) return 8;
    }
    return ctx->qbits;
}

// BATCHED / DYNAMIC MUL_MAT (attention QKᵀ·V, Gated-Delta-Net chunked matmuls) via ork_bmm_fp16. Unlike the
// static-weight path, BOTH operands are computed activations (ne[2]/ne[3]>1, src0 dynamic), so this converts
// each batch's src0/src1 slice to fp16 and calls the batched dynamic GEMM. src0 is the [K,N] "weight" side and
// src1 the [K,M] activations — the same mapping the static mul_mat_i8 uses (ork_bmm packs src0, streams src1).
// GQA/broadcast-aware (src0/src1 batch dims may be < dst's). Gated by ORK_ATTN. Result f32 into dst.
static bool ggml_backend_ork_bmm_fp16(ggml_backend_ork_context * ctx, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const int K = (int) src0->ne[0], N = (int) src0->ne[1], M = (int) src1->ne[1];
    const int64_t ne2 = dst->ne[2], ne3 = dst->ne[3];
    const int64_t r2_0 = src0->ne[2] ? ne2 / src0->ne[2] : 1, r3_0 = src0->ne[3] ? ne3 / src0->ne[3] : 1;
    const int64_t r2_1 = src1->ne[2] ? ne2 / src1->ne[2] : 1, r3_1 = src1->ne[3] ? ne3 / src1->ne[3] : 1;
    std::vector<ork_f16> A((size_t) M * K), B((size_t) K * N);
    std::vector<float>   C((size_t) M * N);
    // strided read of logical element (i0,i1) of a per-head slice — densifies non-contiguous (permuted-view)
    // attention operands (real QK^T/A·V read permuted-Q / non-contig KV-cache views) into the packed A/B.
    auto rds = [](const struct ggml_tensor * t, const char * base, int64_t i0, int64_t i1) -> float {
        const char * p = base + i0 * t->nb[0] + i1 * t->nb[1];
        return t->type == GGML_TYPE_F16 ? (float) *(const ork_f16 *) p : *(const float *) p;
    };
    // DETERMINISTIC IOVA fix: reuse ONE persistent scratch weight across all heads (repack in place per
    // head) instead of ork_bmm_fp16's per-head ork_f16_mm_pack (a fresh ~6MB dma-buf import + cache entry ×
    // heads×layers -> PRIME_FD / bcreate OOM). Footprint = one K×N weight + reused A/C, constant regardless
    // of head/layer count — same pool discipline as ork_ssm/gdn_scan. K%32/N%16 already gated by supports_op.
    // PER-RUN (not per-node) LIFETIME, and CO-DOMAIN: the scratch comes from the shared (dom,K,N)-keyed cache
    // and is NOT freed here. It used to be ork_f16_mm_scratch()+ork_mm_free() per handler call, i.e. per NODE
    // (~100-200 alloc/free cycles per forward pass). Two separate problems, only the second was the killer:
    //   (1) churn — a fresh bcreate GEM per node draws from the limited CONTIGUOUS/CMA pool that bscratch()
    //       documents as fragmenting under the resident dma-heap weights + orkpack mmap pressure. The cache
    //       fixes that, but on its own it did NOT change the failure count (measured 9 -> 9), so churn was
    //       NOT the cause.
    //   (2) DOMAIN PLACEMENT — the real cause. Byte-balanced placement put the scratch in a different IOMMU
    //       domain than the layer's resident int8 weights, so each bmm forced a `switch iommu domain` there
    //       and back; a nonblock doorbell job whose completion IRQ never fired pins the outgoing domain's
    //       refcount, and the switch then stalls the NEXT submit for the whole 60 s timeout. That is why the
    //       op that FAILS is never the fp16 bmm but the int8 matmul right AFTER it — measured, every time,
    //       ork_dyn_colsplit_ks on attn_output (K=4096 N=2048 dom=1 imported=1), errno 110, 3 events x 60 s
    //       == the entire 186 s pp64 wall. ork_get_f16_scratch now allocates in the ALREADY-ACTIVE domain, so
    //       the bmm adds zero switches. See the comment there for the full mechanism.
    // ggml_backend_ork_free releases all f16_scratch entries.
    ork_w * w = ork_get_f16_scratch(ctx, K, N);
    if (!w) return false;
    bool ok = true;
    for (int64_t i3 = 0; ok && i3 < ne3; i3++)
    for (int64_t i2 = 0; ok && i2 < ne2; i2++) {
        const char * s0 = (const char *) src0->data + (i2 / r2_0) * src0->nb[2] + (i3 / r3_0) * src0->nb[3];
        const char * s1 = (const char *) src1->data + (i2 / r2_1) * src1->nb[2] + (i3 / r3_1) * src1->nb[3];
        float      * d  = (float *)((char *) dst->data + i2 * dst->nb[2] + i3 * dst->nb[3]);
        // DEFECT FIX (was: `rds(src0, s0, j % K, j / K)`): ork_f16_mm_repack/ork_f16_mm_pack take B as
        // **[K,N] ROW-MAJOR**, i.e. B[k*N + n] = weight(k, n). ggml stores element (k,n) of a [K,N]
        // tensor at offset n*nb[1] + k*nb[0] — ne[0]-contiguous — so src0's raw buffer is [N][K],
        // i.e. Bᵀ. The old index `(j%K, j/K)` reproduced exactly that raw order, so every batched
        // matmul ran against a TRANSPOSED weight. Proven by emulation: feeding the driver Bᵀ
        // reproduces the NPU output to NRMSE 1.9e-4 (fp16 noise) at K=N=256, K≠N, and every M
        // (scratchpad bmm_probe). This is what took ork_ppl PPL 10.83 -> 67.86 with ORK_ATTN=1:
        // ork_ppl sets flash_attn_type = DISABLED, so attention arrives as batched MUL_MAT here
        // (QKᵀ and A·V), NOT as FLASH_ATTN_EXT — and ORK_ATTN_CPU only moves the FA handler's
        // matmuls, which is why forcing "both FA matmuls to CPU" left the PPL byte-identical.
        // Correct order: B[j] = src0(j / N, j % N)  =>  B[k*N + n] = src0(k, n).
        for (size_t j = 0; j < (size_t) K * N; j++) B[j] = (ork_f16) rds(src0, s0, j / N, j % N);   // B[k*N+n] = src0(k,n)
        for (size_t j = 0; j < (size_t) M * K; j++) A[j] = (ork_f16) rds(src1, s1, j % K, j / K);   // src1 [K,M]
        if (ork_f16_mm_repack(ctx->npu, w, K, N, B.data()) || ork_f16_mm_run(ctx->npu, w, M, A.data(), C.data())) {
            if (getenv("ORK_ATTN_TRACE")) fprintf(stderr, "[bmm] ^^^ FAILED M=%d K=%d N=%d head(%lld,%lld)\n",
                M, K, N, (long long)i2, (long long)i3);
            ok = false; break; }
        for (size_t j = 0; j < (size_t) M * N; j++) d[j] = C[j];
    }
    // NO ork_mm_free(w) — the (K,N) cache owns it (freed in ggml_backend_ork_free). Freeing per node is what
    // churned the contiguous/CMA pool and poisoned later int8 submits (see the comment above).
    return ok;
}

// 2b: attention softmax on the NPU. ggml GGML_OP_SOFT_MAX is soft_max_ext = softmax(scale*x + mask) per
// row over ne[0]. We apply scale+mask on the CPU (cheap) into an fp16 buffer, then ork_f16_npu_softmax
// runs exp() on the NPU (int16 SDP act-LUT) with max/sum/divide on CPU. Masked entries (causal -inf)
// are floored to -64 (exp(-64)~0) so the primitive's int16 quant stays finite. supports_op guarantees
// fp32 in/out, ne[1]>1, ne[0]%32==0, max_bias==0 — so this always completes.
static bool ggml_backend_ork_soft_max(ggml_backend_ork_context * ctx, struct ggml_tensor * dst) {
    const struct ggml_tensor * src  = dst->src[0];
    const struct ggml_tensor * mask = dst->src[1];
    float scale = 1.0f; memcpy(&scale, (const char *) dst->op_params, sizeof(float));
    const int      ne0 = (int) src->ne[0];
    const int64_t  ne1 = src->ne[1], ne2 = src->ne[2], ne3 = src->ne[3];
    const int64_t  nrow = ne1 * ne2 * ne3;
    std::vector<ork_f16> xin((size_t) nrow * ne0), xout((size_t) nrow * ne0);
    for (int64_t i3 = 0; i3 < ne3; i3++)
    for (int64_t i2 = 0; i2 < ne2; i2++)
    for (int64_t i1 = 0; i1 < ne1; i1++) {
        const int64_t r = (i3 * ne2 + i2) * ne1 + i1;
        const float * s = (const float *) ((const char *) src->data + i1*src->nb[1] + i2*src->nb[2] + i3*src->nb[3]);
        ork_f16 * xo = xin.data() + (size_t) r * ne0;
        const char * mrow = nullptr; int mf16 = 0;
        if (mask) {                                   // mask broadcast over heads (ne2); indexed by token i1 (and batch i3)
            const int64_t mi1 = i1 % mask->ne[1];
            const int64_t mi3 = mask->ne[3] > 1 ? (i3 % mask->ne[3]) : 0;
            mrow = (const char *) mask->data + mi1*mask->nb[1] + mi3*mask->nb[3];
            mf16 = (mask->type == GGML_TYPE_F16);
        }
        for (int j = 0; j < ne0; j++) {
            float v = scale * s[j];
            if (mrow) v += mf16 ? (float) ((const ork_f16 *) mrow)[j] : ((const float *) mrow)[j];
            if (!(v > -64.0f)) v = -64.0f;            // causal -inf (and NaN) -> large negative (exp~0)
            xo[j] = (ork_f16) v;
        }
    }
    if (ork_f16_npu_softmax(ctx->npu, (int) nrow, ne0, xin.data(), xout.data())) return false;
    for (int64_t i3 = 0; i3 < ne3; i3++)
    for (int64_t i2 = 0; i2 < ne2; i2++)
    for (int64_t i1 = 0; i1 < ne1; i1++) {
        const int64_t r = (i3 * ne2 + i2) * ne1 + i1;
        float * d = (float *) ((char *) dst->data + i1*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3]);
        const ork_f16 * xo = xout.data() + (size_t) r * ne0;
        for (int j = 0; j < ne0; j++) d[j] = (float) xo[j];
    }
    return true;
}

// ORK_CLAIM_OPS (#1 toward the static graph): compute ONE claimed boundary op (RMSNorm/RoPE/residual-add)
// on the cached CPU backend, treating its srcs as already-computed leaves (no ancestor recompute). Claiming
// these ops keeps them inside ork's subgraph -> ggml hands ork the WHOLE layer contiguously, the prerequisite
// for the layer-level heterogeneous-chain assembler. Correctness comes from ggml's own CPU kernels; the
// on-NPU-native + chained version is the follow-up. Returns false on failure -> caller aborts (never claimed
// unless supports_op allowed it, so shapes are safe).
// ORK_OPS_NPU: run a claimed RoPE on the NPU (ork_f16_npu_rope_neox) instead of CPU-delegate, so Q/K
// rotation stays on the NPU data path. Only the Qwen3 case (NEOX mode, n_dims==head_dim, int32 positions,
// fp32 in/out, contiguous); anything else returns false -> caller CPU-delegates. src0=[hd, n_head, n_tok],
// src1=positions[n_tok]; row (head h, token t) -> pos = positions[t].
static bool ggml_backend_ork_rope_npu(ggml_backend_ork_context * ctx, struct ggml_tensor * node) {
    const struct ggml_tensor * x = node->src[0];
    const struct ggml_tensor * p = node->src[1];
    if (!x || !p || x->type != GGML_TYPE_F32 || node->type != GGML_TYPE_F32 || p->type != GGML_TYPE_I32) return false;
    if (!ggml_is_contiguous(x)) return false;
    const int n_dims = ((const int32_t *) node->op_params)[0];
    const int mode   = ((const int32_t *) node->op_params)[1];
    float freq_base = 10000.0f; memcpy(&freq_base, (const char *) node->op_params + 3*sizeof(int32_t), sizeof(float));
    const int hd = (int) x->ne[0], nh = (int) x->ne[1], nt = (int) x->ne[2];
    if (!(mode & 2) || n_dims != hd || (hd & 7) || x->ne[3] != 1) return false;   // NEOX, full-dim only
    const int nrow = nh * nt;
    const int32_t * pos = (const int32_t *) p->data;
    std::vector<ork_f16> xh((size_t) nrow*hd), oh((size_t) nrow*hd);
    std::vector<int>     rp((size_t) nrow);
    const float * xf = (const float *) x->data;
    for (int64_t t = 0; t < nt; t++) for (int64_t h = 0; h < nh; h++) { int64_t r = t*nh + h;
        for (int d = 0; d < hd; d++) xh[(size_t) r*hd + d] = (ork_f16) xf[(size_t)(t*nh + h)*hd + d];
        rp[r] = pos[t]; }
    if (ork_f16_npu_rope_neox(ctx->npu, xh.data(), hd, nrow, rp.data(), (double) freq_base, oh.data())) return false;
    float * of = (float *) node->data;
    for (int64_t r = 0; r < nrow; r++) for (int d = 0; d < hd; d++) of[(size_t) r*hd + d] = (float) oh[(size_t) r*hd + d];
    return true;
}

// ORK_OPS_NPU: RMSNorm on the NPU (ork_f16_npu_rmsnorm, norm-only — ggml's RMS_NORM has no weight; the
// weight is a separate MUL). fp32 in/out, contiguous, ne[0] a multiple of 8.
static bool ggml_backend_ork_rmsnorm_npu(ggml_backend_ork_context * ctx, struct ggml_tensor * node) {
    const struct ggml_tensor * x = node->src[0];
    if (!x || x->type != GGML_TYPE_F32 || node->type != GGML_TYPE_F32 || !ggml_is_contiguous(x)) return false;
    const int n = (int) x->ne[0]; const int64_t M = ggml_nrows(x);
    if (n < 8 || (n & 7)) return false;
    float eps = 1e-6f; memcpy(&eps, node->op_params, sizeof(float));
    std::vector<ork_f16> xh((size_t) M*n), oh((size_t) M*n), w((size_t) n, (ork_f16) 1.0f);
    const float * xf = (const float *) x->data;
    for (size_t j = 0; j < (size_t) M*n; j++) xh[j] = (ork_f16) xf[j];
    if (ork_f16_npu_rmsnorm(ctx->npu, (int) M, n, xh.data(), w.data(), eps, oh.data())) return false;
    float * of = (float *) node->data;
    for (size_t j = 0; j < (size_t) M*n; j++) of[j] = (float) oh[j];
    return true;
}

// ORK_OPS_NPU: elementwise MUL (norm-weight, broadcast-aware) / ADD (residual) on the NPU. src1 may broadcast
// over src0 (norm-weight w[n]); ggml modulo-broadcast into the fp16 operand. fp32 in/out, ne[0] mult of 8.
static bool ggml_backend_ork_binop_npu(ggml_backend_ork_context * ctx, struct ggml_tensor * node, bool is_add) {
    const struct ggml_tensor * a = node->src[0]; const struct ggml_tensor * b = node->src[1];
    if (!a || !b || a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || node->type != GGML_TYPE_F32) return false;
    if (!ggml_is_contiguous(a) || !ggml_is_contiguous(node)) return false;
    const int n = (int) node->ne[0]; const int64_t M = ggml_nrows(node); const size_t ne = (size_t) M*n;
    if (n < 8 || (n & 7)) return false;
    std::vector<ork_f16> ah(ne), bh(ne), oh(ne);
    const float * af = (const float *) a->data;
    for (size_t j = 0; j < ne; j++) ah[j] = (ork_f16) af[j];
    if (ggml_are_same_shape(a, b) && ggml_is_contiguous(b)) {
        const float * bf = (const float *) b->data; for (size_t j = 0; j < ne; j++) bh[j] = (ork_f16) bf[j];
    } else {   // modulo-broadcast b over node's shape
        for (int64_t i3 = 0; i3 < node->ne[3]; i3++) for (int64_t i2 = 0; i2 < node->ne[2]; i2++)
        for (int64_t i1 = 0; i1 < node->ne[1]; i1++) for (int64_t i0 = 0; i0 < node->ne[0]; i0++) {
            const size_t oi = (((size_t) i3*node->ne[2] + i2)*node->ne[1] + i1)*node->ne[0] + i0;
            const float * bp = (const float *) ((const char *) b->data + (i3 % b->ne[3])*b->nb[3]
                + (i2 % b->ne[2])*b->nb[2] + (i1 % b->ne[1])*b->nb[1] + (i0 % b->ne[0])*b->nb[0]);
            bh[oi] = (ork_f16) *bp;
        }
    }
    int rc = is_add ? ork_f16_npu_add(ctx->npu, ah.data(), bh.data(), (int) M, n, oh.data(), NULL)
                    : ork_f16_npu_ewmul(ctx->npu, ah.data(), bh.data(), (int) M, n, oh.data(), NULL);
    if (rc) return false;
    float * of = (float *) node->data; for (size_t j = 0; j < ne; j++) of[j] = (float) oh[j];
    return true;
}

static bool ork_cpu_delegate_node(ggml_backend_ork_context * ctx, struct ggml_tensor * node) {
    if (!ctx->cpu_backend) { ctx->cpu_backend = ggml_backend_cpu_init(); if (ctx->cpu_backend) ggml_backend_cpu_set_n_threads(ctx->cpu_backend, 4); }
    if (!ctx->cpu_backend) return false;
    const size_t mo = ggml_tensor_overhead() * (GGML_MAX_SRC + 2) + ggml_graph_overhead();
    struct ggml_init_params ip = { mo, nullptr, /*no_alloc=*/true };
    struct ggml_context * g = ggml_init(ip); if (!g) return false;
    struct ggml_tensor * op = ggml_new_tensor(g, node->type, GGML_MAX_DIMS, node->ne);
    op->op = node->op; memcpy(op->op_params, node->op_params, sizeof(op->op_params));
    for (int i = 0; i < GGML_MAX_DIMS; i++) op->nb[i] = node->nb[i];
    op->data = node->data; op->buffer = node->buffer;
    for (int s = 0; s < GGML_MAX_SRC; s++) {
        struct ggml_tensor * sr = node->src[s]; if (!sr) { op->src[s] = nullptr; continue; }
        struct ggml_tensor * lf = ggml_new_tensor(g, sr->type, GGML_MAX_DIMS, sr->ne);
        for (int i = 0; i < GGML_MAX_DIMS; i++) lf->nb[i] = sr->nb[i];
        lf->data = sr->data; lf->buffer = sr->buffer; lf->op = GGML_OP_NONE;
        op->src[s] = lf;
    }
    struct ggml_cgraph * gf = ggml_new_graph(g);
    ggml_build_forward_expand(gf, op);
    enum ggml_status st = ggml_backend_graph_compute(ctx->cpu_backend, gf);
    ggml_free(g);
    return st == GGML_STATUS_SUCCESS;
}

// ORK_SEG_TIME: per-op-category wall-time accounting for the segment-map table (measured, not estimated).
enum { SEG_QKV, SEG_QKT, SEG_SM, SEG_AV, SEG_O, SEG_FFN, SEG_OTH, SEG_N };
static double g_seg_us[SEG_N]; static long g_seg_n[SEG_N]; static int g_segtime = -1;
static const char * g_seg_name[SEG_N] = { "QKV proj", "QK^T scores", "softmax", "A.V", "O proj", "FFN gate/up/GLU/down", "other mul_mat" };
static inline double ork_seg_t0(void) { return g_segtime > 0 ? (double) ggml_time_us() : 0.0; }
static inline void ork_seg_add(int cat, double t0) { if (g_segtime > 0) { g_seg_us[cat] += (double) ggml_time_us() - t0; g_seg_n[cat]++; } }

// ORK_STATIC_GRAPH (assembler, step 1 — the planner). Recognize a contiguous ork layer subgraph (only
// possible once ORK_CLAIM_OPS pulls RMSNorm/RoPE/MUL/ADD into ork's hands) and classify each node into the
// static execution plan: HW-CHAIN groups of independent same-input matmuls (q/k/v, gate/up), SINGLE matmuls
// (O, down), and CPU-delegate steps (norm/rope/residual) between them. This is the front-end the executor +
// regcmd-precompile build on; here it just emits the plan (diagnostic, no execution change) so the segment
// structure is verified on real graphs before the executor rewrites how they run.
// ---- op-capability table: declarative, MEASURED heterogeneous placement / split / overlap -------------------
// Per-op metadata driving the ork_spine dispatcher (data, not branches — cf. ork-driver's XSPEC + SoC caps).
//   place   : the op's optimal unit, by MEASURED cost (tools/cpu_gemm_probe + tools/cpu_op_sweep, RK3588 2026-07-24).
//   split   : if breakable across units, how (NONE everywhere — the CPU is only ~6% of NPU at int8 GEMM, so
//             relocating matmul columns to the CPU has a ~1.06x ceiling: dead. No op is worth splitting today).
//   overlap : an IDENTIFIED, MEASURED overlap opportunity (OVL_NONE everywhere — dense prefill is fused +
//             latency-bound with no independent-work window; never auto-guessed, so it can't regress the fast path).
// MEASURED PLACEMENT (why each row): matmul = NPU 345 vs CPU 20 GMAC/s (17x) at prefill, but M=1 DECODE -> CPU
// (submit-bound, NPU no faster per-row). Standalone SDP/activation LUT ops (silu/gelu/rsqrt/exp) = ~106 ms/call
// on the NPU (per-call LUT calibrate+load) vs ~0.1-19 ms CPU => ~1000x, decisively CPU. add/mul/norm/softmax/rope
// = CPU (a submit round-trip or NEON-beatable). This CONFIRMS supports_op's current routing (matmul->NPU, rest
// ->CPU) is optimal; the table records the measured WHY + guards against re-placing a submit-heavy op on the NPU.
enum ork_place   { ORK_PLACE_CPU = 0, ORK_PLACE_NPU = 1, ORK_PLACE_EITHER = 2 };
enum ork_split   { ORK_SPLIT_NONE = 0, ORK_SPLIT_NCOLS = 1, ORK_SPLIT_MROWS = 2 };
enum ork_overlap { ORK_OVL_NONE = 0, ORK_OVL_RELOCATE_CPU = 1 };
struct ork_op_cap { enum ork_place place; enum ork_split split; enum ork_overlap overlap; };

static struct ork_op_cap ork_op_cap_for(const struct ggml_tensor * node) {
    struct ork_op_cap c = { ORK_PLACE_CPU, ORK_SPLIT_NONE, ORK_OVL_NONE };
    switch (node->op) {
        case GGML_OP_MUL_MAT: case GGML_OP_MUL_MAT_ID:
            c.place = (node->ne[1] > 1) ? ORK_PLACE_NPU : ORK_PLACE_CPU;   // prefill(M>1)->NPU; decode(M=1)->CPU
            break;
        // SDP / activation / norm / elementwise: measured CPU-optimal (NPU LUT-reload ~106ms or a submit floor).
        case GGML_OP_UNARY: case GGML_OP_GLU: case GGML_OP_RMS_NORM: case GGML_OP_NORM:
        case GGML_OP_SOFT_MAX: case GGML_OP_ROPE: case GGML_OP_MUL: case GGML_OP_ADD:
        case GGML_OP_SCALE: case GGML_OP_FLASH_ATTN_EXT:
            c.place = ORK_PLACE_CPU;
            break;
        default:   // metadata (NONE/RESHAPE/VIEW/…) + anything unmeasured -> CPU
            c.place = ORK_PLACE_CPU;
            break;
    }
    return c;
}
static const char * ork_place_name(enum ork_place p){ return p==ORK_PLACE_NPU?"NPU":p==ORK_PLACE_EITHER?"EITHER":"CPU"; }

static void ork_log_static_plan(struct ggml_cgraph * cgraph) {
    fprintf(stderr, "[ORK STATIC-PLAN] %d-node subgraph:\n", cgraph->n_nodes);
    int seg = 0;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * n = cgraph->nodes[i];
        const char * nm = n->src[0] ? n->src[0]->name : "";
        struct ork_op_cap cap = ork_op_cap_for(n);   // measured optimal placement (op-capability table)
        if (n->op == GGML_OP_MUL_MAT) {
            // look ahead for independent same-input siblings (the HW-chain group)
            int grp = 1;
            while (i + grp < cgraph->n_nodes) {
                struct ggml_tensor * nj = cgraph->nodes[i + grp];
                if (nj->op == GGML_OP_MUL_MAT && nj->src[1] == n->src[1] && nj->src[0]->ne[0] == n->src[0]->ne[0]) grp++;
                else break;
            }
            fprintf(stderr, "  S%-2d  %-9s x%d  %-26s K=%ld N=%ld M=%ld  opt=%s [%s]\n", seg++,
                    grp > 1 ? "HW-CHAIN" : "matmul", grp, nm,
                    (long) n->src[0]->ne[0], (long) n->ne[0], (long) n->ne[1],
                    ork_place_name(cap.place),
                    grp > 1 ? "run_chain_i8" : "run_i8");
            i += grp - 1;
        } else if (n->op == GGML_OP_SOFT_MAX) {
            fprintf(stderr, "  S%-2d  softmax   (exp on NPU)\n", seg++);
        } else {
            fprintf(stderr, "  --   %-9s %-26s  [CPU-delegate]\n", ggml_op_name(n->op), nm);
        }
    }
}

// GGML_OP_SSM_SCAN (Mamba-2) -> ork_ssm_scan_f32 (chunked mode-5: NPU pooled 3-core matmul stream + CPU
// elementwise). Thin marshaller: srcs are contiguous fp32 + ns==1 (guaranteed by supports_op); y + s_new
// are written into the contiguous dst ([y | s_new] at s_off). Falls back to CPU on failure.
static bool ggml_backend_ork_ssm_scan(ggml_backend_ork_context * ctx, struct ggml_tensor * dst) {
    const struct ggml_tensor *S=dst->src[0],*X=dst->src[1],*DT=dst->src[2],*A=dst->src[3],*B=dst->src[4],*C=dst->src[5];
    const int nc=(int)S->ne[0], nr=(int)S->ne[1], nh=(int)X->ne[1], ng=(int)B->ne[1], nt=(int)X->ne[2], ns=(int)X->ne[3];
    const size_t s_off=(size_t)ggml_nelements(X)*sizeof(float);
    int rc=ork_ssm_scan_f32(ctx->npu, nc,nr,nh,ng,nt,ns,
        (const float*)S->data,(const float*)X->data,(const float*)DT->data,(const float*)A->data,
        (const float*)B->data,(const float*)C->data,
        (float*)dst->data,(float*)((char*)dst->data+s_off));
    return rc==0;
}

// GGML_OP_FLASH_ATTN_EXT -> fused attention on the NPU. Batched QK^T + A·V via the fused-multicore fp16
// stream (ONE submit each for all heads — slashes the per-op submit floor that sank the per-op path),
// softmax on CPU (scale+mask). Persistent scratch pool (no per-op bcreate/import -> no IOVA OOM). GQA-aware.
// Materializes scores (mathematically == the online CPU flash form). nkv padded to %32 (pad cols masked out).
// v1: no alibi (max_bias) / no softcap — supports_op gates those to CPU. Gated ORK_ATTN. Keep-warm carries
// the matmul<->matmul (and future SDP) transitions. Falls back to CPU (ret false) on any shape/alloc issue.
// (ork_attn_pool / ork_attn_sm are defined above the context struct; the pool lives PER-CONTEXT in ctx->attnp.)
// LEGAL fp16 M-TILE for the stream/chain primitives, and the M-tiling wrapper the FA handler needs.
//
// DEFECT (fixed here): ork_f16_mm_run_stream / _chain synth ONE regcmd program per task — they are
// documented "Single M-tile (the SSD scan is M<=64 <= one tile)" and do NOT M-schedule. The rows of a
// program are only computed against the right K-partition while M stays inside the 0x1040 K-reduction
// schedule's validated tile: CBUF_CON0 = base - slope*(mg-1) must stay >= 0x1b, with mg = ceil(M/64),
// base/slope derived from K (npu.c synth(), and npu.c run() which does tile to exactly mg_max*64).
// Past that ceiling the excess rows are SILENTLY WRONG — no error, no submit failure.
//   K=DK=256   -> mg_max=11 -> 704 rows      (QKᵀ: fine at ubatch 512)
//   K=nkvp=512 -> mg_max=5  -> 320 rows      (A·V at n_kv 512: WRONG for M>~320)
//   K=nkvp=1024-> mg_max=2  -> 128 rows      (A·V at n_kv 1024: WRONG for M>128)
// The handler passed M = n_tokens straight through, so at ubatch 512 the A·V output was garbage from
// row ~352 on. Measured with scratchpad fa_probe (CPU-backend oracle, DK=DV=256 H=16 Hkv=2):
//   n_kv 512, nb 64/128/256 PASS (NRMSE 2e-4); nb 384/512 FAIL from query 352 (NRMSE 30-69)
//   n_kv 1024, nb 512 FAIL from query 0 (NRMSE 153); ORK_ATTN_CPU=2 (A·V on CPU) PASS => A·V only.
// Fix: chunk M to ork_f16_mtile(K) and issue one chain submit per chunk (A/C advance by their row
// strides). Mirrors npu.c run()'s own M-scheduler. Cost: 2 submits instead of 1 at ubatch 512.
static int ork_f16_mtile(int K) {
    const int sched = (K & (K - 1)) == 0 && K >= 128 && K < 2048;   // same gate as synth()/run()
    if (!sched) { int t = (32768 / 2) / (K > 0 ? K : 1); return t < 1 ? 1 : t; }
    const double scale = (double) K / 256.0;                        // fp16 reference K (int8 uses 512)
    const int base = (int) (177.0 - 15.0 * (scale - 1.0)), slope = (int) (15.0 * scale);
    const int mg_max = (slope > 0 && base >= 0x1b) ? (base - 0x1b) / slope + 1 : 1;
    const int t = mg_max * 64;
    return t < 1 ? 1 : t;
}
// M-tile-safe ork_f16_mm_run_stream_chain. All tasks must share the same M (they do here: one task per
// head). K = the weight's contraction dim (A row stride), Nout = the weight's N (C row stride).
static int ork_attn_chain_mtiled(ork_npu * npu, int S, const ork_mm_task_f16 * tk, int K, int Nout) {
    const int M = tk[0].M, cap = ork_f16_mtile(K);
    if (M <= cap) return ork_f16_mm_run_stream_chain(npu, S, tk);
    std::vector<ork_mm_task_f16> sub((size_t) S);
    for (int m0 = 0; m0 < M; m0 += cap) {
        const int mm = (M - m0 < cap) ? (M - m0) : cap;
        for (int i = 0; i < S; i++)
            sub[i] = (ork_mm_task_f16){ tk[i].w, mm,
                                        tk[i].A + (size_t) m0 * K,
                                        tk[i].C + (size_t) m0 * Nout };
        if (ork_f16_mm_run_stream_chain(npu, S, sub.data())) return -1;
    }
    return 0;
}
static void attn_pool_free_one(ggml_backend_ork_context * ctx, struct ork_attn_pool * P) {
    ork_npu * c = ctx->npu;
    if (P->wqk) for (int h=0; h<P->H; h++) { if (P->wqk[h]) ork_mm_free(c, P->wqk[h]); if (P->wav && P->wav[h]) ork_mm_free(c, P->wav[h]); if (P->wet && P->wet[h]) ork_mm_free(c, P->wet[h]); }
    free(P->wqk); free(P->wav); free(P->wet); free(P->Qf); free(P->KT); free(P->Vf); free(P->Pf); free(P->Oh16);
    free(P->scores); free(P->outf); free(P->invS); free(P->tk);
    int dom = P->dom;
    *P = (struct ork_attn_pool){0};
    P->dom = dom;   // keep the domain: a pool object stays bound to its map key even when empty
}
static void attn_pool_free(ggml_backend_ork_context * ctx) {   // teardown: every domain's pool
    for (auto & kv : ctx->attnp) attn_pool_free_one(ctx, &kv.second);
    ctx->attnp.clear();
}
// CO-DOMAIN fused-attention pool (the ORK_ATTN submit-timeout fix). The pool's fp16 QK^T / A·V scratch
// weights are TRANSIENT — unlike a resident weight they have no natural home domain, so they must be placed
// where they cost the LEAST, and the cost that matters is not bytes, it is IOMMU DOMAIN SWITCHES.
//
// Previously this called ork_f16_mm_scratch() with no domain control at all, so the scratch landed in
// ork_dom(c->pack_domain) = wherever the weight loader last left the pack cursor (the LAST domain it filled),
// and there was ONE pool shared by every attention layer. Every attention layer whose int8 weights live in a
// different domain therefore forced two `switch iommu domain` events (into the pool's domain and back).
// A switch waits for the outgoing domain's job refcount to reach 0, and a nonblock doorbell job whose
// completion IRQ never fired holds that refcount forever — so the switch stalls the NEXT submit for the full
// 60 s submit timeout (kernel: "commit elapse time: 61.2s" then "job timeout"), after which ork self-heals
// with a reset. That is the entire ORK_ATTN collapse: 2-3 such events x 60 s == the whole ~125-186 s pp64
// wall (measured), which is why the op that FAILS is never the attention matmul but the int8 matmul right
// AFTER it — every time ork_dyn_colsplit_ks on attn_output (K=4096 N=2048 dom=1 imported=1), errno 110.
// ork-driver already has reap-at-boundary machinery for exactly this (dom_dirty / ork_dom_flush_if_dirty)
// but it is armed only for int4 doorbell drops.
//
// Fix: one pool PER DOMAIN, allocated in the domain that is ALREADY ACTIVE when that layer's attention runs.
// The FA node sits between the same layer's int8 attn_q/k/v and attn_output, which are co-domain by the
// layer-aligned residence rule, so the active domain IS that layer's domain -> attention adds ZERO switches
// and the run switches exactly as often as the (0-failure) ORK_ATTN=0 baseline.
// Returns the pool for the active domain, or NULL on a shape/alloc failure (caller falls back to CPU).
static struct ork_attn_pool * attn_pool_ensure(ggml_backend_ork_context * ctx, int DK, int DV, int N, int nkvp, int H) {
    ork_npu * c = ctx->npu;
    const int adom = ork_npu_active_domain(c);
    struct ork_attn_pool * P = &ctx->attnp[adom];   // creates a zeroed pool for this domain on first use
    P->dom = adom;
    if (P->wqk && P->DK==DK && P->DV==DV && P->N==N && P->nkvp==nkvp && P->H==H) return P;   // warm reuse
    attn_pool_free_one(ctx, P);
    // Pin the scratch allocation to THIS domain (ork_f16_mm_scratch stamps w->domain from pack_domain), then
    // restore the loader's cursor so a later weight pack is unaffected.
    const int saved = ork_npu_pack_domain(c);
    ork_npu_set_pack_domain(c, adom);
    P->wqk = (ork_w**)calloc(H, sizeof(ork_w*)); P->wav = (ork_w**)calloc(H, sizeof(ork_w*)); P->wet = (ork_w**)calloc(H, sizeof(ork_w*));
    if (!P->wqk || !P->wav || !P->wet) { ork_npu_set_pack_domain(c, saved); attn_pool_free_one(ctx, P); return nullptr; }
    for (int h=0; h<H; h++) { P->wqk[h] = ork_f16_mm_scratch(c, DK, nkvp); P->wav[h] = ork_f16_mm_scratch(c, nkvp, DV);
        P->wet[h] = ork_f16_mm_scratch(c, nkvp, N);   // transposed A·V weight = e^T[nkvp][N]
        if (!P->wqk[h] || !P->wav[h] || !P->wet[h]) { P->H=H; ork_npu_set_pack_domain(c, saved); attn_pool_free_one(ctx, P); return nullptr; } }
    ork_npu_set_pack_domain(c, saved);
    P->Qf = (ork_f16*)malloc((size_t)H*N*DK*2); P->KT = (ork_f16*)malloc((size_t)H*DK*nkvp*2);
    P->Vf = (ork_f16*)malloc((size_t)H*nkvp*DV*2); P->Pf = (ork_f16*)malloc((size_t)H*N*nkvp*2);
    P->Oh16 = (ork_f16*)malloc((size_t)H*DV*N*2);
    P->scores = (float*)malloc((size_t)H*N*nkvp*4); P->outf = (float*)malloc((size_t)H*DV*N*4);
    P->invS = (float*)malloc((size_t)H*N*4);
    P->tk = (ork_mm_task_f16*)malloc((size_t)H*sizeof(ork_mm_task_f16));
    if (!P->Qf||!P->KT||!P->Vf||!P->Pf||!P->Oh16||!P->scores||!P->outf||!P->invS||!P->tk) { P->H=H; attn_pool_free_one(ctx, P); return nullptr; }
    P->DK=DK; P->DV=DV; P->N=N; P->nkvp=nkvp; P->H=H;
    if (getenv("ORK_ATTN_TRACE"))
        fprintf(stderr, "[attnpool] NEW pool dom=%d DK=%d DV=%d N=%d nkvp=%d H=%d (pools=%zu)\n",
                adom, DK, DV, N, nkvp, H, ctx->attnp.size());
    return P;
}
// Resident-KV variant of the int8 decode attention (ORK_ATTN_KV): instead of packing K^T/V every call
// (the O(nkv)/token repack tax that makes the default path perf-negative), pack ONCE per (layer, kv-head)
// via ork_kv_resident_alloc, then append just the new key/value each token (ork_kv_append — one tile write,
// no repack). Keyed on k->data (the per-layer K-cache view base, stable across decode steps). Per-head int8
// scales are FIXED at the first pack (appended keys reuse them — a v1 tradeoff: a later key that exceeds the
// initial abs-max clips). The matmuls (QK^T, e.V) run against the resident weights; softmax stays host-fp.
static bool ggml_backend_ork_flash_attn_decode_kv(ggml_backend_ork_context * ctx, struct ggml_tensor * dst) {
    const struct ggml_tensor *q=dst->src[0],*k=dst->src[1],*v=dst->src[2],*mask=dst->src[3];
    const int DK=(int)q->ne[0], H=(int)q->ne[2], nkv_pad=(int)k->ne[1], Hkv=(int)k->ne[2], DV=(int)v->ne[0];
    const int rk2 = H/Hkv, Kp=512, LmaxCap=2048;   // cap matches the supports_op gate (nkv<=2048); RK3588 nmax=8192
    if (DK!=DV || DK>Kp || nkv_pad>LmaxCap || nkv_pad<1) return false;   // API bundles ONE HD for K^T & V; single N-tile
    float scale=1.0f; memcpy(&scale,(char*)dst->op_params+0,4);
    ork_npu *c = ctx->npu;
    auto rdf = [](const struct ggml_tensor * t, int64_t i0,int64_t i1,int64_t i2,int64_t i3) -> float {
        const char * p=(const char*)t->data + i0*t->nb[0]+i1*t->nb[1]+i2*t->nb[2]+i3*t->nb[3];
        return t->type==GGML_TYPE_F16 ? (float)*(const ork_f16*)p : *(const float*)p; };
    // k->ne[1] is the FULL/padded KV-cache width, NOT the live token count — it stays constant as generated
    // tokens fill the cache, so "append when nkv grows" would never fire and the resident KV would go stale
    // (the repack path dodges this by re-reading the whole cache every call). Derive the live count from the
    // mask: valid keys have a finite mask value, padding/future keys are -inf. Valid region is contiguous
    // [0,nkv) for causal decode. No mask -> assume the whole (padded) width is live.
    int nkv = nkv_pad;
    if (mask) {
        auto mval = [&](int j) -> float { const char * p=(const char*)mask->data + (size_t)j*mask->nb[0];
            return mask->type==GGML_TYPE_F16 ? (float)*(const ork_f16*)p : *(const float*)p; };
        nkv = 0; for (int j=0;j<nkv_pad;j++) if (mval(j) > -1e30f) nkv = j+1;
    }
    if (nkv < 1 || nkv > LmaxCap) return false;
    { static long n=0; if ((n++ % 256)==0) fprintf(stderr,"[ork-attn-kv] resident decode: live_nkv=%d (pad=%d) H=%d Hkv=%d DK=%d\n", nkv, nkv_pad, H, Hkv, DK); }
    auto & LY = ctx->attn_kv[(const void*)k->data];
    // ORK_ATTN_FUSED: run the whole attention core as Hkv fused chains fanned RR across cores in ONE orkd
    // round-trip (vs the 2-submit QK^T + host-softmax + e.V path). Needs GLOBAL K/Q scales (the RR dispatch
    // shares one exp LUT) + a resident ones[Lmax,32] reduce weight.
    static int fused_env=-1; if(fused_env<0) fused_env=getenv("ORK_ATTN_FUSED")?1:0;
    // (re)alloc on first touch, a sequence reset (cache shrank), a head-count change, OR growth past the
    // current resident width. Lmax is sized to the live nkv rounded to a 512-chunk (NOT the 2048 cap) so the
    // QK^T/e.V matmuls stream only ~nkv-wide weights (weight-DMA-bound) instead of always 2048 — growth
    // re-packs all keys but only crosses a 512 boundary a few times over a full generation. Scales refreshed
    // from the current [0,nkv) abs-max at each (re)pack so all keys quantize consistently.
    if (LY.kv.empty() || nkv < LY.packed || LY.Hkv != Hkv || nkv > LY.Lmax || LY.fused != fused_env) {
        for (ork_kv_resident * r : LY.kv) if (r) ork_kv_resident_free(c, r);
        if (LY.ones) { ork_mm_free(c, LY.ones); LY.ones = nullptr; }
        int Lm = (nkv+511)&~511; if (Lm < 32) Lm = 32; if (Lm > LmaxCap) Lm = LmaxCap;
        LY.kv.assign(Hkv, nullptr); LY.ks.assign(Hkv, 1.0f); LY.vs.assign(Hkv, 1.0f);
        LY.packed=0; LY.Lmax=Lm; LY.Hkv=Hkv; LY.DK=DK; LY.DV=DV; LY.fused=fused_env;
        float kmax_g=1e-6f;
        for (int hkv=0; hkv<Hkv; hkv++) {
            LY.kv[hkv] = ork_kv_resident_alloc(c, DV, Lm);
            if (!LY.kv[hkv]) { for (ork_kv_resident * r : LY.kv) if (r) ork_kv_resident_free(c, r); LY.kv.clear(); return false; }
            float kmax=1e-6f, vmax=1e-6f;
            for (int j=0;j<nkv;j++){ for(int e=0;e<DK;e++){ float a=fabsf(rdf(k,e,j,hkv,0)); if(a>kmax)kmax=a; }
                                     for(int e=0;e<DV;e++){ float a=fabsf(rdf(v,e,j,hkv,0)); if(a>vmax)vmax=a; } }
            LY.ks[hkv]=127.0f/kmax; LY.vs[hkv]=127.0f/vmax; if(kmax>kmax_g)kmax_g=kmax;   // vs stays per-head (host dequant)
        }
        if (fused_env) {   // GLOBAL K scale (shared exp LUT ⇒ qs·ks must be constant across chains) + resident ones[Lm,32]
            float ksg=127.0f/kmax_g; for (int hkv=0; hkv<Hkv; hkv++) LY.ks[hkv]=ksg;
            std::vector<int8_t> ones((size_t)Lm*32, 1); LY.ones=ork_i8_mm_pack(c, Lm, 32, ones.data());
            if (!LY.ones) { for (ork_kv_resident * r : LY.kv) if (r) ork_kv_resident_free(c, r); LY.kv.clear(); return false; }
        }
    }
    // per-stage timers (ORK_ATTN_PROF): where the decode-attention wall goes — append(+first-fill) / Q-quant /
    // QK^T orkd submit / host softmax / e.V orkd submit. Answers pack-bound vs submit(round-trip)-bound.
    static int aprof=-1; if(aprof<0) aprof=getenv("ORK_ATTN_PROF")?1:0;
    static double a_app=0,a_qq=0,a_qk=0,a_sm=0,a_av=0; static long a_n=0,a_appk=0; double _pt;
    // append newly-cached keys [packed, nkv) to each kv-head (1/step steady state; the whole prompt at first touch)
    _pt = aprof?ork_now_us():0;
    if (nkv > LY.packed) {
        int8_t kcol[512], vrow[512];
        for (int hkv=0; hkv<Hkv; hkv++) { float ks=LY.ks[hkv], vs=LY.vs[hkv];
            for (int j=LY.packed; j<nkv; j++) {
                for (int e=0;e<DK;e++) kcol[e]=(int8_t)lrintf(rdf(k,e,j,hkv,0)*ks);
                for (int e=0;e<DV;e++) vrow[e]=(int8_t)lrintf(rdf(v,e,j,hkv,0)*vs);
                if (ork_kv_append(c, LY.kv[hkv], j, kcol, vrow)) return false;
                if (aprof) a_appk++;
            } }
        LY.packed = nkv;
    }
    if (aprof) a_app += ork_now_us()-_pt;
    if (LY.fused && LY.ones) {
        // ORK_ATTN_FUSED: fan the Hkv attention chains [QK^T→exp((s-bias))→reduce,e·V] across the NPU cores in
        // ONE orkd round-trip. GLOBAL qs/ks ⇒ one shared exp LUT; in_scale/r_mult derived so score_i8·in_scale ==
        // the true logit (scale·QK^T), so the exp curve is calibrated from known host scales (no data pass).
        // max_bias keeps args ≤0 (int8 exp never saturates); padding keys [nkv,Lmax) have QK^T=0 ⇒ a constant
        // e_pad subtracted from Σ (av is unaffected — padding V=0). Falls through to the 2-submit path on error.
        const int Kp2=512, Nk=LY.Lmax, dv=DV, Nq=rk2;
        float qmax_g=1e-6f; for(int h=0;h<H;h++) for(int e=0;e<DK;e++){ float a=fabsf(rdf(q,e,0,h,0)); if(a>qmax_g)qmax_g=a; }
        double qs=127.0/qmax_g, ks=LY.ks[0];
        static double insc=-1, biasv=-1;
        if(insc<0){ const char*e=getenv("ORK_ATTN_INSCALE"); insc=e?atof(e):0.0625; const char*b=getenv("ORK_ATTN_BIAS"); biasv=b?atof(b):127.0; }
        double out_scale=1.0/127.0;
        int r_shift=16; double F=(double)scale/(qs*ks)/insc;        // score_i8 = raw·F ⇒ score·insc = scale·QK^T
        long rm=llround(F*(double)(1LL<<r_shift));
        while(rm<64 && r_shift<30){ r_shift++; rm=llround(F*(double)(1LL<<r_shift)); }
        while(rm>(1L<<22) && r_shift>2){ r_shift--; rm=llround(F*(double)(1LL<<r_shift)); }
        if(rm<1)rm=1; int r_mult=(int)rm;
        int8_t  *Qall =(int8_t*) calloc((size_t)Hkv*rk2*Kp2,1);
        int32_t *ssall=(int32_t*)calloc((size_t)Hkv*rk2*32,4), *avall=(int32_t*)calloc((size_t)Hkv*rk2*dv,4);
        std::vector<ork_w*> wkt(Hkv), wv(Hkv);
        if (Qall && ssall && avall) {
            for(int hkv=0;hkv<Hkv;hkv++){ wkt[hkv]=LY.kv[hkv]->wkt; wv[hkv]=LY.kv[hkv]->wv; }
            for(int h=0;h<H;h++){ int hkv=h/rk2, qh=h%rk2; int8_t *dq=Qall+((size_t)hkv*rk2+qh)*Kp2;
                for(int e=0;e<DK;e++) dq[e]=(int8_t)lrintf(rdf(q,e,0,h,0)*(float)qs); }
            _pt=aprof?ork_now_us():0;
            int rc=ork_mm_attn_rr_orkd(c, Hkv, wkt.data(), LY.ones, wv.data(), Nq, Nk, Kp2, dv,
                                       r_mult, r_shift, insc, out_scale, biasv, Qall, ssall, avall);
            if(aprof) a_qk+=ork_now_us()-_pt;
            if(rc==0){
                double epad=exp((0.0-biasv)*insc)/out_scale; long ep=lround(epad); if(ep<0)ep=0; if(ep>127)ep=127;
                long padc=(long)Nk-nkv;
                for(int h=0;h<H;h++){ int hkv=h/rk2, qh=h%rk2; float vs=LY.vs[hkv];
                    int32_t *ss=ssall+((size_t)hkv*rk2+qh)*32, *av=avall+((size_t)hkv*rk2+qh)*dv;
                    double S=(double)ss[0]-(double)padc*ep; if(S<=0)S=1;
                    for(int e=0;e<dv;e++) ((float*)dst->data)[(size_t)h*dv+e]=(float)((double)av[e]/((double)vs*S)); }
                free(Qall);free(ssall);free(avall);
                if(aprof){ a_n++; if(a_n%256==0) fprintf(stderr,"[ork-attn-fused] Hkv=%d Nq=%d Nk=%d r_mult=%d r_shift=%d in_scale=%.4f bias=%.0f\n",Hkv,Nq,Nk,r_mult,r_shift,insc,biasv); }
                return true;
            }
            fprintf(stderr,"[ork-attn-fused] rr rc=%d -> fallback to 2-submit path\n", rc);
        }
        free(Qall);free(ssall);free(avall);   // fall through to the batched path (global ks is self-consistent there)
    }
    // BATCHED submits: all H heads' QK^T go in ONE ork_i8_mm_run_chain (one orkd round-trip), then host
    // softmax, then all H e.V in ONE chain. Collapses the 2*H per-head submits/layer -> 2 — the profiled
    // 87% submit wall. Per-head buffers are contiguous [H][...]; GQA heads share their kv-head's resident wkt/wv.
    int8_t  *Qp =(int8_t*) calloc((size_t)H*Kp,1),      *w8  =(int8_t*) calloc((size_t)H*LmaxCap,1);
    int32_t *scores=(int32_t*)malloc((size_t)H*LmaxCap*4), *attv=(int32_t*)malloc((size_t)H*DV*4);
    double  *sc =(double*)malloc((size_t)nkv*sizeof(double));
    float   *qsA=(float*) malloc((size_t)H*4),          *wsA =(float*) malloc((size_t)H*4);
    ork_mm_task_i8 *t1=(ork_mm_task_i8*)malloc((size_t)H*sizeof(ork_mm_task_i8));
    ork_mm_task_i8 *t2=(ork_mm_task_i8*)malloc((size_t)H*sizeof(ork_mm_task_i8));
    bool ok=false;
    if (!Qp||!w8||!scores||!attv||!sc||!qsA||!wsA||!t1||!t2) goto done;
    // Phase 1: quantize Q per head + build the QK^T task list
    _pt=aprof?ork_now_us():0;
    for (int h=0; h<H; h++) { int hkv=h/rk2;
        float qmax=1e-6f; for (int e=0;e<DK;e++){ float a=fabsf(rdf(q,e,0,h,0)); if(a>qmax)qmax=a; }
        float qs=127.0f/qmax; qsA[h]=qs;
        int8_t *Qh=Qp+(size_t)h*Kp; for (int e=0;e<DK;e++) Qh[e]=(int8_t)lrintf(rdf(q,e,0,h,0)*qs);
        t1[h]=(ork_mm_task_i8){LY.kv[hkv]->wkt,1,Qh,scores+(size_t)h*LmaxCap};
    }
    if(aprof){ a_qq+=ork_now_us()-_pt; _pt=ork_now_us(); }
    if (ork_i8_mm_run_chain(c, H, t1)) goto done;                     // ONE submit: all H QK^T
    if(aprof){ a_qk+=ork_now_us()-_pt; _pt=ork_now_us(); }
    // Phase 2: host softmax per head (per-head max-subtraction) -> int8 weights + e.V task list
    for (int h=0; h<H; h++) { int hkv=h/rk2; float qs=qsA[h], ks=LY.ks[hkv];
        int32_t *sco=scores+(size_t)h*LmaxCap; int8_t *w=w8+(size_t)h*LmaxCap;
        double mx=-1e300; for(int j=0;j<nkv;j++){ sc[j]=(double)sco[j]/((double)qs*ks)*scale; if(sc[j]>mx)mx=sc[j]; }
        double Z=0; for(int j=0;j<nkv;j++){ sc[j]=exp(sc[j]-mx); Z+=sc[j]; } if(Z<=0)Z=1;
        double wmax=0; for(int j=0;j<nkv;j++){ sc[j]/=Z; if(sc[j]>wmax)wmax=sc[j]; } double ws=127.0/(wmax>1e-9?wmax:1.0); wsA[h]=(float)ws;
        for(int j=0;j<nkv;j++){ int wi=(int)lrint(sc[j]*ws); w[j]=(int8_t)(wi>127?127:(wi<0?0:wi)); }
        t2[h]=(ork_mm_task_i8){LY.kv[hkv]->wv,1,w,attv+(size_t)h*DV};
    }
    if(aprof){ a_sm+=ork_now_us()-_pt; _pt=ork_now_us(); }
    if (ork_i8_mm_run_chain(c, H, t2)) goto done;                     // ONE submit: all H e.V
    if(aprof) a_av+=ork_now_us()-_pt;
    // Phase 3: dequant + write
    for (int h=0; h<H; h++) { float ws=wsA[h], vs=LY.vs[h/rk2]; int32_t *av=attv+(size_t)h*DV;
        for(int e=0;e<DV;e++) ((float*)dst->data)[(size_t)h*DV+e]=(float)((double)av[e]/(ws*vs)); }
    if (aprof) { a_n++; if (a_n%256==0) { double T=a_app+a_qq+a_qk+a_sm+a_av; if(T<1)T=1;
        fprintf(stderr,"[ork-attn-kv PROF] layer-calls=%ld total-appends=%ld | append+fill %.0fms(%.0f%%) Qquant %.0fms(%.0f%%) QKT-submit %.0fms(%.0f%%) softmax %.0fms(%.0f%%) AV-submit %.0fms(%.0f%%) | attn-tot %.0fms\n",
            a_n,a_appk, a_app/1e3,100*a_app/T, a_qq/1e3,100*a_qq/T, a_qk/1e3,100*a_qk/T, a_sm/1e3,100*a_sm/T, a_av/1e3,100*a_av/T, T/1e3); } }
    ok=true;
done:
    free(Qp);free(w8);free(scores);free(attv);free(sc);free(qsA);free(wsA);free(t1);free(t2);
    return ok;
}
// GGML_OP_FLASH_ATTN_EXT, DECODE (N==1) -> int8 attention on the NPU under orkd. The fp16 path above is
// prefill-only (N>=64) and can't run under orkd (its scratch bcreate needs a direct fd). This fills the decode
// gap: per query head, QK^T and the weighted e.V run int8 on the NPU (via the orkd-routed matmul), the [1,nkv]
// softmax runs in fp on the HOST (real per-head max-subtraction -> correct for arbitrary scores). Opt-in
// ORK_ATTN_DEC; gated to nkv in [256, 2048] (int8 0x1040 sched floor .. single-N-tile). v1 packs K^T/V per call
// (perf-negative — the resident-KV append via ork_kv_* is the next step); this proves orkd-native decode
// attention is COHERENT first. Returns true on success (supports_op guarantees the shape is handled).
static bool ggml_backend_ork_flash_attn_decode(ggml_backend_ork_context * ctx, struct ggml_tensor * dst) {
    const struct ggml_tensor *q=dst->src[0],*k=dst->src[1],*v=dst->src[2];
    const int DK=(int)q->ne[0], H=(int)q->ne[2], nkv=(int)k->ne[1], Hkv=(int)k->ne[2], DV=(int)v->ne[0];
    const int rk2 = H/Hkv, Kp=512, Lp=(nkv+511)&~511;
    if (DK>Kp || Lp>2048) return false;
    { static long n=0; if ((n++ % 256)==0) fprintf(stderr,"[ork-attn-dec] NPU decode attention engaged (call %ld): H=%d Hkv=%d DK=%d DV=%d nkv=%d\n", n, H, Hkv, DK, DV, nkv); }
    // ORK_ATTN_KV: resident-KV (pack-once + append/token) — the perf path. ORK_ATTN_KV_REPACK forces the
    // per-call repack below (default) for A/B. Resident needs DK==DV (the ork_kv_* API bundles one HD).
    static const int kv_res = getenv("ORK_ATTN_KV_REPACK") ? 0 : (getenv("ORK_ATTN_KV") ? 1 : 0);
    if (kv_res && DK==DV) return ggml_backend_ork_flash_attn_decode_kv(ctx, dst);
    float scale=1.0f; memcpy(&scale,(char*)dst->op_params+0,4);
    ork_npu *c = ctx->npu;
    auto rdf = [](const struct ggml_tensor * t, int64_t i0,int64_t i1,int64_t i2,int64_t i3) -> float {
        const char * p=(const char*)t->data + i0*t->nb[0]+i1*t->nb[1]+i2*t->nb[2]+i3*t->nb[3];
        return t->type==GGML_TYPE_F16 ? (float)*(const ork_f16*)p : *(const float*)p; };
    int8_t *Qp=(int8_t*)calloc((size_t)Kp,1), *KTp=(int8_t*)calloc((size_t)Kp*Lp,1),
           *Vp=(int8_t*)calloc((size_t)Lp*DV,1), *w8=(int8_t*)calloc((size_t)Lp,1);
    int32_t *scores=(int32_t*)malloc((size_t)Lp*4), *attv=(int32_t*)malloc((size_t)DV*4);
    double *sc=(double*)malloc((size_t)Lp*sizeof(double));
    bool ok=false;
    if (!Qp||!KTp||!Vp||!w8||!scores||!attv||!sc) goto done;
    for (int h=0; h<H; h++) { int hkv=h/rk2;
        float qmax=1e-6f,kmax=1e-6f,vmax=1e-6f;
        for (int e=0;e<DK;e++){ float a=fabsf(rdf(q,e,0,h,0)); if(a>qmax)qmax=a; }
        for (int j=0;j<nkv;j++){ for(int e=0;e<DK;e++){ float a=fabsf(rdf(k,e,j,hkv,0)); if(a>kmax)kmax=a; }
                                 for(int e=0;e<DV;e++){ float a=fabsf(rdf(v,e,j,hkv,0)); if(a>vmax)vmax=a; } }
        float qs=127.0f/qmax, ks=127.0f/kmax, vs=127.0f/vmax;
        for (int e=0;e<DK;e++){ Qp[e]=(int8_t)lrintf(rdf(q,e,0,h,0)*qs); for(int j=0;j<nkv;j++) KTp[(size_t)e*Lp+j]=(int8_t)lrintf(rdf(k,e,j,hkv,0)*ks); }
        ork_w *wkt=ork_i8_mm_pack(c,Kp,Lp,KTp); if(!wkt) goto done;
        ork_mm_task_i8 t1={wkt,1,Qp,scores}; int rc=ork_i8_mm_run_chain(c,1,&t1); ork_w_free(wkt); if(rc) goto done;
        double mx=-1e300; for(int j=0;j<nkv;j++){ sc[j]=(double)scores[j]/((double)qs*ks)*scale; if(sc[j]>mx)mx=sc[j]; }
        double Z=0; for(int j=0;j<nkv;j++){ sc[j]=exp(sc[j]-mx); Z+=sc[j]; } if(Z<=0)Z=1;
        double wmax=0; for(int j=0;j<nkv;j++){ sc[j]/=Z; if(sc[j]>wmax)wmax=sc[j]; } double ws=127.0/(wmax>1e-9?wmax:1.0);
        for(int j=0;j<nkv;j++){ int wi=(int)lrint(sc[j]*ws); w8[j]=(int8_t)(wi>127?127:(wi<0?0:wi)); }
        for(int j=0;j<nkv;j++)for(int e=0;e<DV;e++) Vp[(size_t)j*DV+e]=(int8_t)lrintf(rdf(v,e,j,hkv,0)*vs);
        ork_w *wv=ork_i8_mm_pack(c,Lp,DV,Vp); if(!wv) goto done;
        ork_mm_task_i8 t2={wv,1,w8,attv}; rc=ork_i8_mm_run_chain(c,1,&t2); ork_w_free(wv); if(rc) goto done;
        for(int e=0;e<DV;e++) ((float*)dst->data)[(size_t)h*DV+e]=(float)((double)attv[e]/(ws*vs));
    }
    ok=true;
done:
    free(Qp);free(KTp);free(Vp);free(w8);free(scores);free(attv);free(sc);
    return ok;
}
static bool ggml_backend_ork_flash_attn_ext(ggml_backend_ork_context * ctx, struct ggml_tensor * dst) {
    if ((int)dst->src[0]->ne[1] == 1) return ggml_backend_ork_flash_attn_decode(ctx, dst);   // decode -> int8 orkd path
    const struct ggml_tensor *q=dst->src[0],*k=dst->src[1],*v=dst->src[2],*mask=dst->src[3];
    const int DK=(int)q->ne[0], N=(int)q->ne[1], H=(int)q->ne[2], Bt=(int)q->ne[3];
    const int nkv=(int)k->ne[1], Hkv=(int)k->ne[2], DV=(int)v->ne[0];
    float scale=1.0f,max_bias=0.0f,softcap=0.0f;
    memcpy(&scale,(char*)dst->op_params+0,4); memcpy(&max_bias,(char*)dst->op_params+4,4); memcpy(&softcap,(char*)dst->op_params+8,4);
    if (max_bias!=0.0f || softcap!=0.0f) return false;                 // v1: no ALiBi / softcap
    if (Hkv<1 || H%Hkv) return false;
    const int rk2 = H/Hkv;
    const int nkvp = (nkv+31)&~31;                                     // pad KV to %32 for the fp16 stream (K%32/N%16)
    if (DK%32 || DV%16) return false;
    struct ork_attn_pool * P = attn_pool_ensure(ctx, DK, DV, N, nkvp, H);   // co-domain with the active domain
    if (!P) return false;
    auto rdf = [](const struct ggml_tensor * t, int64_t i0,int64_t i1,int64_t i2,int64_t i3) -> float {
        const char * p=(const char*)t->data + i0*t->nb[0]+i1*t->nb[1]+i2*t->nb[2]+i3*t->nb[3];
        return t->type==GGML_TYPE_F16 ? (float)*(const ork_f16*)p : *(const float*)p; };
    static int prof=-1; if(prof<0) prof = getenv("ORK_ATTN_PROF")?1:0;   // per-stage timing for the bottleneck flow chart
    static double a_dqk=0,a_qk=0,a_sm=0,a_dv=0,a_av=0,a_sc=0; static long a_calls=0; double _t;
    static double a_smp=0,a_smq=0,a_sme=0,a_smr=0,a_smn=0;                // softmax sub-steps: prep / quant / exp-NPU / reduce-NPU / normalize
    // PER-CONTEXT softmax scratch (was function-`static` -> aliased into ctx->attn_sm so concurrent ork
    // backends don't share it). References keep the 42 downstream sm_* uses byte-identical + warm-reused.
    int16_t *&sm_xi=ctx->attn_sm.xi, *&sm_ei=ctx->attn_sm.ei; float *&sm_mx=ctx->attn_sm.mx, *&sm_ss=ctx->attn_sm.ss; ork_f16 *&sm_e=ctx->attn_sm.e; size_t &sm_cap=ctx->attn_sm.cap;
    int8_t *&sm_q8=ctx->attn_sm.q8; size_t &sm_cap2=ctx->attn_sm.cap2; int8_t *sm_maxq=ctx->attn_sm.maxq;   // int8 scores + per-row max for on-NPU max-reduce
    ork_f16 *&sm_invf=ctx->attn_sm.invf; int &sm_invf_n=ctx->attn_sm.invf_n;   // fp16 1/Σ per-head vector for the transposed per-channel normalize
    ork_w *&sm_ones=ctx->attn_sm.ones; int &sm_ones_n=ctx->attn_sm.ones_n;               // reduce-matmul weight ones[nkvp,16] for the on-NPU Sigma
    for (int b=0; b<Bt; b++) {
        // (1) densify per-head fp16 operands + QK^T weights (K^T), batched submit -> scores_h[N,nkvp] f32
        _t=ork_now_us();
        for (int h=0; h<H; h++) { int hkv=h/rk2;
            ork_f16 *Qh=P->Qf+(size_t)h*N*DK, *KTh=P->KT+(size_t)h*DK*nkvp;
            for (int m=0;m<N;m++)  for (int e=0;e<DK;e++) Qh[(size_t)m*DK+e]=(ork_f16)rdf(q,e,m,h,b);
            for (int e=0;e<DK;e++) for (int j=0;j<nkvp;j++) KTh[(size_t)e*nkvp+j]=(j<nkv)?(ork_f16)rdf(k,e,j,hkv,b):(ork_f16)0.0f;
            if (ork_f16_mm_repack(ctx->npu,P->wqk[h],DK,nkvp,KTh)) return false;
            P->tk[h]=(ork_mm_task_f16){P->wqk[h],N,Qh,P->scores+(size_t)h*N*nkvp}; }
        if(prof) a_dqk+=ork_now_us()-_t;
        const char *dbg_e = getenv("ORK_ATTN_CPU"); const int dbg_cpu = dbg_e ? atoi(dbg_e) : 0;   // bit0=QK^T on CPU, bit1=A·V on CPU
        _t=ork_now_us();
        if (dbg_cpu & 1) { for (int h=0;h<H;h++){ ork_f16 *Qh=P->Qf+(size_t)h*N*DK, *KTh=P->KT+(size_t)h*DK*nkvp; float *sc=P->scores+(size_t)h*N*nkvp;
            for (int m=0;m<N;m++) for (int j=0;j<nkvp;j++){ float a=0; for (int e=0;e<DK;e++) a+=(float)Qh[(size_t)m*DK+e]*(float)KTh[(size_t)e*nkvp+j]; sc[(size_t)m*nkvp+j]=a; } } }
        else if (ork_attn_chain_mtiled(ctx->npu,H,P->tk,DK,nkvp)) return false;   // M-tiled: see ork_f16_mtile
        if(prof) a_qk+=ork_now_us()-_t;
        // (2) softmax. DEFAULT IS NOW THE CPU SOFTMAX (scale+mask+max+exp+normalize, fp32) — it is the only
        // one that is correct. The FUSED softmax-on-NPU chain (scale+mask+max CPU prep -> quantize -> exp on
        // the NPU via ork_i16_npu_exp, ONE batched SDP submit over all H*N rows, in-chain between the two
        // fp16 matmuls -> sum via reduce-matmul -> normalize) is OPT-IN under ORK_ATTN_SM_NPU because it is
        // QUALITY-BROKEN at some row counts: fa_probe (CPU-backend oracle) gives NRMSE 2e-4 at MR=H*N=8192
        // but 0.55-0.69 at MR=6144 and MR=4096 — a factor-1000 blowup, wrong from row 0, and IDENTICAL with
        // ORK_ATTN_TNORM_OFF / ORK_ATTN_MAX_CPU / ORK_ATTN_SM_SUMCPU all set, which pins it on
        // ork_i16_npu_exp itself (an int16 SDP act-LUT shape/row-count envelope issue, not this handler).
        // ORK_ATTN_SM_CPU is kept as an accepted no-op alias so existing invocations still mean "CPU softmax".
        static int sm_npu=-1; if(sm_npu<0) sm_npu = (getenv("ORK_ATTN_SM_NPU") && !getenv("ORK_ATTN_SM_CPU")) ? 1 : 0;
        static int tnorm=-1; if(tnorm<0) tnorm = getenv("ORK_ATTN_TNORM_OFF")?0:1;  // transposed on-NPU normalize (A·V transposed + per-channel 1/Σ)
        int tnorm_ok=0;
        _t=ork_now_us();
        if (sm_npu) {
            const int MR=H*N; const size_t NB=(size_t)MR*nkvp;
            if (sm_cap < NB) { free(sm_xi); free(sm_ei); free(sm_mx); free(sm_e); free(sm_ss);
                sm_xi=(int16_t*)malloc(NB*2); sm_ei=(int16_t*)malloc(NB*2); sm_mx=(float*)malloc((size_t)MR*4);
                sm_e=(ork_f16*)malloc(NB*2); sm_ss=(float*)malloc(NB*4); sm_cap=NB; }
            // 2a: scale+mask into scores; per-row max ON NPU (ork_i8_npu_row_max, coarse int8 => softmax
            // max-invariant so exact-enough); then global min-delta lo. ORK_ATTN_MAX_CPU=1 forces CPU max.
            static int max_npu=-1; if(max_npu<0) max_npu = getenv("ORK_ATTN_MAX_CPU")?0:1;
            double _p=ork_now_us(); float lo=0.0f; const int MR2=H*N;
            // pass 1: scale+mask into scores (mask is data-dependent -> CPU)
            #pragma omp parallel for schedule(static)
            for (int h=0; h<H; h++) {
                const char * mbase = mask ? (const char*)mask->data + (h%(int)mask->ne[2])*mask->nb[2] + (b%(int)mask->ne[3])*mask->nb[3] : nullptr;
                float *sc=P->scores+(size_t)h*N*nkvp;
                for (int m=0;m<N;m++){ const ork_f16 * mp = mbase ? (const ork_f16*)(mbase + (size_t)m*mask->nb[1]) : nullptr;
                    for (int j=0;j<nkv;j++) sc[(size_t)m*nkvp+j]=scale*sc[(size_t)m*nkvp+j]+(mp?(float)mp[j]:0.0f); }
            }
            int have_npu_max=0;
            if (max_npu && MR2<=8192) {                          // per-row max on the NPU (coarse int8)
                if (sm_cap2 < NB) { free(sm_q8); sm_q8=(int8_t*)malloc(NB); sm_cap2=NB; }
                int8_t *maxq=sm_maxq;                             // MR2<=8192
                #pragma omp parallel for schedule(static)
                for (int r=0;r<MR2;r++){ float *sc=P->scores+(size_t)r*nkvp; int8_t *q=sm_q8+(size_t)r*nkvp;
                    for (int j=0;j<nkv;j++){ float s=sc[j]; q[j]=(int8_t)(s<-127.f?-128:s>127.f?127:(int)lrintf(s)); }
                    for (int j=nkv;j<nkvp;j++) q[j]=-128; }
                if (ork_i8_npu_row_max(ctx->npu, sm_q8, MR2, nkvp, maxq, NULL)==0){
                    for (int r=0;r<MR2;r++) sm_mx[r]=(float)maxq[r]; have_npu_max=1; }
            }
            if (!have_npu_max) {                                 // CPU max fallback (batch too big, or forced)
                #pragma omp parallel for schedule(static)
                for (int r=0;r<MR2;r++){ float *sc=P->scores+(size_t)r*nkvp; float mx=-INFINITY;
                    for (int j=0;j<nkv;j++) if(sc[j]>mx)mx=sc[j]; sm_mx[r]=mx; }
            }
            // global min-delta lo (for the exp int16 quant scale)
            #pragma omp parallel for schedule(static) reduction(min:lo)
            for (int r=0;r<MR2;r++){ float *sc=P->scores+(size_t)r*nkvp; float mx=sm_mx[r];
                for (int j=0;j<nkv;j++){ float d=sc[j]-mx; if(d<lo)lo=d; } }
            if(prof) a_smp+=ork_now_us()-_p;
            double in_scale=(-lo)/32000.0; if(in_scale<=0) in_scale=1e-6; double out_scale=1.0/32000.0;
            // 2b: quantize (x-max) -> int16; masked/pad cols -> -32768 (exp ~ 0)
            double _q=ork_now_us();
            #pragma omp parallel for schedule(static)
            for (int h=0; h<H; h++) for (int m=0;m<N;m++){ float *sc=P->scores+((size_t)h*N+m)*nkvp; int16_t *xi=sm_xi+((size_t)h*N+m)*nkvp; float mx=sm_mx[(size_t)h*N+m];
                for (int j=0;j<nkv;j++){ long qv=lround((sc[j]-mx)/in_scale); if(qv<-32768)qv=-32768; if(qv>32767)qv=32767; xi[j]=(int16_t)qv; }
                for (int j=nkv;j<nkvp;j++) xi[j]=-32768; }
            if(prof) a_smq+=ork_now_us()-_q;
            // 2c: exp on the NPU (single batched SDP submit, in-chain between the two matmuls)
            double _e=ork_now_us();
            int erc = ork_i16_npu_exp(ctx->npu, sm_xi, MR, nkvp, in_scale, out_scale, sm_ei, NULL);
            if(prof) a_sme+=ork_now_us()-_e;
            // 2c2: Sigma over the row ON the NPU via reduce-matmul e . ones[nkvp,16] (sum in col 0).
            // Everything-on-NPU: exp (SDP) -> sum (reduce-matmul) both on device; only max (2a) + the
            // per-row divide (2d) remain CPU. ORK_ATTN_SM_SUMCPU=1 keeps the sum on CPU for A/B.
            static int sumnpu=-1; if(sumnpu<0) sumnpu = getenv("ORK_ATTN_SM_SUMCPU")?0:1;
            int rrc=-1;
            if(!erc && sumnpu){
                if(sm_ones_n!=nkvp){ if(sm_ones) ork_mm_free(ctx->npu,sm_ones);
                    ork_f16 *on=(ork_f16*)malloc((size_t)nkvp*16*2); for(size_t i=0;i<(size_t)nkvp*16;i++) on[i]=(ork_f16)1.0f;
                    sm_ones=ork_f16_mm_pack(ctx->npu,nkvp,16,on); free(on); sm_ones_n=nkvp; }
                double _r=ork_now_us();
                // exp output ei (int16) -> fp16 e for the reduce matmul; pad cols already ~0 (exp of -32768)
                #pragma omp parallel for schedule(static)
                for (size_t i=0;i<(size_t)MR*nkvp;i++) sm_e[i]=(ork_f16)((double)sm_ei[i]*out_scale);
                if(sm_ones) rrc = ork_f16_mm_run(ctx->npu, sm_ones, MR, sm_e, sm_ss);   // sm_ss[MR,16], col0 = Sigma
                if(prof) a_smr+=ork_now_us()-_r;
            }
            // 2d: normalize -> Pf (pad cols 0). Sigma from NPU reduce (sm_ss) if it ran, else CPU sum.
            double _n=ork_now_us();
            #pragma omp parallel for schedule(static)
            for (int h=0; h<H; h++) for (int m=0;m<N;m++){ size_t r=(size_t)h*N+m; float *sc=P->scores+r*nkvp; int16_t *ei=sm_ei+r*nkvp; ork_f16 *Ph=P->Pf+r*nkvp; float mx=sm_mx[r];
                double sum;
                if(!rrc){ sum=sm_ss[r*16]; for (int j=0;j<nkv;j++) sc[j]=(float)((double)ei[j]*out_scale); }        // Sigma on NPU
                else if(!erc){ sum=0; for (int j=0;j<nkv;j++){ double e=(double)ei[j]*out_scale; sc[j]=(float)e; sum+=e; } } // exp NPU, sum CPU
                else { sum=0; for (int j=0;j<nkv;j++){ float e=expf(sc[j]-mx); sc[j]=e; sum+=e; } }                  // full CPU fallback
                float inv=(sum>0)?1.0f/(float)sum:0.0f;
                P->invS[r]=inv;                                    // for the transposed on-NPU normalize (1/Σ per query)
                for (int j=0;j<nkv;j++)  Ph[j]=(ork_f16)(sc[j]*inv);
                for (int j=nkv;j<nkvp;j++) Ph[j]=(ork_f16)0.0f; }
            if(prof) a_smn+=ork_now_us()-_n;
            tnorm_ok = tnorm && !erc && !rrc;   // transposed path needs unnormalized exp (sm_e) + Σ (sm_ss) from NPU
        } else {
            #pragma omp parallel for schedule(static)
            for (int h=0; h<H; h++) {
                const char * mbase = mask ? (const char*)mask->data + (h%(int)mask->ne[2])*mask->nb[2] + (b%(int)mask->ne[3])*mask->nb[3] : nullptr;
                float *sc=P->scores+(size_t)h*N*nkvp; ork_f16 *Ph=P->Pf+(size_t)h*N*nkvp;
                for (int m=0;m<N;m++) {
                    const ork_f16 * mp = mbase ? (const ork_f16*)(mbase + (size_t)m*mask->nb[1]) : nullptr;
                    float mx=-INFINITY;
                    for (int j=0;j<nkv;j++){ float s=scale*sc[(size_t)m*nkvp+j] + (mp?(float)mp[j]:0.0f); sc[(size_t)m*nkvp+j]=s; if(s>mx)mx=s; }
                    float sm=0; for (int j=0;j<nkv;j++){ float e=expf(sc[(size_t)m*nkvp+j]-mx); sc[(size_t)m*nkvp+j]=e; sm+=e; }
                    float inv=(sm>0)?1.0f/sm:0.0f;
                    for (int j=0;j<nkv;j++)  Ph[(size_t)m*nkvp+j]=(ork_f16)(sc[(size_t)m*nkvp+j]*inv);
                    for (int j=nkv;j<nkvp;j++) Ph[(size_t)m*nkvp+j]=(ork_f16)0.0f;
                }
            }
        }
        if(prof) a_sm+=ork_now_us()-_t;
        if (tnorm_ok) {
          // (3T) TRANSPOSED A·V: Ô[DV][N] = V^T[DV][nkvp] · e^T[nkvp][N] (queries become the output channel);
          // weight = e^T (repack per head, from unnormalized exp sm_e), activation = V^T (densified transposed).
          _t=ork_now_us();
          for (int h=0; h<H; h++) { int hkv=h/rk2;
            ork_f16 *VT=P->Vf+(size_t)h*DV*nkvp;                                 // V^T[DV][nkvp] activation
            for (int d=0;d<DV;d++) for (int j=0;j<nkvp;j++) VT[(size_t)d*nkvp+j]=(j<nkv)?(ork_f16)rdf(v,d,j,hkv,b):(ork_f16)0.0f;
            ork_f16 *eT=P->Pf+(size_t)h*N*nkvp, *es=sm_e+(size_t)h*N*nkvp;        // reuse Pf slot as e^T[nkvp][N]
            for (int j=0;j<nkvp;j++) for (int m=0;m<N;m++) eT[(size_t)j*N+m]=es[(size_t)m*nkvp+j];
            if (ork_f16_mm_repack(ctx->npu,P->wet[h],nkvp,N,eT)) return false;
            P->tk[h]=(ork_mm_task_f16){P->wet[h],DV,VT,P->outf+(size_t)h*DV*N}; } // C[DV][N] = Ô
          if(prof) a_dv+=ork_now_us()-_t;
          _t=ork_now_us();
          if (ork_attn_chain_mtiled(ctx->npu,H,P->tk,nkvp,N)) return false;       // M-tiled (M=DV here)
          if(prof) a_av+=ork_now_us()-_t;
          // (3.5) on-NPU per-channel normalize Ô[d][m]*=(1/Σ)[m]. BATCHED across heads into ONE submit:
          // lay Ô as [DV][H*N] (channel = head*N+query) so a single per-channel scale by invS[H*N] does all
          // heads. Falls back to per-head when H*N exceeds the op limits (>8192 or not %8).
          double _sn=ork_now_us();
          const int HN=H*N; const int batch=(HN<=8192 && (HN&7)==0);
          if (batch) {
            if (sm_invf_n < HN) { free(sm_invf); sm_invf=(ork_f16*)malloc((size_t)HN*2); sm_invf_n=HN; }
            #pragma omp parallel for schedule(static)
            for (int d=0; d<DV; d++) for (int h=0;h<H;h++) for (int m=0;m<N;m++)
              P->Oh16[(size_t)d*HN + (size_t)h*N + m] = (ork_f16)P->outf[(size_t)h*DV*N + (size_t)d*N + m];
            for (int h=0;h<H;h++) for (int m=0;m<N;m++) sm_invf[(size_t)h*N+m]=(ork_f16)P->invS[(size_t)h*N+m];
            if (ork_f16_npu_mul_perchan(ctx->npu,P->Oh16,sm_invf,DV,HN,P->Oh16,NULL)) return false;  // ONE submit, all heads
          } else {
            if (sm_invf_n < N) { free(sm_invf); sm_invf=(ork_f16*)malloc((size_t)N*2); sm_invf_n=N; }
            for (int h=0; h<H; h++) { float *o=P->outf+(size_t)h*DV*N; ork_f16 *o16=P->Oh16+(size_t)h*DV*N;
              for (size_t i=0;i<(size_t)DV*N;i++) o16[i]=(ork_f16)o[i];
              for (int m=0;m<N;m++) sm_invf[m]=(ork_f16)P->invS[(size_t)h*N+m];
              if (ork_f16_npu_mul_perchan(ctx->npu,o16,sm_invf,DV,N,o16,NULL)) return false; }
          }
          if(prof) a_smn+=ork_now_us()-_sn;
          // (4T) scatter: dst(d,h,m,b) = Ô_norm  (layout depends on batch vs per-head)
          _t=ork_now_us();
          if (batch) {
            #pragma omp parallel for schedule(static)
            for (int h=0; h<H; h++) for (int d=0;d<DV;d++) for (int m=0;m<N;m++)
              ((float*)dst->data)[(((size_t)b*N+m)*H+h)*DV+d] = (float)P->Oh16[(size_t)d*HN + (size_t)h*N + m];
          } else {
            #pragma omp parallel for schedule(static)
            for (int h=0; h<H; h++) for (int d=0;d<DV;d++) for (int m=0;m<N;m++)
              ((float*)dst->data)[(((size_t)b*N+m)*H+h)*DV+d] = (float)P->Oh16[(size_t)h*DV*N + (size_t)d*N + m];
          }
        } else {
        // (3) A·V: densify V weights, batched submit -> out_h[N,DV] f32
        _t=ork_now_us();
        for (int h=0; h<H; h++) { int hkv=h/rk2; ork_f16 *Vh=P->Vf+(size_t)h*nkvp*DV;
            for (int j=0;j<nkvp;j++) for (int e=0;e<DV;e++) Vh[(size_t)j*DV+e]=(j<nkv)?(ork_f16)rdf(v,e,j,hkv,b):(ork_f16)0.0f;
            if (ork_f16_mm_repack(ctx->npu,P->wav[h],nkvp,DV,Vh)) return false;
            P->tk[h]=(ork_mm_task_f16){P->wav[h],N,P->Pf+(size_t)h*N*nkvp,P->outf+(size_t)h*N*DV}; }
        if(prof) a_dv+=ork_now_us()-_t;
        _t=ork_now_us();
        if (dbg_cpu & 2) { for (int h=0;h<H;h++){ ork_f16 *Ph=P->Pf+(size_t)h*N*nkvp, *Vh=P->Vf+(size_t)h*nkvp*DV; float *o=P->outf+(size_t)h*N*DV;
            for (int m=0;m<N;m++) for (int e=0;e<DV;e++){ float a=0; for (int j=0;j<nkvp;j++) a+=(float)Ph[(size_t)m*nkvp+j]*(float)Vh[(size_t)j*DV+e]; o[(size_t)m*DV+e]=a; } } }
        else if (ork_attn_chain_mtiled(ctx->npu,H,P->tk,nkvp,DV)) return false;   // M-tiled: A·V K=nkvp caps M
        if(prof) a_av+=ork_now_us()-_t;
        // (4) scatter to dst [DV,H,N,B]: dst(dv,h,m,b) = out_h[m,dv]
        _t=ork_now_us();
        #pragma omp parallel for schedule(static)
        for (int h=0; h<H; h++) for (int m=0;m<N;m++) for (int e=0;e<DV;e++)
            ((float*)dst->data)[(((size_t)b*N+m)*H+h)*DV+e] = P->outf[(size_t)h*N*DV + (size_t)m*DV + e];
        }
        if(prof) a_sc+=ork_now_us()-_t;
    }
    if(prof){ a_calls++; fprintf(stderr,
        "[attnprof] calls=%ld  dqk=%.0f qk=%.0f | SM=%.0f [prep=%.0f quant=%.0f exp-NPU=%.0f red-NPU=%.0f norm=%.0f] | dv=%.0f av=%.0f sc=%.0f  total=%.0f ms | NPU(qk+av+exp+red)=%.0f CPU=%.0f\n",
        a_calls, a_dqk/1e3,a_qk/1e3, a_sm/1e3, a_smp/1e3,a_smq/1e3,a_sme/1e3,a_smr/1e3,a_smn/1e3, a_dv/1e3,a_av/1e3,a_sc/1e3,
        (a_dqk+a_qk+a_sm+a_dv+a_av+a_sc)/1e3, (a_qk+a_av+a_sme+a_smr)/1e3, (a_dqk+a_smp+a_smq+a_smn+a_dv+a_sc)/1e3); }
    return true;
}

static enum ggml_status ggml_backend_ork_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    if (g_segtime < 0) g_segtime = getenv("ORK_SEG_TIME") ? 1 : 0;
    if (getenv("ORK_STATIC_GRAPH") && cgraph->n_nodes >= 8) { static int _p = 0; if (_p++ < 2) ork_log_static_plan(cgraph); }
    ggml_backend_ork_context * ctx = (ggml_backend_ork_context *) backend->context;
    ctx->last_src1 = nullptr;
    if(getenv("ORK_VERBOSE"))fprintf(stderr, "[ORK] START graph_compute, %d nodes\n", cgraph->n_nodes); fflush(stderr);
    // MULTI-DOMAIN RESIDENCE: the FIRST decode graph (max matmul M==1) means the full prefill has run and
    // every weight is now resident; from here ANY further pack/load is per-token churn (the thing this
    // breakthrough eliminates). Clear load_phase + report the per-domain resident split, once.
    if (ctx->n_domains > 1 && ctx->load_phase) {
        int64_t max_m = 0;
        for (int i = 0; i < cgraph->n_nodes; i++) {
            const struct ggml_tensor * nd = cgraph->nodes[i];
            if ((nd->op == GGML_OP_MUL_MAT || nd->op == GGML_OP_MUL_MAT_ID) && nd->ne[1] > max_m) max_m = nd->ne[1];
        }
        if (max_m == 1) {
            ctx->load_phase = 0;
            size_t tot = 0; for (int d = 0; d < ctx->n_domains; d++) tot += ctx->domain_bytes[d];
            fprintf(stderr, "[ork RESIDENT] full model resident: %.2f GiB across %d domains:\n",
                    tot / (1024.0*1024.0*1024.0), ctx->n_domains);
            for (int d = 0; d < ctx->n_domains; d++)
                fprintf(stderr, "[ork RESIDENT]   domain %d: %.2f GiB\n", d, ctx->domain_bytes[d] / (1024.0*1024.0*1024.0));
        }
    }
    // EXPERIMENT #1 (ORK_MOE_PHASE_EVICT): detect the prefill->decode phase boundary and BULK-FREE the
    // resident dense-backbone wcache, reclaiming its IOVA (~2.8 GiB) for the MoE hot-expert cache. We
    // classify this graph by its max matmul M: M>1 = prefill, M==1 = decode. A clean bulk free at the
    // boundary (vs incremental LRU) avoids rk_iommu fragmentation; the next prefill repopulates wcache.
    if (ctx->phase_evict) {
        int64_t max_m = 0;
        for (int i = 0; i < cgraph->n_nodes; i++) {
            const struct ggml_tensor * nd = cgraph->nodes[i];
            if (nd->op == GGML_OP_MUL_MAT || nd->op == GGML_OP_MUL_MAT_ID)
                if (nd->ne[1] > max_m) max_m = nd->ne[1];
        }
        if (max_m > 0) {
            const int is_decode = (max_m == 1) ? 1 : 0;
            if (is_decode && ctx->last_graph_decode == 0 && !ctx->wcache.empty()) {
                size_t freed = ctx->wcache_bytes; size_t n = ctx->wcache.size();
                for (auto & kv : ctx->wcache) ork_mm_free(ctx->npu, kv.second.w);
                ctx->wcache.clear(); ctx->wcache_bytes = 0;
                ctx->backbone_evicts++; ctx->backbone_evict_bytes += freed;
                if (getenv("ORK_VERBOSE")) fprintf(stderr,
                    "[ORK PHASE] prefill->decode: bulk-freed backbone wcache (%zu weights, %.3f GiB) for experts\n",
                    n, freed / (1024.0*1024.0*1024.0));
            }
            ctx->last_graph_decode = is_decode;
        }
    }
    // QKV/gate-up group fusion: fuses independent same-input matmuls (q/k/v, gate/up) into ONE packed
    // submit — the crossing/submit reducer. VALIDATED bit-exact on qwen3-1.7B-Q8_0 (board .236, 0.6.73,
    // 2026-07-10): PREFILL pp64 56.8->66.4 t/s (+17%), submits 588->504 (-14%), PPL 4.2005 unchanged;
    // DECODE tg32 7.00->7.02 (neutral). It only pays off with enough work to amortize the wider
    // matmul+scatter: at M=1 decode that cost exceeds the saved submits (prior measurement: decode
    // 9.4->6.4), so we gate fusion to M>=32 (prefill) — decode stays unfused = bit-identical to baseline.
    // Default-ON for prefill; ORK_NO_FUSE disables entirely; ORK_FUSE forces fusion at ALL M (experiments).
    const bool fuse_force = getenv("ORK_FUSE") != nullptr;
    // orkd: group fusion is daemon-routed — it packs a CONCATENATED weight via ork_i8_mm_pack and runs it as
    // ONE ork_i8_mm_run (host A/C), so it works through the daemon AND amortizes the per-matmul submit floor
    // (q/k/v, gate/up -> 1 submit). Enabled under orkd.
    const int  fuse = (ctx->qbits == 8) && !getenv("ORK_NO_FUSE");
    // FUSED FFN chain (gate+SiLU, up, GLU, down). The fused per-tensor gate (fc.wg) is packed as an import
    // co-resident in its own layer's domain (ork_pack_pt_f32), so it works at ANY domain count — single
    // domain (<=4GiB) or many (>4GiB), gate/up/down naturally sharing their layer's domain. Just needs int8.
    // orkd: the FFN SwiGLU chain stays disabled — it uses fd-local run_i8_silu / run_f16_silu + the GU_CHAIN
    // DMA scratch (not daemon-routed). The generic MUL_MAT chain (below) IS routed and stays on.
    const bool ffn_chain = ork_ffn_chain_on() && ctx->qbits == 8 && !ctx->via_orkd;
    const bool ffn_dec = getenv("ORK_FFN_DEC") != nullptr && ctx->qbits == 8 && ctx->via_orkd;   // decode FFN via fused orkd chain
    if (getenv("ORK_VERBOSE")) { static int once = 0; if (!once++)
        fprintf(stderr, "[FFN-CHAIN gate] ffn_chain=%d (chain_on=%d qbits=%d n_domains=%d domain_layers=%d)\n",
                ffn_chain, ork_ffn_chain_on(), ctx->qbits, ctx->n_domains, ctx->domain_layers); }
    if (getenv("ORK_DUMP_GRAPH")) { static int dumped = 0;
        // ORK_LAYER_SPINE recon: dump the FULL assigned span (op + this-node name + src0/src1 names) so the
        // whole-layer matcher can be written against the real runtime graph and its fragmentation seen.
        int cap = getenv("ORK_DUMP_ALL") ? 200 : 40;
        bool has_ffn = getenv("ORK_DUMP_ALL") ? true : false, has_glu = false;
        for (int i = 0; i < cgraph->n_nodes; i++) { struct ggml_tensor * n = cgraph->nodes[i];
            if (n->op == GGML_OP_GLU) has_glu = true;
            if (n->src[0] && (strstr(n->src[0]->name,"ffn_gate")||strstr(n->src[0]->name,"ffn_down"))) has_ffn = true; }
        if ((has_ffn || has_glu) && dumped++ < 3) {
            fprintf(stderr, "[ORK GRAPH] span %d nodes (has_ffn=%d has_glu=%d):\n", cgraph->n_nodes, has_ffn, has_glu);
            for (int i = 0; i < cgraph->n_nodes && i < cap; i++) { struct ggml_tensor * n = cgraph->nodes[i];
                fprintf(stderr, "  [%2d] %-12s %-18s src0=%-28s src1=%-20s ne=[%ld,%ld]\n", i, ggml_op_name(n->op), n->name[0]?n->name:"-",
                        n->src[0]?n->src[0]->name:"-", n->src[1]?n->src[1]->name:"-", (long)n->ne[0], (long)n->ne[1]);
            } fflush(stderr); } }
    // Step 1 (ORK_SCAN_AHEAD): gather INDEPENDENT same-input matmuls that graph order interleaves with movable
    // ops (q/k/v, split by RoPE/norm) into one HW-chain group, computing them early and marking them done so
    // the loop skips them when reached. Safe because a grouped matmul reads only its weight leaf + the shared
    // src[1] (already computed) -> independent of everything between.
    static const int scan_ahead = getenv("ORK_SCAN_AHEAD") != nullptr;
    std::unordered_set<const struct ggml_tensor *> sa_done;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if (scan_ahead && !sa_done.empty() && sa_done.count(node)) continue;   // computed early by a scan-ahead group
        // ORK_FFN_DEC: DECODE (M==1) SwiGLU inner via the fused orkd chain (one submit against the resident
        // weights). Fires at the gate node; the matcher takes up from glu->src[1] and consumes glu/down.
        if (ffn_dec && ctx->via_orkd && node->op == GGML_OP_MUL_MAT && node->src[0] &&
            strstr(node->src[0]->name, "ffn_gate") && node->ne[1] == 1) {
            struct ggml_tensor *g,*u,*gl,*dn; int last;
            if (ork_ffn_chain_match(cgraph, i, &g, &u, &gl, &dn, &last) &&
                ggml_backend_ork_ffn_decode_orkd(ctx, g, u, gl, dn)) { i = last; continue; }
        }
        // ORK_FFN_CHAIN: recognize the fused-SwiGLU FFN inner and run it as ONE round-trip-free int8 chain
        // (int8 intermediates never touch fp32). Consumes all 4 nodes; falls through to per-node on any miss.
        if (ffn_chain && node->op == GGML_OP_MUL_MAT) {
            struct ggml_tensor *g,*u,*gl,*dn; int last;
            int matched = ork_ffn_chain_match(cgraph, i, &g, &u, &gl, &dn, &last);
            if (getenv("ORK_VERBOSE") && node->src[0] && strstr(node->src[0]->name, "ffn_gate"))
                fprintf(stderr, "[FFN-CHAIN match] node %d %s -> matched=%d\n", i, node->src[0]->name, matched);
            if (matched) {
                double _tf = ork_seg_t0();
                bool _fok = ggml_backend_ork_ffn_swiglu_chain(ctx, g, u, gl, dn);
                ork_seg_add(SEG_FFN, _tf);
                if (_fok) {
                    i = last;  // fused OK: skip past gate/up/GLU/down — all handled by the chain
                    continue;
                }
                // Chain failed (alloc/shape/calibration guard). The handler writes dst only at the very end
                // and keeps intermediates in local buffers, so on failure the gate/up/GLU/down tensors are
                // untouched — FALL THROUGH to per-node processing (gate here, up/GLU/down as later nodes)
                // instead of aborting the graph. This never breaks the model AND avoids the leak-on-FAILED
                // (returning GGML_STATUS_FAILED stranded already-allocated buffers and degraded the IOVA).
                if (getenv("ORK_VERBOSE")) fprintf(stderr, "[FFN-CHAIN] fuse failed at node %d -> per-node fallback\n", i);
            }
        }
        switch (node->op) {
            case GGML_OP_MUL_MAT: {
                // ORK_ATTN: batched/dynamic matmul (attention QKᵀ·V, GDN chunked) -> ork_bmm_fp16.
                // PREFILL-ONLY (M = dst->ne[1] > 1): the M=1 DECODE A·V wedges in-graph after prefill (a
                // post-prefill IOVA/state issue — the fp16 bmm is bit-correct at M=1 in isolation, confirmed
                // by tools/bmm_probe; only the in-graph decode call faults). Decode attention stays on CPU
                // (it's the CPU path anyway — see decode-is-cpu-path). The parity target is prefill (M>1),
                // where the batched attention matmuls run clean on the NPU.
                static const int ork_attn = getenv("ORK_ATTN") != nullptr;
                if (ork_attn && (node->ne[2] > 1 || node->ne[3] > 1) && node->ne[1] > 1) {
                    double _t = ork_seg_t0();
                    if (!ggml_backend_ork_bmm_fp16(ctx, node)) return GGML_STATUS_FAILED;
                    ork_seg_add((node->src[0]->name && strstr(node->src[0]->name, "cache_v")) ? SEG_AV : SEG_QKT, _t);
                    break;
                }
                std::vector<struct ggml_tensor *> chain_nodes;
                ork_chain_type type = get_node_chain_type(ctx, node);

                // STREAM-POOL mode handles each MUL_MAT individually (the per-node int8 path is
                // stream-pool-aware; the multi-weight chain handler is not — it needs distinct weights
                // co-resident under allow_evict=false, which the RAM/IOVA tiering can't guarantee).
                // (orkd: the MUL_MAT chain is routed — ork_resolve_weight_i8 + ork_dispatch_i8 dispatch it
                // through the daemon's run_chain_i8 / run_i8, so it stays enabled.)
                if (ctx->spool) type = ORK_CHAIN_NONE;

                if (type != ORK_CHAIN_NONE && node->ne[2] == 1 && node->ne[3] == 1) {
                    chain_nodes.push_back(node);
                    while (i + chain_nodes.size() < cgraph->n_nodes && chain_nodes.size() < 32) {
                        struct ggml_tensor * next_node = cgraph->nodes[i + chain_nodes.size()];
                        if (get_node_chain_type(ctx, next_node) != type) {
                            break;
                        }
                        if (next_node->ne[2] != 1 || next_node->ne[3] != 1) {
                            break;
                        }
                        bool depends = false;
                        for (struct ggml_tensor * prev : chain_nodes) {
                            if (next_node->src[0] == prev || next_node->src[1] == prev) {
                                depends = true;
                                break;
                            }
                        }
                        if (depends) {
                            break;
                        }
                        chain_nodes.push_back(next_node);
                    }
                }

                double _tg = ork_seg_t0();
                if (chain_nodes.size() >= 2) {
                    bool chain_ok = false;
                    if (type == ORK_CHAIN_I8) {
                        chain_ok = ggml_backend_ork_mul_mat_chain_i8(ctx, chain_nodes.data(), chain_nodes.size());
                    } else if (type == ORK_CHAIN_I4) {
                        chain_ok = ggml_backend_ork_mul_mat_chain_i4(ctx, chain_nodes.data(), chain_nodes.size());
                    }
                    if (!chain_ok) return GGML_STATUS_FAILED;
                    i += chain_nodes.size() - 1;
                } else {
                    struct ggml_tensor * grp[16]; int ng = 1; grp[0] = node;
                    // Group-fuse consecutive independent same-input MUL_MATs (q/k/v, gate/up) into ONE
                    // packed-weight submit — fewer round-trips. Members must share ONE compute tier: an
                    // all-W8A8 group -> mul_mat_group_i8, an all-W4A4 group -> mul_mat_group_i4 (the W4A4 twin,
                    // which also shares the FWHT rotation once). Default mixed path (q4->W8A8) => all-i8 groups;
                    // ORK_MIXED_W4A4 => i4 groups for the 4-bit tier. Mixed-tier runs stay per-node.
                    const int grp_tier = ork_node_qbits(ctx, node);   // group members must all share this tier (8=W8A8, 4=W4A4)
                    // node->ne[1] = M (output rows). M-sweep (qwen3-1.7B-Q8_0, 2026-07-10, warm -r3): fusion WINS
                    // at EVERY M>=2 — pp2 +10.8%, pp4 +10.5%, pp8 +8-9%, pp64 +12-17%. At M=1: pp1 is +1% but
                    // autoregressive tg-decode REGRESSES (prior: 9.4->6.4), so the floor is M>=2 (decode M=1 stays
                    // unfused). Covers all prefill + DFlash's M=block batched-verify. ORK_FUSE_MINM overrides.
                    static const int fuse_minm = getenv("ORK_FUSE_MINM") ? atoi(getenv("ORK_FUSE_MINM")) : 2;
                    if (fuse && node->ne[2] == 1 && node->ne[3] == 1 && (fuse_force || node->ne[1] >= fuse_minm)) {
                        if (scan_ahead) {
                            // Step 1: scan the WHOLE remaining subgraph for INDEPENDENT same-input matmuls,
                            // skipping past the movable ops (RoPE/reshape/norm) that interleave q/k/v in graph
                            // order. src[1]==node->src[1] guarantees independence (reads only the shared input +
                            // a weight leaf), so computing them early is safe. Grouped-ahead nodes -> sa_done.
                            for (int j = i + 1; j < cgraph->n_nodes && ng < 16; j++) {
                                struct ggml_tensor * nj = cgraph->nodes[j];
                                if (sa_done.count(nj)) continue;
                                if (nj->op == GGML_OP_MUL_MAT && nj->src[1] == node->src[1] &&
                                    nj->src[0]->ne[0] == node->src[0]->ne[0] && nj->ne[2] == 1 && nj->ne[3] == 1 &&
                                    ork_node_qbits(ctx, nj) == grp_tier)
                                    grp[ng++] = nj;   // else: skip (keep scanning past movable ops)
                            }
                        } else {
                            while (i + ng < cgraph->n_nodes && ng < 16) {
                                struct ggml_tensor * nj = cgraph->nodes[i + ng];
                                if (nj->op == GGML_OP_MUL_MAT && nj->src[1] == node->src[1] &&
                                    nj->src[0]->ne[0] == node->src[0]->ne[0] && nj->ne[2] == 1 && nj->ne[3] == 1 &&
                                    ork_node_qbits(ctx, nj) == grp_tier)
                                    grp[ng++] = nj;
                                else break;
                            }
                        }
                    }
                    if (ng >= 2) {
                        bool grp_ok = (grp_tier == 4) ? ggml_backend_ork_mul_mat_group_i4(ctx, grp, ng)
                                                      : ggml_backend_ork_mul_mat_group_i8(ctx, grp, ng);
                        if (!grp_ok) return GGML_STATUS_FAILED;
                        if (scan_ahead) { for (int k = 1; k < ng; k++) sa_done.insert(grp[k]); }   // skip when reached
                        else i += ng - 1;   // consecutive group -> advance past it
                    } else {
                        const char * name = node->src[0]->name;
                        bool is_ffn = strstr(name, "ffn_") || ork_is_expert(name);
                        bool is_attn = strstr(name, "attn_q") || strstr(name, "attn_k") || strstr(name, "attn_v") || strstr(name, "attn_output");
                        
                        int target_qbits = ctx->qbits;
                        if (ork_mixed_dispatch_on()) {
                            // per-tensor dispatch driven by the GGUF's own quant precision. The 4-bit tier's
                            // COMPUTE path: native W4A4 (mul_mat_i4) is structurally single-row on this NPU
                            // (no M-tile weight-DMA amortization → re-streams each weight ~M times → loses to
                            // int8 at prefill), so by DEFAULT the 4-bit tier computes W8A8 (dequant q4→int8,
                            // amortized + coherent ≈ Q8_0 speed); the compact-int4 win is realized in the
                            // .orkpack STORAGE (i4a8), not the compute. ORK_MIXED_W4A4=1 opts the 4-bit tier
                            // into native W4A4 for the decode/streaming regimes where it actually wins.
                            static const int w4a4 = env_enabled("ORK_MIXED_W4A4");
                            double b = ork_src_type_bits(node->src[0]->type);
                            target_qbits = (w4a4 && b >= 0.0 && b < 5.0) ? 4 : 8;
                            if (getenv("ORK_VERBOSE"))
                                fprintf(stderr, "[ORK MIXED] %s src=%s bits=%.2f -> W%dA%d\n",
                                        name, ggml_type_name(node->src[0]->type), b, target_qbits, target_qbits);
                        } else if (ctx->hybrid) {
                            if (is_ffn) target_qbits = 4;
                            else if (is_attn) target_qbits = 8;
                        }

                        // int4 COMPUTE routing: native W4A4 is fragile at prefill (grouped M_padded wedges) and stays
                        // opt-in via ORK_MIXED_W4A4 (ctx->hadamard). When selected it is ALWAYS the rotated variant —
                        // the un-rotated grouped route is gone, because a symmetric 4-bit MAC without R is incoherent
                        // (see ork_w4a4_native_on). The DEFAULT int4 (incl. ORK_QUANT=4) computes W8A8 via mul_mat_i8
                        // (int4 weights inflated int4->int8 on the NPU — robust, no wedge); compact i4a8 STORAGE is
                        // chosen separately at persist (ork_orkpack_tier).
                        bool native_w4a4 = (target_qbits == 4) && ctx->hadamard;
                        bool mm_ok = native_w4a4 ? ggml_backend_ork_mul_mat_i4_hadamard(ctx, node)
                                                 : ggml_backend_ork_mul_mat_i8(ctx, node);
                        if (!mm_ok) return GGML_STATUS_FAILED;
                    }
                }
                { const char * nm = node->src[0] ? node->src[0]->name : "";
                  int cat = (strstr(nm,"attn_q")||strstr(nm,"attn_k")||strstr(nm,"attn_v")) ? SEG_QKV
                          : strstr(nm,"attn_output") ? SEG_O
                          : strstr(nm,"ffn_") ? SEG_FFN : SEG_OTH;
                  ork_seg_add(cat, _tg); }
                break;
            }
            case GGML_OP_MUL_MAT_ID: {
                if (!ggml_backend_ork_mul_mat_id_i8(ctx, node)) return GGML_STATUS_FAILED;
                break;
            }
            case GGML_OP_MUL: {   // ORK_OPS_NPU -> NPU ewmul (broadcast-aware); ORK_CLAIM_OPS -> CPU-delegate; else NPU ewmul
                if (getenv("ORK_OPS_NPU") && ggml_backend_ork_binop_npu(ctx, node, false)) break;
                if (getenv("ORK_CLAIM_OPS") ? !ork_cpu_delegate_node(ctx, node) : !ggml_backend_ork_mul_f16(ctx, node)) return GGML_STATUS_FAILED;
                break;
            }
            case GGML_OP_ADD: {   // residual add: fp16 accumulates error over layers -> OPT-IN only (ORK_OPS_NPU_ADD);
                                  // default keeps it fp32 (delegate). ORK_CLAIM_OPS -> CPU-delegate; else NPU add.
                if (getenv("ORK_OPS_NPU_ADD") && ggml_backend_ork_binop_npu(ctx, node, true)) break;
                if (getenv("ORK_CLAIM_OPS") ? !ork_cpu_delegate_node(ctx, node) : !ggml_backend_ork_add_f16(ctx, node)) return GGML_STATUS_FAILED;
                break;
            }
            case GGML_OP_UNARY: {
                const bool gelu = ggml_get_unary_op(node) == GGML_UNARY_OP_GELU;
                if (!ggml_backend_ork_unary_i8(ctx, node, gelu)) return GGML_STATUS_FAILED;
                break;
            }
            case GGML_OP_GLU: {
                if (!ggml_backend_ork_glu(ctx, node)) return GGML_STATUS_FAILED;
                break;
            }
            case GGML_OP_SOFT_MAX: {   // 2b: attention softmax — exp on the NPU (supports_op gates the claimed cases)
                double _t = ork_seg_t0();
                if (!ggml_backend_ork_soft_max(ctx, node)) return GGML_STATUS_FAILED;
                ork_seg_add(SEG_SM, _t);
                break;
            }
            case GGML_OP_RMS_NORM:
            case GGML_OP_ROPE: {   // #1: claimed boundary op. ORK_OPS_NPU -> run RoPE on the NPU
                                   // (ork_f16_npu_rope_neox); else CPU-delegate (keeps the layer one ork subgraph).
                bool done = false;
                if (getenv("ORK_OPS_NPU")) done = (node->op == GGML_OP_ROPE) ? ggml_backend_ork_rope_npu(ctx, node)
                                                                              : ggml_backend_ork_rmsnorm_npu(ctx, node);
                if (!done && !ork_cpu_delegate_node(ctx, node)) return GGML_STATUS_FAILED;
                break;
            }
            case GGML_OP_SSM_SCAN: {
                if (!ggml_backend_ork_ssm_scan(ctx, node) && !ork_cpu_delegate_node(ctx, node)) return GGML_STATUS_FAILED;
                break;
            }
            case GGML_OP_FLASH_ATTN_EXT: {
                if (!ggml_backend_ork_flash_attn_ext(ctx, node) && !ork_cpu_delegate_node(ctx, node)) return GGML_STATUS_FAILED;
                break;
            }
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;
            default:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }
    if (g_segtime > 0) { static long _c = 0; if (++_c % 40 == 0) {
        double tot = 0; for (int k = 0; k < SEG_N; k++) tot += g_seg_us[k];
        fprintf(stderr, "[ORK SEG-TIME] cumulative NPU-op wall over %ld graphs (warmup+prefill dominate):\n", _c);
        for (int k = 0; k < SEG_N; k++) if (g_seg_n[k])
            fprintf(stderr, "  %-24s %8.1f ms (%5.1f%%) n=%ld\n", g_seg_name[k], g_seg_us[k]/1000.0, 100.0*g_seg_us[k]/(tot+1e-9), g_seg_n[k]);
    } }
    return GGML_STATUS_SUCCESS;
    GGML_UNUSED(backend);
}

// ---- ASYNC CROSS-STREAM SUBMIT PATH (opt-in; nothing calls these unless a consumer/harness does) ----
// The RKNPU is single-stream (one hardware command queue), so two streams' NPU submits must serialize — but a
// stream's CPU work overlaps another stream's NPU work for FREE (measured hidden~100%, ork-driver overlap_prof).
// These let a driver run one context's NPU graph on a worker thread while the launching thread does other work
// (e.g. a speculative draft's routing/sampling), then join. g_npu_queue_mu serializes the NPU across contexts;
// the plain (synchronous) graph_compute does NOT take it, so the default make-test path is unchanged.
static std::mutex g_npu_queue_mu;

extern "C" enum ggml_status ggml_backend_ork_synchronize(ggml_backend_t backend) {
    ggml_backend_ork_context * ctx = (ggml_backend_ork_context *) backend->context;
    if (ctx->async_inflight) {
        if (ctx->async_thr.joinable()) ctx->async_thr.join();          // join = full memory barrier
        ctx->async_inflight = false;
    }
    return ctx->async_status;
}

extern "C" void ggml_backend_ork_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_ork_context * ctx = (ggml_backend_ork_context *) backend->context;
    if (ctx->async_inflight) ggml_backend_ork_synchronize(backend);    // one in-flight job per backend
    ctx->async_inflight = true;
    ctx->async_status   = GGML_STATUS_SUCCESS;
    ctx->async_thr = std::thread([backend, cgraph, ctx]() {
        // The NPU worker (matmul dispatch + quant/dequant) belongs on the big cores; without this it would
        // INHERIT the launcher's affinity, so a launcher pinned elsewhere would drag the whole matmul onto
        // slow cores. ORK_ASYNC_BIG pins it to the A76 cluster (RK3588: cores 4-7) so a concurrent CPU stream
        // pinned to the little cores runs on disjoint cores (the core-partitioning path to free overlap).
        if (getenv("ORK_ASYNC_BIG")) {
            cpu_set_t s; CPU_ZERO(&s); for (int c = 4; c < 8; c++) CPU_SET(c, &s);
            pthread_setaffinity_np(pthread_self(), sizeof(s), &s);
        }
        std::lock_guard<std::mutex> lk(g_npu_queue_mu);                // single hardware queue: serialize NPU
        ctx->async_status = ggml_backend_ork_graph_compute(backend, cgraph);
    });
}

static struct ggml_backend_i ork_backend_i = {
    /* .get_name                = */ ggml_backend_ork_get_name,
    /* .free                    = */ ggml_backend_ork_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_ork_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_ork_guid(void) {
    static ggml_guid guid = { 0x0a,0xc5,0x11,0x3d,0x6e,0x42,0x7b,0x90,0xa1,0xff,0x52,0x88,0x14,0x33,0x9c,0x01 };
    return &guid;
}

ggml_backend_t ggml_backend_ork_init(void) {
    // PRODUCT CONFIG (2 load-time options -> internal validated defaults; see ggml-ork.h). Translated to the
    // internal mechanisms ONCE here, before any of them is read below. The user sets only the 2 options via
    // ggml_backend_ork_set_load_config(); everything else is hardcoded to the validated values (not a knob).
    // If the config was never set, fall through to legacy env behavior (dev/debug), unchanged.
    if (g_ork_cfg_set) {
        setenv("ORK_FFN_CHAIN", "1", 1);        // round-trip-free on-NPU SwiGLU chain (both paths use it)
        setenv("ORK_MIXED_NOTHRASH", "1", 1);   // no int8<->int4/mode re-warm thrash
        setenv("ORK_NO_BF", "1", 1);            // compact footprint -> fits one IOMMU domain (dodges #36 dom>0)
        if (g_ork_cfg_silu_int8fused) {         // int8 FULLY FUSED through-and-through: SiLU fused into the gate
            setenv("ORK_FFN_FUSED_SILU", "1", 1);                                        // matmul, all-NPU (fast, ~PPL 22)
            unsetenv("ORK_FFN_SILU_I16"); unsetenv("ORK_FFN_SILU_CPU_GMAX");
        } else {                                // DEFAULT: int16 coherent — SiLU is the UNFUSED standalone NPU op (~PPL 19)
            setenv("ORK_FFN_SILU_I16", "1", 1);
            unsetenv("ORK_FFN_FUSED_SILU"); unsetenv("ORK_FFN_SILU_CPU_GMAX");
        }
        if (g_ork_cfg_dflash) setenv("ORK_DFLASH", "1", 1); else unsetenv("ORK_DFLASH");   // drafter gate (read by the dflash path)
        GGML_LOG_INFO("%s: load config: silu=%s, dflash=%s\n", __func__,
                      g_ork_cfg_silu_int8fused ? "int8-fused(through)" : "int16-coherent", g_ork_cfg_dflash ? "on" : "off");
    }
    // DIRECT (in-process lib) is the DEFAULT NPU path — it is what oRKLLM and any single-process run get with
    // no extra env. The lib and orkd paths are behaviorally IDENTICAL (same op set + shapes, proven by
    // test_api_parity and the direct-vs-orkd A/B), differing only in transport. orkd (the daemon that owns the
    // single-stream NPU and serializes every submit, so concurrent processes can't wedge the IOMMU — a dev-time
    // safety property, not a separate architecture) is OPT-IN via the single env ORK_USE_ORKD=1.
    // Transport is selected by CHOOSING the ork-driver entry point (not by env magic): ork_npu_init() = direct
    // (the DEFAULT), ork_npu_init_orkd() = orkd client. orkd is OPT-IN via ORK_USE_ORKD=1; unset (the common
    // case, incl. oRKLLM) = direct in-process NPU. (There is no ORK_DIRECT any more — direct being the default
    // makes an opt-in-to-direct switch redundant.)
    const char * uo = getenv("ORK_USE_ORKD");
    const bool want_orkd = uo && atoi(uo) != 0;
    ork_npu * npu;
    if (want_orkd) {
        setenv("ORK_ORKD_RING", "1", 1);   // the low-latency decode transport async ork_mm_submit/collect ride
        npu = ork_npu_init_orkd();         // explicit orkd client; NULL (no silent fallback) if the daemon can't connect
        if (!npu) {
            GGML_LOG_ERROR("%s: ORK_USE_ORKD=1 but orkd did not connect — check ORKD_BIN points at the orkd binary "
                           "and it is spawnable, or unset ORK_USE_ORKD to use the default in-process lib path "
                           "(single-stream: do not run concurrent NPU processes).\n", __func__);
            return NULL;
        }
    } else {
        unsetenv("ORK_USE_ORKD");   // neutralize the driver's legacy env override so ork_npu_init() stays direct
        // In-process direct mode shares the 4 big cores with llama.cpp's compute threadpool (-t 4). Pinning the
        // NPU-driver worker threads to those same cores (the default) OVERSUBSCRIBES them against the threadpool
        // and drags prefill (measured: 136 -> 213 t/s at M=228, i.e. faster than orkd's 168 once unpinned). orkd
        // doesn't hit this — its NPU threads live in a separate process. So default direct mode to no-affinity;
        // overwrite=0 leaves an explicit user ORK_NO_AFFINITY untouched.
        setenv("ORK_NO_AFFINITY", "1", 0);
        npu = ork_npu_init();       // DEFAULT: direct in-process NPU
        if (!npu) { GGML_LOG_ERROR("%s: ork_npu_init failed (no NPU / no perms)\n", __func__); return NULL; }
        GGML_LOG_INFO("%s: in-process lib NPU path (direct — default; single-stream, no concurrent NPU procs)\n", __func__);
    }
    ggml_backend_ork_context * ctx = new ggml_backend_ork_context;
    ctx->npu = npu;
    ctx->via_orkd = ork_npu_uses_orkd(npu) != 0;   // true iff ORK_USE_ORKD opted in and the daemon connected; false in the default direct path — gates the orkd-routed weight/matmul path
    g_ork_ctx = ctx;
    ork_persist_init(ctx);   // .orkpack: read (fast load) if present, else build it this run
    // (b) load the persisted gmax sidecar (<orkpack>.gmax) if present: a known model's per-layer gate
    // ranges available at LOAD (before prep recomputes them) — foundation for a load-time selective policy.
    // Uses the RESOLVED pack path (ctx->persist_final, set by ork_persist_init: override > legacy > derived).
    { const char * pp = ctx->persist_final.empty() ? nullptr : ctx->persist_final.c_str();
      if (pp && pp[0]) { std::string sp = std::string(pp) + ".gmax"; FILE * gf = fopen(sp.c_str(), "r");
          if (gf) { char nm[256]; float gv;
              while (fscanf(gf, "%255s %f", nm, &gv) == 2) ctx->gmax_loaded[nm] = gv;
              fclose(gf);
              if (getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK FFN-GMAX] loaded %zu-layer gmax profile from %s\n", ctx->gmax_loaded.size(), sp.c_str()); } } }
    // Weight tier. The .orkpack is the normal source of truth: ork_persist_init (above) decoded the tier the
    // pack was BUILT at into persist_qbits, so loading an int4 pack selects the int4 path with nothing set.
    // ORK_QUANT stays as a DEVELOPMENT OVERRIDE — it forces the tier for a run with no pack yet (the build
    // pass that CREATES an int4 pack) or to deliberately rebuild at a different tier. With no pack and no
    // override the default is W8A8.
    const char * q = getenv("ORK_QUANT");
    ctx->qbits = (q && *q)         ? ((q[0] == '4') ? 4 : 8)
               : ctx->persist_qbits ? ctx->persist_qbits
                                    : 8;
    ctx->profile = getenv("ORK_PROFILE") != nullptr;
    if (ctx->profile) atexit(ork_profile_atexit);   // LEVER3: dump under llama-bench (no backend free)
    ctx->no_reuse = getenv("ORK_NOREUSE") != nullptr;
    ctx->no_cache = getenv("ORK_NOCACHE") != nullptr;
    ctx->slice_route = getenv("ORK_SLICE_ROUTE") != nullptr && !ctx->via_orkd;   // wide int8 via sliced doorbell (fd-local only)
    ctx->hybrid = g_ork_hybrid_loading || getenv("ORK_HYBRID") != nullptr;
    // Rotation is now exactly "native W4A4 is selected" — one flag, no separate opt-in. Selecting the 4-bit
    // datapath routes to mul_mat_i4_hadamard (per-channel, single submit, persist-able); the un-rotated
    // grouped route is no longer reachable. ORK_QUANT=4 alone does NOT land here — that is the int4 STORAGE
    // tier computing W8A8. See ork_w4a4_native_on for the full rule and why ork-driver cannot own it.
    ctx->hadamard = ork_w4a4_native_on();
    ctx->phase_evict = env_enabled("ORK_MOE_PHASE_EVICT");   // #1 phase-aware backbone eviction (default OFF)
    // SHIP HARDENING (2026-07-11): the FFN chain ships with a COMPACT resident footprint — default ORK_NO_BF
    // when ORK_FFN_CHAIN is on. The full-K Bf decode-copies ~double the footprint (~1.9->3.6 GiB on 1.7B),
    // which pushes the auto-domain sizer to n_domains>1, where the imported dom>0 submits INTERMITTENTLY
    // WEDGE (#36: errno=110 on dom=1 weights). Decode uses the per-node path (not the chain), so shedding
    // Bf costs the chain nothing. Set BEFORE the domain sizing below (which now honors NO_BF). Opt out with
    // ORK_KEEP_BF=1; an explicit ORK_NO_BF (either value) also wins.
    if (ork_ffn_chain_on() && !getenv("ORK_KEEP_BF") && !getenv("ORK_NO_BF")) setenv("ORK_NO_BF", "1", 1);
    // MULTI-DOMAIN RESIDENCE: weights spread across as many IOMMU domains as the AUTO-SIZER computes (each
    // with its own ~4 GiB IOVA window) so a >4 GiB model stays fully resident with NO streaming/churn. The
    // domain count is auto-only (no ORK_DOMAINS env, no fixed cap); ORK_DOMAIN_LAYERS is a manual layout knob.
    // Pass a large ORK_WCACHE_BUDGET_MB so the residence never evicts.
    {
      // The auto-sizer below is the SOLE authority for the IOMMU domain count — computed from the resident
      // orkpack footprint. The old ORK_DOMAINS env override is gone (it was only ever a clamp-UP escape hatch
      // for an auto UNDER-count; the under-count root cause — unbudgeted fused-group weights + missing orkd
      // domain ownership — is fixed, so the auto count is trustworthy). There is NO fixed domain cap: the
      // driver arrays grow dynamically (ork_npu_set_ndomains) and the daemon hands out up to its 64-bit
      // owned_dom width. ORK_DOMAIN_LAYERS remains a manual layer->domain LAYOUT knob (default 0 = auto).
      if (!ctx->persist_idx.empty()) {
          // AUTO from .orkpack footprint: sum the int8 blob bytes, inflate for the resident Bb+Bf footprint,
          // and size to the 3.0 GiB per-domain fill cap (matches ork_weight_domain). A model that fits one
          // 4 GiB domain -> n_domains=1 (the single-domain FAST PATH — and the regime the fused FFN chain
          // requires: all of a layer's weights co-resident so its submits never cross a domain). A >4 GiB
          // model -> exactly the domains it needs. INFLATION = 2.1x: the resident set is Bb PLUS a full-K Bf
          // rebuild for every K<=4096 weight (qkv/gate/up/o = most of the model), so the real footprint is
          // ~2.0x the int8 blob (MEASURED: 7B = 12.26 GiB resident / 6.08 GiB blob = 2.02x), NOT 1.7x. The old
          // 1.7x UNDER-sized (7B -> 4 domains, but 4 x 3.0 = 12.0 GiB < 12.26 -> the last domain, which
          // ork_domain_for() leaves UNCAPPED, overflowed to 3.46 GiB -> a late weight's Bf import PRIME-failed
          // -> partial residence -> warmup NPU soft-reset/wedge). 2.1x (> measured 2.02x) sizes 7B -> 5 domains
          // so no domain overflows and each keeps IOVA headroom for imports + scratch. Explicit ORK_DOMAINS wins.
          // Size domains from the RESIDENT footprint — derived from each weight's K*N and THIS quant path's
          // resident tile, NOT from the compact blob_size (which under-sizes: an int4 blob is ~half the bytes
          // of the int8 tile it expands to, so the old blob*2.1 gave the 35B Q4_K pack n_domains=1 -> the
          // single domain overflowed at ~2.7 GiB and every import PRIME-failed -> "loaded 0 from disk").
          // The resident tile is quant-path-specific (each path packs its own IOVA form), so branch on it:
          //   W8A8 (qbits==8): int8 tile K*N  (also the resident form for Q4_K/q4_0 sources — the i4a8 storage
          //                    path INFLATES their nibbles to int8; that's why we size from K*N, not the blob).
          //   W4A4 (qbits==4): native int4 tile K*N/2 (nibbles resident, DT_I4).
          // Plus a full-K Bf rebuild (another tile for K<=4096, ~all weights). ORK_FFN_CHAIN: the fused
          // per-tensor gate (fc.wg, ork_pack_pt_f32) COEXISTS with the per-channel gate — ork_ffn_prep
          // deliberately does NOT evict it (freeing mid-eval fragments the 32-bit IOVA -> import ENOMEM ->
          // wedge). So fc.wg ADDS a K*Nff int8 tile + its full-K Bf per ffn_gate; it is packed FRESH at
          // runtime and is NOT in persist_idx, so the loop below must add it explicitly or the byte-balanced
          // domains have no room -> the runtime fc.wg pack overflows -> OOM (load) / re-pack churn (JIT). (The
          // old "fc.wg REPLACES the gate 1:1, adds no volume" belief was stale — the code keeps both.)
          // MEASURED (int8): 7B non-chain = 12.26 GiB resident.
          // ORK_FFN_F16: the FFN gate/up/down may run the fp16 route, whose buffers (resident fp16 tile =
          // 2*K*N, no Bf; or the JIT shared scratch + per-mode-switch Cc realloc) are NOT part of the int8
          // footprint below. Left unaccounted, they overflow near-full domains and a mid-forward alloc
          // failure WEDGES the mixed fp16 path (blocker b, root-caused via tools/mode_switch_probe). Add the
          // fp16 delta per FFN weight so auto-sizing leaves IOVA HEADROOM. gmax subset is unknown at init ->
          // assume all FFN layers (worst case; JIT over-counts but that's safe slack).
          const bool f16route = getenv("ORK_FFN_F16") != nullptr;
          const bool chain    = ork_ffn_chain_on() && ctx->qbits == 8;   // packs a coexisting per-tensor fc.wg per ffn_gate
          const bool no_bf = getenv("ORK_NO_BF") != nullptr;   // NO_BF sheds the full-K Bf rebuild -> don't size for it
          // RAM budget first — the adaptive Bf decision below is a footprint-vs-budget comparison.
          long pg = sysconf(_SC_PHYS_PAGES), ps = sysconf(_SC_PAGE_SIZE);
          size_t phys = (pg > 0 && ps > 0) ? (size_t) pg * (size_t) ps : (size_t) 31 * 1024*1024*1024;
          size_t budget = (size_t) (0.72 * (double) phys);   // ~72% for resident weights; rest = activations/KV/page-cache
          if (const char * rb = getenv("ORK_RESIDENCE_RAM_MB")) budget = (size_t) atoll(rb) * 1024*1024;
          // Split the footprint into base (Bb + fp16/chain extras, ALWAYS resident) and bf_extra (the decode-only
          // full-K Bf companion, ~doubles int4 bytes). Bf is optional; base is not.
          size_t base = 0, bf_extra = 0;
          for (const auto & kv : ctx->persist_idx) {
              const int K = (int) kv.second.K, N = (int) kv.second.N;
              size_t tile = (ctx->qbits == 4) ? ((size_t) K * N / 2)   // W4A4: native int4 nibble tile
                                              : ((size_t) K * N);      // W8A8: int8 tile (incl. inflated q4)
              base += tile;
              if (K <= 4096) bf_extra += tile;                         // full-K Bf rebuild (decode fast path only)
              if (f16route && (kv.first.find("ffn_gate") != std::string::npos ||
                               kv.first.find("ffn_up")   != std::string::npos ||
                               kv.first.find("ffn_down") != std::string::npos))
                  base += (size_t) 2 * K * N;                          // fp16 tile headroom (no Bf)
              // ORK_FFN_CHAIN: the fused per-tensor gate fc.wg (packed fresh at runtime, NOT in persist_idx)
              // coexists with the per-channel gate — size for its tile (+Bf) per ffn_gate so a runtime pack
              // never overflows the domains (the multi-domain 7B chain bug).
              if (chain && kv.first.find("ffn_gate") != std::string::npos) {
                  base += tile; if (K <= 4096) bf_extra += tile;
              }
          }
          // ADAPTIVE Bf (no env gate — behaviour follows the scenario): build Bf iff the WITH-Bf footprint fits
          // the RAM budget; auto-shed it when it wouldn't (e.g. a ~15 GiB int4 MoE whose Bf would push it to
          // ~30 GiB > budget) so the compact set stays RESIDENT instead of dropping to STREAM/CPU. The driver's
          // pack/load honor ORK_NO_BF, so the decision is communicated by setting it INTERNALLY here — never a
          // user gate. An EXPLICIT user ORK_NO_BF (already read into `no_bf`) still wins.
          const bool want_bf = no_bf ? false : (base + bf_extra <= budget);
          if (!want_bf && !no_bf) {
              setenv("ORK_NO_BF", "1", 1);
              fprintf(stderr, "[ORK] Bf auto-shed: with-Bf %.2f GiB > RAM budget %.2f GiB — keeping the compact "
                              "%.2f GiB (Bb-only) set RESIDENT (decode full-K path off, prefill unaffected)\n",
                      (base + bf_extra)/(1024.0*1024*1024), budget/(1024.0*1024*1024), base/(1024.0*1024*1024));
          }
          const size_t inflated = want_bf ? (base + bf_extra) : base;
          // Per-domain fill TARGET 1.6 GiB (was 2.5): each domain must leave IOVA HEADROOM under the ~2.9 GiB
          // hard limit for the RUN's own scratch (mtk_all multi-core task buffer, output C, activation tiles) —
          // NOT just the weights. At 2.5 GiB the byte-balanced fill + a full-layer overshoot packed domain 0 to
          // ~2.95 GiB (past the hard edge) so mc_ensure could not allocate the run scratch (IOMMU full -> decode
          // -3). At 1.6 GiB the fill + overshoot stays ~2.3 GiB, leaving ~0.6 GiB per domain for run scratch.
          /* Per-domain fill cap: domains are page-cache-mapped and plentiful (count limit >> 8), so MANY light
           * domains is fine and keeps each domain's IOVA window mostly free for the run scratch. 1 GiB/domain. */
          const size_t cap = (size_t) 1000 * 1024 * 1024;
          long nd = (long) ((inflated + cap - 1) / cap);
          ctx->n_domains = nd < 1 ? 1 : (nd > 63 ? 63 : (int) nd);   // 63 = owned_dom bitmask ceiling (~155 GiB)
          if (ctx->n_domains > 0) ctx->domain_fill_cap = inflated / (size_t) ctx->n_domains + (size_t) 64 * 1024 * 1024;
          // Hard headroom clamp: never target a fill so high that fill + a full-layer overshoot could approach the
          // ~2.9 GiB hard IOVA edge and starve the run scratch (protects models the byte-balance leaves lumpy).
          if (ctx->domain_fill_cap > (size_t) 1900 * 1024 * 1024) ctx->domain_fill_cap = (size_t) 1900 * 1024 * 1024;
          ctx->residence_footprint = inflated;
          ctx->residence_ram_budget = budget;
          ctx->residence_stream = (inflated > budget) ? 1 : 0;       // even Bb-only overflows -> stream by layer
          if (getenv("ORK_VERBOSE"))
              fprintf(stderr, "[ORK] auto n_domains=%d (@%.2f GiB/dom) footprint %.2f GiB (Bf %s) vs RAM budget %.2f GiB -> %s\n",
                      ctx->n_domains, ctx->domain_fill_cap/(1024.0*1024*1024), inflated/(1024.0*1024*1024),
                      want_bf ? "on" : "auto-shed", budget/(1024.0*1024*1024), ctx->residence_stream ? "STREAM-by-layer" : "RESIDENT");
      } else {
          // No .orkpack index (live-pack / write mode): footprint unknown up front (weights arrive one matmul
          // at a time). Keep a domain ceiling; ork_weight_domain() fills only as many as the resident set needs.
          ctx->n_domains = 8;
      }
      // ORK_DOMAIN_LAYERS: explicit LAYOUT override ONLY (layers-per-domain). domain_layers stays 0 (its
      // default) unless set — the auto branch sizes by byte-balanced fill (ork_weight_domain), not layer
      // alignment. Guard on `dl` so an unset var can't clobber an explicit value set for a fusion experiment.
      const char * dl = getenv("ORK_DOMAIN_LAYERS"); if (dl) ctx->domain_layers = atoi(dl); }
    // Pre-size the driver's per-domain arrays to the auto-sized count (grows on demand otherwise; this just
    // avoids a mid-load realloc). The driver has NO fixed domain cap — the count is whatever we computed above.
    ork_npu_set_ndomains(ctx->npu, ctx->n_domains);
    // ORKD DOMAIN OWNERSHIP (the multi-domain 7B OOM fix). Under orkd the daemon REJECTS any pack/import into a
    // domain the client does not OWN (handle_import → "domain not owned by client"), so a weight the auto-sizer
    // placed in logical domain 1..n_domains-1 fails to import, the client ork_domain_advance()s through the rest,
    // and residency collapses into domain 0 only (measured: 7B loaded ~2 GiB/dom then OOM at layer ~10, vs direct
    // mode's 2.53 GiB × 5 = full model). So ACQUIRE ownership of each non-zero logical domain up front. The
    // daemon's dom_alloc_explicit hands out contiguous ids 1,2,3,… to a fresh client ⇒ physical == logical, no
    // remap needed; warn if that breaks (multi-client / pool exhaustion). Direct mode owns every domain implicitly.
    if (ctx->via_orkd && ctx->n_domains > 1) {
        for (int d = 1; d < ctx->n_domains; d++) {
            int got = ork_npu_domain_alloc(ctx->npu);
            if (got != d)
                fprintf(stderr, "[ork] WARNING: orkd domain acquire: logical %d -> physical %d (non-contiguous or pool exhausted); "
                                "multi-domain residency may still reject — a single fresh client should map 1:1\n", d, got);
            else if (getenv("ORK_VERBOSE"))
                fprintf(stderr, "[ork] orkd: acquired domain %d\n", got);
        }
    }
    // Residency = ork-driver's MULTI-DOMAIN mechanism (weights resident across up to 16 IOMMU domains,
    // each its own ~4 GiB IOVA window; dom_activate zero-copy-swaps the active domain per submit). This
    // resides up to ~64 GiB with no streaming — every practical model. The stream pool (RAM-hold + dma-buf
    // map/unmap) is only needed BEYOND that; leave it off (NULL) so the domain path is used.
    ctx->spool = nullptr;
    fprintf(stderr, "[ork] residency: multi-domain (up to %d domains x ~4 GiB IOVA, dom_activate swap)\n", ctx->n_domains);
    // NPU routing: DIRECT (in-process lib) by default, or through the orkd daemon when ORK_USE_ORKD=1 opted in
    // (serialized — the safe way to share the single-stream NPU across concurrent processes; direct concurrent
    // access wedges the IOMMU). Report whichever is actually in effect for the run log.
    fprintf(stderr, "[ork] routing: %s\n", ctx->via_orkd ? "orkd daemon (serialized) + shm ring"
                                                          : "direct in-process lib (single-stream)");
    // One-line version banner to stderr — visible even under llama-bench (which suppresses
    // GGML_LOG_INFO). Cheap, once per backend init. ork_npu_version() = semver (+git hash if built
    // with one). Makes "which build is this?" answerable from any benchmark/run log.
    fprintf(stderr, "[ork] ork-driver %s (W%dA%d%s)\n", ork_npu_version(),
            ctx->qbits, ctx->qbits, ctx->hadamard ? "+Had" : "");
    GGML_LOG_INFO("%s: ork backend ready (ork-driver %s, orkd %sW%dA%d%s)\n", __func__, ork_npu_version(),
                  ctx->hybrid ? "Hybrid " : "",
                  ctx->qbits, ctx->qbits,
                  ctx->hadamard ? "+Hadamard" : "");
    ggml_backend_t backend = new ggml_backend {
        /* .guid      = */ ggml_backend_ork_guid(),
        /* .interface = */ ork_backend_i,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_ork_reg(), 0),
        /* .context   = */ ctx,
    };
    return backend;
}

bool ggml_backend_is_ork(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_ork_guid());
}

// device interface

static const char * ggml_backend_ork_device_get_name(ggml_backend_dev_t dev) { return "ORK"; GGML_UNUSED(dev); }
static const char * ggml_backend_ork_device_get_description(ggml_backend_dev_t dev) { return "Rockchip NPU (ork-driver)"; GGML_UNUSED(dev); }
#include <unistd.h>
#include <fstream>
#include <string>

static void ggml_backend_ork_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    *free = 0;
    *total = 0;
    
    std::ifstream meminfo("/proc/meminfo");
    if (meminfo.is_open()) {
        std::string line;
        while (std::getline(meminfo, line)) {
            if (line.compare(0, 9, "MemTotal:") == 0) {
                size_t kb;
                if (sscanf(line.c_str(), "MemTotal: %zu kB", &kb) == 1) {
                    *total = kb * 1024;
                }
            } else if (line.compare(0, 13, "MemAvailable:") == 0) {
                size_t kb;
                if (sscanf(line.c_str(), "MemAvailable: %zu kB", &kb) == 1) {
                    *free = kb * 1024;
                }
            }
        }
    }
    
    // Fallback if parsing fails
    if (*total == 0) {
        *total = 8ull * 1024 * 1024 * 1024;
        *free = 8ull * 1024 * 1024 * 1024;
    }
    GGML_UNUSED(dev);
}
static enum ggml_backend_dev_type ggml_backend_ork_device_get_type(ggml_backend_dev_t dev) { return GGML_BACKEND_DEVICE_TYPE_ACCEL; GGML_UNUSED(dev); }

static void ggml_backend_ork_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_ork_device_get_name(dev);
    props->description = ggml_backend_ork_device_get_description(dev);
    props->type        = ggml_backend_ork_device_get_type(dev);
    ggml_backend_ork_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = { /* async */ false, /* host_buffer */ false, /* buffer_from_host_ptr */ true, /* events */ false };
}

static ggml_backend_t ggml_backend_ork_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_ork_init(); GGML_UNUSED(dev); GGML_UNUSED(params);
}
// ================= M1: ork's own weight buffer type =================
// MUST be distinct from ggml_backend_cpu_buffer_type() so make_cpu_buft_list (llama-model.cpp:880,
// "if (buft != cpu_buffer_type)") ADDS it to the weight-preference list. The loader then routes every
// matmul weight ork supports_op onto this buft; those weights live on the ork backend, so the scheduler
// runs their MUL_MAT on ork by weight-locality — which is what finally makes graph_compute fire and the
// NPU light up (previously get_buffer_type aliased the CPU buffer type -> line 880 skipped ork -> weights
// went to the aarch64 repack buffer (is_host=false) -> ork never got any op -> NPU 0%).
// Host-backed for M1 (holds the plain ggml bytes; CPU fallback + normal weight load work unchanged).
// M2: set_tensor will substitute the pre-tiled .orkpack blob (native NPU, mmap) instead of the ggml bytes.
static void * ggml_backend_ork_buffer_get_base(ggml_backend_buffer_t buffer) { return buffer->context; }
static void   ggml_backend_ork_buffer_free_buffer(ggml_backend_buffer_t buffer) { free(buffer->context); }
static void   ggml_backend_ork_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    memcpy((char *) tensor->data + offset, data, size); GGML_UNUSED(buffer);
}
static void   ggml_backend_ork_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    memcpy(data, (const char *) tensor->data + offset, size); GGML_UNUSED(buffer);
}
static void   ggml_backend_ork_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    memset((char *) tensor->data + offset, value, size); GGML_UNUSED(buffer);
}
static bool   ggml_backend_ork_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    if (ggml_backend_buffer_is_host(src->buffer)) { memcpy(dst->data, src->data, ggml_nbytes(src)); return true; }
    return false; GGML_UNUSED(buffer);
}
static void   ggml_backend_ork_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) { memset(buffer->context, value, buffer->size); }

static const struct ggml_backend_buffer_i ggml_backend_ork_buffer_i = {
    /* .free_buffer   = */ ggml_backend_ork_buffer_free_buffer,
    /* .get_base      = */ ggml_backend_ork_buffer_get_base,
    /* .init_tensor   = */ NULL,
    /* .memset_tensor = */ ggml_backend_ork_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_ork_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_ork_buffer_get_tensor,
    /* .set_tensor_2d = */ NULL,
    /* .get_tensor_2d = */ NULL,
    /* .cpy_tensor    = */ ggml_backend_ork_buffer_cpy_tensor,
    /* .clear         = */ ggml_backend_ork_buffer_clear,
    /* .reset         = */ NULL,
};
static const char * ggml_backend_ork_buffer_type_get_name(ggml_backend_buffer_type_t buft) { return "ORK_Weights"; GGML_UNUSED(buft); }
static ggml_backend_buffer_t ggml_backend_ork_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = NULL;
    if (posix_memalign(&data, 64, size ? size : 64) != 0) return NULL;
    g_ork_weight_bytes += size;   // (diagnostic only; ork weights live on CPU buffers so this stays ~0)
    return ggml_backend_buffer_init(buft, ggml_backend_ork_buffer_i, data, size);
}
static size_t ggml_backend_ork_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) { return 64; GGML_UNUSED(buft); }
static bool   ggml_backend_ork_buffer_type_is_host(ggml_backend_buffer_type_t buft) { return true; GGML_UNUSED(buft); }
static ggml_backend_buffer_type_t ggml_backend_ork_buffer_type(void) {
    static struct ggml_backend_buffer_type buft = {
        /* .iface = */ {
            /* .get_name       = */ ggml_backend_ork_buffer_type_get_name,
            /* .alloc_buffer   = */ ggml_backend_ork_buffer_type_alloc_buffer,
            /* .get_alignment  = */ ggml_backend_ork_buffer_type_get_alignment,
            /* .get_max_size   = */ NULL,
            /* .get_alloc_size = */ NULL,
            /* .is_host        = */ ggml_backend_ork_buffer_type_is_host,
        },
        /* .device  = */ NULL,
        /* .context = */ NULL,
    };
    if (buft.device == NULL) buft.device = ggml_backend_reg_dev_get(ggml_backend_ork_reg(), 0);
    return &buft;
}
static ggml_backend_buffer_type_t ggml_backend_ork_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_ork_buffer_type(); GGML_UNUSED(dev);
}
static ggml_backend_buffer_t ggml_backend_ork_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size); GGML_UNUSED(dev); GGML_UNUSED(max_tensor_size);
}


static bool ggml_backend_ork_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    static const int ork_off = getenv("ORK_OFF") != nullptr;   // CPU baseline: force everything to CPU
    if (ork_off) return false;
    // orkd: only MUL_MAT is daemon-routed (ork_i8_mm_pack + ork_i8_mm_run). The other NPU ops here — MoE
    // (MUL_MAT_ID), attention (SOFT_MAX/FLASH_ATTN), SDP activations (GLU/UNARY/MUL/ADD), SSM_SCAN — run on
    // fd-local primitives (run_i8_silu, doorbell, dom_activate, DMA scratch) that break when the daemon owns
    // the NPU. Decline them so ggml keeps them on the CPU backend (correct; they are all opt-in/experimental
    // and default-off anyway). Routing these through orkd is the follow-on. MUL_MAT + the metadata ops
    // (NONE/RESHAPE/VIEW/PERMUTE/TRANSPOSE) fall through to the normal per-op logic below.
    if (g_ork_ctx && g_ork_ctx->via_orkd) {
        switch (op->op) {
            case GGML_OP_NONE: case GGML_OP_RESHAPE: case GGML_OP_VIEW:
            case GGML_OP_PERMUTE: case GGML_OP_TRANSPOSE:
            case GGML_OP_MUL_MAT:
                break;
            case GGML_OP_FLASH_ATTN_EXT:
                // task #20: the int8 DECODE attention path (ggml_backend_ork_flash_attn_decode) is
                // orkd-routed (ork_i8_mm_run_chain / ork_i8_mm_pack), unlike the fp16 prefill path and the
                // other fd-local primitives this gate declines. Let it through ONLY under ORK_ATTN_DEC so
                // the real FLASH_ATTN_EXT case below applies the full decode gate (N==1 + shape). Off by
                // default: without ORK_ATTN_DEC this still declines to CPU, preserving orkd safety.
                if (getenv("ORK_ATTN_DEC")) break;
                return false;
            case GGML_OP_GLU:
                // ORK_FFN_DEC: the fused DECODE FFN chain (ggml_backend_ork_ffn_decode_orkd) needs the SwiGLU
                // node in the ORK split so the matcher keeps the gate/up/glu/down subgraph whole. The node's
                // own compute is consumed (skipped) by the chain; it only needs to be scheduled to ORK.
                if (getenv("ORK_FFN_DEC")) break;
                return false;
            default:
                return false;
        }
    }
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    // MoE AUTO-PROFILE detection: a MUL_MAT_ID op means this model is a routed MoE -> enable the measured
    // MoE profile (expert-int4 tier + experts-on-CPU + decode-on-CPU) with NO env knobs. Latches once.
    if (op->op == GGML_OP_MUL_MAT_ID && !g_ork_is_moe) {
        g_ork_is_moe = true;
        if (!(getenv("ORK_MOE_AUTO") && atoi(getenv("ORK_MOE_AUTO")) == 0))
            fprintf(stderr, "[ork] MoE model detected -> auto profile: experts int4/NF4 on CPU (batched NEON), "
                            "attn/dense int8 on NPU at prefill, all-CPU at decode. (ORK_MOE_AUTO=0 to disable.)\n");
    }
    if (getenv("ORK_VERBOSE") && op->op == GGML_OP_MUL_MAT_ID)   // DIAG: is ork ever asked about the MoE op?
        fprintf(stderr, "[ORK MMID-ENTRY] name=%s type=%s buft=%s via_orkd=%d\n",
            src0 ? src0->name : "?", src0 ? ggml_type_name(src0->type) : "?",
            (src0 && src0->buffer) ? ggml_backend_buft_name(src0->buffer->buft) : "(none)",
            (g_ork_ctx && g_ork_ctx->via_orkd) ? 1 : 0);
    switch (op->op) {
        case GGML_OP_NONE: case GGML_OP_RESHAPE: case GGML_OP_VIEW:
        case GGML_OP_PERMUTE: case GGML_OP_TRANSPOSE:
            return true;
        case GGML_OP_MUL_MAT: {
            const int64_t K = src0->ne[0], N = op->ne[0], M = op->ne[1];
            // ORK_ATTN: accept BATCHED / dynamic-operand matmul (attention QKᵀ·V, Gated-Delta-Net chunked
            // matmuls) for the fp16 ork_bmm path — the static-weight gate below declines these (ne[2]/ne[3]>1,
            // computed src0). Require contiguous f16/f32 operands + K%32, N%16 (fp16 tile). Gated (off by
            // default) — this is the experimental attention/GDN offload; measure NPU% + coherence before
            // enabling by default.
            {
                static const int ork_attn = getenv("ORK_ATTN") != nullptr;
                if (ork_attn && (op->ne[2] > 1 || op->ne[3] > 1)) {
                    // DEDICATED ATTENTION DISPATCH: claim batched/dynamic matmuls ONLY for M>1 (prefill).
                    // M=1 (decode) is DECLINED here so ggml keeps it on the CPU backend — never letting a
                    // dynamic KV-cache operand fall into the int8 static-weight resolve path (which packs it
                    // as a weight -> "not in orkpack" slow live-convert) NOR into the M=1 in-graph A·V wedge.
                    // Decode attention is a CPU path anyway (decode-is-cpu-path); the parity target is prefill.
                    // NOTE: contiguity NOT required — the bmm handler gathers permuted/non-contig operands
                    // via nb[] strides (rds). This is the densify unblock: real QK^T/A·V read permuted-Q /
                    // non-contig KV-cache views that ggml_is_contiguous previously declined.
                    return M > 1 && K % 32 == 0 && N % 16 == 0
                        && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)
                        && (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
                }
            }
            // Explicitly block output and lm_head (vocabulary projection) layers from offloading to NPU.
            // These layers are extremely wide (e.g. N=151936), causing massive DMA buffer allocation and
            // packing overhead, which can trigger NPU driver IOVA allocation failures and kernel hangs.
            const char * name = src0->name;
            if(getenv("ORK_VERBOSE"))fprintf(stderr, "[ORK DEBUG supports_op] name='%s' K=%ld N=%ld M=%ld\n", name, (long)K, (long)N, (long)M);
            fflush(stderr);
            // q4/low-bit GGUF sources are UNSUPPORTED territory for the NPU path (this SoC's int4 MAC is
            // symmetric W4A4 — no coherent int4-resident mode — and re-quantizing q4->int8 is lossy
            // double-quant). Route any sub-5-bit source (Q4_*/Q3_K/Q2_K/IQ*) to CPU, and — since supports_op
            // also gates the convert-time forward — this stops the .orkpack from ever packing a q4 model.
            // Escape hatch: an explicit int4 RESEARCH mode (ORK_QUANT=4 / ORK_HYBRID / ORK_ORKPACK_TIERMAP)
            // still opts into the experimental int4 path — as does simply having LOADED an int4 .orkpack,
            // which is a deliberate enough act to count as opting in. ORK_MIXED_DISPATCH also opts in: it
            // ACCEPTS the sub-5-bit tensors and runs them native-W4A4 (per-tensor dispatch in graph_compute),
            // keeping the >4-bit tensors on W8A8 — the mixed-precision q4 NPU path.
            {
                static const int i4_env = ((getenv("ORK_QUANT") && getenv("ORK_QUANT")[0] == '4')
                    || getenv("ORK_HYBRID") || getenv("ORK_ORKPACK_TIERMAP")) ? 1 : 0;
                // Read the ctx live rather than folding it into the static: supports_op can be reached
                // before backend init has published g_ork_ctx, and a cached 0 would stick for the run.
                const int i4_research = i4_env || (g_ork_ctx && g_ork_ctx->qbits == 4);
                if (!i4_research && !ork_mixed_dispatch_on()) {
                    double sbits = ork_src_type_bits(src0->type);
                    if (sbits >= 0.0 && sbits < 5.0) return false;   // q4/low-bit source -> CPU-only
                }
            }
            // N-cap: keep only the EXTREME vocab-projection (lm_head/output, N~152k, ~0.5 GiB resident,
            // once-per-token) on CPU — it's an IOVA-budget policy, NOT a matmul limit. Byte-exact sweep
            // (int8 K=3584, isolated, 2026-06-30): N=18944/32768/65536/131072/152064 ALL bit-exact, no
            // hang — the NPU handles any real width via N-tiling. The old default 16384 was arbitrary and
            // wrongly EXCLUDED the FFN gate/up (N=18944, the bulk of NPU-worthy compute), forcing it to CPU.
            // Default 32768: FFN in (with headroom for wider models), 152k vocab out. ORK_MAXN overrides
            // (e.g. =200000 to put lm_head on NPU too for a full-residence test).
            static const int max_n = getenv("ORK_MAXN") ? atoi(getenv("ORK_MAXN")) : 32768;
            // LEVER D: dropped strstr("output")/strstr("lm_head") over-match (it wrongly declined attn_output.weight,
            // a normal [3584,3584] matmul N=3584). Real lm_head (output.weight N=152064) still excluded by N>max_n.
            if (N > max_n) {
                return false;
            }
            // Measured (RK3588, Qwen3-1.7B-w8a8): the ~365us/matmul NPU submit floor makes per-token
            // DECODE (M=1) a net LOSS vs CPU (4.7 vs 9.4 tok/s) — ~197 submits/token at 365us each is
            // ~72ms before any compute benefit, and M=1 matmuls are tiny. PREFILL (large M) is the
            // opposite: M>1 amortizes the floor over many rows, so NPU wins (39.6 vs 13.6 tok/s).
            // Gate on M (the token/batch dim) ONLY — NOT N. The old `M>=min || N>=min` always passed
            // because every weight has a large N, dragging M=1 decode onto the NPU. ORK_MINM tunes it.
            static const int min_m = getenv("ORK_MINM") ? atoi(getenv("ORK_MINM")) : 32;
            int target_qbits = g_ork_ctx ? g_ork_ctx->qbits : ((getenv("ORK_QUANT") && getenv("ORK_QUANT")[0] == '4') ? 4 : 8);
            bool hybrid = g_ork_ctx ? g_ork_ctx->hybrid : (g_ork_hybrid_loading || getenv("ORK_HYBRID") != nullptr);
            const char * name_src = src0->name;
            bool is_expert = ork_is_expert(name_src);
            if (hybrid) {
                bool is_ffn = strstr(name_src, "ffn_") || is_expert;
                bool is_attn = strstr(name_src, "attn_q") || strstr(name_src, "attn_k") || strstr(name_src, "attn_v") || strstr(name_src, "attn_output");
                if (!is_ffn && !is_attn) {
                    return false; // Keep on CPU NEON or Mali GPU
                }
                if (is_ffn) target_qbits = 4;
                else if (is_attn) target_qbits = 8;
            }

            // Residency does NOT make single-token (M=1) decode worth it for dense layers — the per-submit
            // floor dominates regardless. Keep the M threshold so dense decode stays on CPU.
            // Bypassed for expert layers (MoE) where CPU weight streaming is a catastrophic ~32ms bottleneck.
            // Native W4A4 is env-selected, so this reads the same before and after the ctx exists.
            bool hadamard = g_ork_ctx ? (bool) g_ork_ctx->hadamard : ork_w4a4_native_on();
            bool is_grouped = (src0->type == GGML_TYPE_Q4_0  ||
                               src0->type == GGML_TYPE_Q4_1  ||
                               src0->type == GGML_TYPE_Q4_K  ||
                               src0->type == GGML_TYPE_IQ4_NL ||
                               src0->type == GGML_TYPE_IQ4_XS) && !hadamard;
            int threshold = is_expert ? 1 : (is_grouped ? min_m : (min_m > 32 ? 32 : min_m));
            // i8 M=1-on-NPU chaining is a win ONLY for single-domain (≤4GiB) models. For multi-domain (>4GiB)
            // the M=1 decode submit faults on imported weights in a non-0 IOMMU domain (mcworker_dec_active
            // errno 22) AND loses to CPU decode anyway — so keep the M threshold there, routing decode to CPU
            // (the validated winning config; see decode-is-cpu-path). Single-domain keeps the fast M=1 NPU path.
            // CONVERT/WRITE mode (persist_mode==2) MUST keep M=1 on the NPU: the .orkpack build is a single
            // 1-token (M=1) forward that packs every weight, so declining M=1 there would pack ZERO weights
            // and produce no .orkpack for >4GiB models. Only SERVING (read/off) routes multi-domain M=1 to CPU.
            // CONVERT/WRITE (persist_mode==2) MUST keep M=1 on NPU (packs every weight). For SERVING,
            // single-domain M=1-on-NPU is submit-floor-bound and LOSES to CPU on MoE models — profiled on
            // LFM2.5/Qwen3.6-35B: ~1440 M=1 run_i8 submits/decode = 82% of run time, decode 14/6.16 vs ggml
            // CPU 20/7. Route single-domain serving decode to CPU too (threshold stays min_m); ORK_M1_NPU
            // restores the old always-NPU behavior for the dense-single-domain case it was tuned for.
            if (g_ork_ctx && g_ork_ctx->persist_mode == 2) threshold = 1;   // WRITE/convert: force M>=1 on NPU for EVERY dtype (int8 AND int4) so every weight packs — else int4 FFN falls to CPU and packs ZERO
            else if (target_qbits == 8 && (!g_ork_ctx || g_ork_ctx->n_domains <= 1) && env_enabled("ORK_M1_NPU")) threshold = 1;
            // EXPERIMENT #1 (ORK_MOE_PHASE_EVICT): at DECODE (M==1) DECLINE the dense backbone matmuls so
            // the scheduler routes them to CPU (bandwidth-bound, cheap at M=1) — this frees the ~2.8 GiB of
            // IOVA the backbone otherwise pins, handing it to the MoE hot-expert cache. Experts go through
            // MUL_MAT_ID (a separate case, still accepted). MUL_MAT here is the dense/attn backbone (the
            // _exps tensors never reach this case), so declining at M==1 is exactly the backbone-at-decode.
            {
                static const int pe = env_enabled("ORK_MOE_PHASE_EVICT");
                // NEVER decline M==1 in WRITE/convert mode: the .orkpack is built by a single 1-token forward,
                // so declining M==1 there would pack ZERO dense weights (and leave a useless pack behind).
                const bool writing = g_ork_ctx && g_ork_ctx->persist_mode == 2;
                if ((pe || ork_moe_auto()) && !writing && M == 1 && op->ne[2] == 1 && op->ne[3] == 1) return false;
            }
            // ORK_FFN_DEC: admit the DECODE (M==1) FFN gate/up/down to the NPU so the whole gate/up/GLU/down
            // subgraph is ONE contiguous ORK split (a CPU-side up would fragment it and the matcher couldn't
            // span the splits). The gate-anchored matcher fuses all 4 (i=last skips up/GLU/down — up never
            // runs standalone). Scoped by weight name so other decode matmuls (attention proj) stay on CPU.
            if (getenv("ORK_FFN_DEC") && g_ork_ctx && g_ork_ctx->via_orkd && M == 1 &&
                op->ne[2] == 1 && op->ne[3] == 1 && src0->name &&
                (strstr(src0->name, "ffn_gate") || strstr(src0->name, "ffn_up") || strstr(src0->name, "ffn_down"))) {
                const char * mn = getenv("ORK_FFN_DEC_MIN");   // diagnostic: only fire FFN-dec for layer >= MIN
                if (!mn || ork_layer_of(src0->name) >= atoi(mn)) return true;
            }
            bool pass_m_threshold = (M >= threshold || (M > 1 && (op->ne[2] > 1 || op->ne[3] > 1)));

            // src0 must be a plain 2-D STATIC weight. mul_mat_i8 resolves src0 as ONE resident 2-D
            // weight keyed on its data pointer (packed/tiled once, cached). A BATCHED src0 (ne[2]/ne[3]>1,
            // e.g. per-head Gated-Delta-Net state or attention-score K) is multiple weights → mispacked;
            // a DYNAMIC/computed src0 (op != NONE — GDN delta-net intermediates like k_cumdecay/v_t_new,
            // views/reshapes of activations) is repacked every token → IOVA churn/exhaustion. Both
            // segfaulted mul_mat_i8 on Qwen3.5/3.6 (GDN arch). Require a leaf 2-D weight → the rest go to CPU.
            const bool src0_static_2d = src0->ne[2] == 1 && src0->ne[3] == 1 && src0->op == GGML_OP_NONE;
            bool ork_accept = pass_m_threshold && src0_static_2d &&
                   ggml_is_contiguous(src0) && ggml_is_contiguous(src1) &&
                   src1->type == GGML_TYPE_F32 &&
                   K % 32 == 0 && N % 64 == 0 &&
                   K >= 32 &&
                   src1->ne[2] % src0->ne[2] == 0 &&
                   src1->ne[3] % src0->ne[3] == 0 &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->to_float != NULL);
            if (getenv("ORK_VERBOSE") && name && strstr(name, "attn_output"))
                fprintf(stderr, "[ORK LEVERD attn_output] M=%ld contigSrc0=%d contigSrc1=%d pass_m=%d -> ACCEPT=%d\n", (long)M, (int)ggml_is_contiguous(src0), (int)ggml_is_contiguous(src1), (int)pass_m_threshold, (int)ork_accept);
            return ork_accept;
        }
        case GGML_OP_SOFT_MAX: {
            // 2b: attention softmax on the NPU (exp via the int16 SDP act-LUT; max/sum/divide stay CPU).
            // Claim only what the handler runs on the NPU and always completes: prefill rows (ne[1]>1),
            // fp32 scores, softmax width ne[0]%32==0 (the ork_i16_npu_exp tile), and NO ALiBi slope
            // (max_bias==0 — op_params[1]). Everything else stays on the CPU backend. Part of the
            // attention block, gated with ORK_ATTN.
            static const int ork_attn = getenv("ORK_ATTN") != nullptr;
            // ORK_ATTN_SM_CPU=1 must ALSO un-claim the standalone SOFT_MAX, not just the FA-internal softmax.
            // It previously only switched ggml_backend_ork_flash_attn_ext's inner softmax (the `sm_npu` flag),
            // so with ORK_ATTN=1 this claim stayed live and swallowed EVERY qualifying softmax in the graph —
            // including, on a routed-MoE model, the **expert-router softmax** (ne[0]=n_expert, %32==0;
            // ne[1]=n_tokens>1 in prefill). Routing weights then came back through the coarse int16 exp LUT
            // (in_scale = -lo/32000) instead of fp32 expf, which perturbs top-k expert selection and their
            // mixing weights. Measured on qwen3.6-35B-A3B: PPL 70.06 -> 67.86, i.e. real but only ~3%.
            // (The 10.83 -> 67.86 blowup was NOT this: it was ggml_backend_ork_bmm_fp16 feeding the driver
            // a TRANSPOSED B — see the DEFECT FIX comment there. Both are invisible to ORK_ATTN_CPU, which
            // only moves the FLASH_ATTN_EXT handler's matmuls.) The exp LUT is fine for attention scores (renormalized per row, differences
            // only) but not for a 128-way router distribution. Keep this claim tied to the same flag that
            // gates the NPU softmax everywhere else — which is now OPT-IN (ORK_ATTN_SM_NPU), because
            // ork_i16_npu_exp is also quality-broken at some row counts (see the FA handler's step (2)).
            // So plain ORK_ATTN=1 no longer claims SOFT_MAX at all; ORK_ATTN_SM_CPU stays accepted as the
            // explicit "keep softmax on the CPU" alias (it now only has to override an explicit SM_NPU).
            static const int sm_npu = getenv("ORK_ATTN_SM_NPU") != nullptr && getenv("ORK_ATTN_SM_CPU") == nullptr;
            float mb = 0.0f; memcpy(&mb, (const char *) op->op_params + sizeof(float), sizeof(float));
            return ork_attn && sm_npu && op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32
                && op->ne[1] > 1 && op->ne[0] % 32 == 0 && mb == 0.0f;
        }
        case GGML_OP_RMS_NORM:
        case GGML_OP_ROPE: {
            // #1 static-graph prerequisite: claim the movable boundary ops RMSNorm + RoPE so a whole layer
            // arrives as ONE ork subgraph. Computed via ork_cpu_delegate_node (ggml's own CPU kernel —
            // coherent) for now; the on-NPU-native + chained versions land with the assembler. Gated
            // ORK_CLAIM_OPS (off by default). Residual ADD / norm-weight MUL are the existing ORK_PPU_ADD/OPS.
            static const int claim = getenv("ORK_CLAIM_OPS") != nullptr;
            return claim;
        }
        case GGML_OP_MUL_MAT_ID: {
            // MoE expert offload to the NPU (hot-expert partition: conforming-K experts go NPU-resident
            // via the async round-robin stream, the rest stay on the threaded CPU GEMV). EXPERIMENTAL and
            // OFF BY DEFAULT — NOT recommended on RK3588-class hardware. M2 verdict (Qwen3-MoE, board
            // 10.3.0.236, -t 4): NPU decode ~6.56 t/s vs CPU ~19.09 t/s (~2.9x SLOWER). The walls are
            // RK3588-specific: (1) ~1.2 GiB usable IOVA — the 32-bit NPU IOMMU caps mappable weights at
            // ~4 GiB and the GGUF backbone already eats ~2.8 GiB, so the resident hot set tops out at a
            // ~17% expert hit-rate; (2) every cold-expert miss is a per-token GEMV that is LPDDR4X-
            // bandwidth-bound and loses to ggml's fused batched MUL_MAT_ID. REVISIT when any of these
            // changes: a wider-IOVA device (more resident experts), DDR5 / higher memory bandwidth
            // (cold path competitive), or the M>1 regime (batched-verify / prefill amortizes the submit).
            // Enable with ORK_MOE_NPU=1 (truthy: 1/true/yes/on; UNSET or 0/false/off => experts on CPU;
            // legacy alias: ORK_NO_EXPERT_REPACK). NOTE: the matching repack-buffer exclusion in
            // ggml-cpu/repack.cpp is gated on bare getenv presence, so to actually route experts to the
            // NPU set the var to a truthy value (don't disable here by setting 0 while leaving it present).
            // ORK_MOE_CPU (task #54, NF4/int4 CPU route): claim MUL_MAT_ID so the handler runs and routes
            // every expert to the batched CPU int4/NF4 NEON GEMM (the "prefill using only int4 from orkpack
            // on CPU" A/B). Also required in WRITE mode so persist_write_experts emits the O4N1 experts.
            const bool cpu_only_op = env_enabled("ORK_MOE_CPU") || (ork_moe_auto() && !env_enabled("ORK_MOE_NPU"));
            if (!cpu_only_op && !env_enabled("ORK_MOE_NPU") && !env_enabled("ORK_NO_EXPERT_REPACK")) return false;
            if (!cpu_only_op) {   // one-time loud warning when the experimental MoE-on-NPU path is actually enabled
                static bool warned = false;
                if (!warned) { warned = true;
                    fprintf(stderr, "[ork] WARNING: ORK_MOE_NPU MoE-on-NPU expert offload is EXPERIMENTAL "
                        "and NOT recommended on RK3588-class hardware (4GiB IOVA + LPDDR4X) — it loses ~3x "
                        "vs CPU at M=1 decode. Intended for wider-IOVA / DDR5 devices or the M>1 "
                        "batched-verify path.\n");
                }
            }
            const struct ggml_tensor * a = op->src[0];   // experts [K, N, n_expert]
            const struct ggml_tensor * b = op->src[1];   // input
            const struct ggml_tensor * c = op->src[2];   // ids (i32)
            const int64_t K = a->ne[0], N = a->ne[1];
            // NOTE: must accept ALL n_tokens (not just decode) — the graph split is planned with a
            // worst-case multi-token batch, and gating on b->ne[2]==1 made the planner leave the
            // experts on CPU. The handler loops over tokens, so multi-token is handled (correctly,
            // if not yet optimally — see the prefill group-by-expert TODO).
            const bool ok =
                   b->type == GGML_TYPE_F32 && c && c->type == GGML_TYPE_I32 &&
                   K % 32 == 0 && N % 32 == 0 && K >= 32 && N <= 8192 &&
                   c->ne[0] >= 1 && c->ne[0] <= 1024 &&
                   ggml_is_contiguous(b) &&
                   (a->type == GGML_TYPE_F32 || ggml_get_type_traits(a->type)->to_float != NULL);
            if(getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK supid] name=%s K=%ld N=%ld bne2=%ld contigB=%d cont_a=%d -> %d\n",
                a->name, (long)K, (long)N, (long)b->ne[2], (int)ggml_is_contiguous(b), (int)ggml_is_contiguous(a), (int)ok);
            return ok;
        }
        case GGML_OP_MUL:
        case GGML_OP_ADD: {
            // fp16 SwiGLU multiply / residual add (ork_f16_npu_ewmul / add_f16). EXPERIMENTAL,
            // default-off. MUL under ORK_PPU_OPS (bit-exact fp16); ADD needs ORK_PPU_ADD (fp16
            // residual accumulation caveat). Only the same-shape contiguous f32 case (SwiGLU
            // gate*up, residual x+y); broadcast MUL (norm-weight, RoPE) stays on CPU.
            // #1 static-graph contiguity: ORK_CLAIM_OPS claims ANY MUL/ADD (incl the broadcast norm-weight
            // MUL and the residual ADD) via CPU-delegate, so the whole layer is one ork subgraph.
            if (getenv("ORK_CLAIM_OPS")) return true;
            const bool on = (op->op == GGML_OP_MUL) ? ork_ppu_ops_on() : ork_ppu_add_on();
            if (!on) return false;
            if (!src0 || !src1) return false;
            if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) return false;
            if (!ggml_are_same_shape(src0, src1) || !ggml_are_same_shape(src0, op)) return false;
            if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(op)) return false;
            const int64_t N = op->ne[0], ne = ggml_nelements(op), M = ne / N;
            return N >= 8 && N <= 8192 && (N & 7) == 0 && M >= ork_ppu_minm() && M <= 8192;
        }
        case GGML_OP_UNARY: {
            // int8 SILU / GELU (ork_i8_npu_silu / gelu_i8). EXPERIMENTAL, per-op opt-in
            // (ORK_PPU_SILU / ORK_PPU_GELU) — int8 activation-quant is a quality trade.
            const enum ggml_unary_op u = ggml_get_unary_op(op);
            const bool on = (u == GGML_UNARY_OP_SILU) ? ork_ppu_silu_on()
                          : (u == GGML_UNARY_OP_GELU) ? ork_ppu_gelu_on() : false;
            if (!on) return false;
            if (!src0 || src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) return false;
            if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(op) || !ggml_are_same_shape(src0, op)) return false;
            const int64_t N = op->ne[0], ne = ggml_nelements(op), M = ne / N;
            return N >= 16 && N <= 8192 && (N & 15) == 0 && M >= ork_ppu_minm() && M <= 8192;
        }
        case GGML_OP_GLU: {
            // SwiGLU (split form: silu(gate=src0) * up=src1) on the NPU. EXPERIMENTAL, ORK_PPU_GLU.
            // ORK_FFN_CHAIN also needs GLU on ork so the FFN's 4 nodes land in one ork subgraph (fused there).
            if (!ork_ppu_glu_on() && !ork_ffn_chain_on() && !getenv("ORK_FFN_DEC")) return false;
            { const char * mn = getenv("ORK_FFN_DEC_MIN");   // match the FFN-matmul layer gate (src0=gate MM node -> its weight)
              if (mn && src0 && src0->src[0] && ork_layer_of(src0->src[0]->name) < atoi(mn)) return false; }
            if (ggml_get_glu_op(op) != GGML_GLU_OP_SWIGLU) return false;
            if (!src0 || !src1) return false;                        // split form only (two inputs)
            if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) return false;
            if (!ggml_are_same_shape(src0, src1) || !ggml_are_same_shape(src0, op)) return false;
            if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(op)) return false;
            const int64_t N = op->ne[0], ne = ggml_nelements(op), M = ne / N;
            // ORK_FFN_CHAIN consumes the GLU INSIDE the fused chain handler (the ewmul runs on CPU/tiled), so
            // it is NOT bound by the standalone ork_i8_npu_ewmul N<=8192 cap. Claim support at any N so the
            // FFN's gate/up/GLU/down land in ONE ork subgraph and the chain matcher can fuse them — otherwise
            // the 7B's Nff=18944 GLU is rejected, the scheduler splits the FFN across backends, and the 4
            // nodes never share a graph_compute (chain can never fire). Standalone GLU (ORK_PPU_GLU) keeps cap.
            if (ork_ffn_chain_on() || getenv("ORK_FFN_DEC")) return N >= 16 && (N & 15) == 0 && M >= 1 && M <= 8192;
            return N >= 16 && N <= 8192 && (N & 15) == 0 && M >= ork_ppu_minm() && M <= 8192;
        }
        case GGML_OP_SSM_SCAN: {
            // Mode-5 on-NPU Mamba-2 scan. SIZE-GATED default (2026-07-13): wins ~2x at Q8 2.7B+ but LOSES on
            // small Mamba-2 (130m/nh=24) & decode, so auto-enable ONLY for large models (nh>=64 ≈ >=1.5B;
            // head_dim/d_state are fixed in the family so nh is the size proxy). ORK_SSM_NPU=1 forces on at
            // any size (A/B), =0 forces off. Unset = auto.
            const char *ssm_env = getenv("ORK_SSM_NPU");
            const int ssm_force = ssm_env ? atoi(ssm_env) : -1;
            if (ssm_force == 0) return false;
            const struct ggml_tensor *S=op->src[0],*X=op->src[1],*A=op->src[3],*B=op->src[4];
            if (!S||!X||!A||!B) return false;
            const int nc=(int)S->ne[0], nr=(int)S->ne[1], nh=(int)X->ne[1], ng=(int)B->ne[1], nt=(int)X->ne[2], ns=(int)X->ne[3];
            if (A->ne[0] != 1) return false;        // Mamba-2 scalar decay only
            if (nt < 64 || ns != 1) return false;   // prefill single-seq (chunking worthwhile); decode/multi-seq -> CPU
            if (nc%32 || nr%16 || nh%ng) return false;
            if (ssm_force < 0 && nh < 64) return false;   // AUTO: small Mamba-2 (nh<64 ≈ <1.5B) stays on CPU
            for (int k=0;k<6;k++) if (op->src[k] && !ggml_is_contiguous(op->src[k])) return false;
            return true;
        }
        case GGML_OP_FLASH_ATTN_EXT: {
            // NPU attention. PREFILL (N>=64): fp16 batched QK^T+A·V, opt-in ORK_ATTN. DECODE (N==1): int8 path
            // on the NPU under orkd, opt-in ORK_ATTN_DEC (the fp16 path is prefill-only + can't run under orkd).
            const struct ggml_tensor *q=op->src[0],*k=op->src[1],*v=op->src[2];
            if (!q||!k||!v || op->src[4]) return false;              // src[4]=sinks -> CPU (v1)
            float max_bias=0.0f, softcap=0.0f;
            memcpy(&max_bias,(char*)op->op_params+4,4); memcpy(&softcap,(char*)op->op_params+8,4);
            if (max_bias!=0.0f || softcap!=0.0f) return false;      // v1: no ALiBi / softcap
            const int DK=(int)q->ne[0], N=(int)q->ne[1], H=(int)q->ne[2], Hkv=(int)k->ne[2], DV=(int)v->ne[0];
            const int nkv=(int)k->ne[1];
            { static int nq=0; if (getenv("ORK_ATTN_DEC") && nq++ < 8) fprintf(stderr,"[ork-fa-supp] FLASH_ATTN_EXT queried: N=%d nkv=%d DK=%d DV=%d Hkv=%d\n", N, nkv, DK, DV, Hkv); }
            if (Hkv<1 || H%Hkv || DK%32 || DV%16) return false;
            if (N == 1) {                                            // DECODE -> int8 orkd path (ggml_backend_ork_flash_attn_decode)
                if (getenv("ORK_ATTN_DEC") == nullptr) return false;
                if (DK > 512 || nkv < 256 || ((nkv+511)&~511) > 2048) return false;  // sched floor .. single-N-tile
                return true;
            }
            if (getenv("ORK_ATTN") == nullptr) return false;
            if (N < 64) return false;                                // prefill only
            return true;
        }
        default:
            return false;
    }
    GGML_UNUSED(dev);
}

static bool ggml_backend_ork_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft); GGML_UNUSED(dev);
}

// This is a buffer-less (BLAS-style) backend: weights live on the CPU buffer, so the scheduler only
// routes an op to us if offload_op() returns true. Mirror supports_op so MUL_MAT_ID (MoE experts)
// actually gets offloaded — without this, supports_op=true alone leaves the experts on CPU.
static bool ggml_backend_ork_device_offload_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    bool r = ggml_backend_ork_device_supports_op(dev, op);
    if (getenv("ORK_VERBOSE") && op->op == GGML_OP_MUL_MAT_ID)
        fprintf(stderr, "[ORK offload_op] MUL_MAT_ID name=%s src0_usage=%d -> %d\n",
            op->src[0]->name, op->src[0]->buffer ? (int)op->src[0]->buffer->usage : -99, (int)r);
    return r;
}

static const struct ggml_backend_device_i ggml_backend_ork_device_i = {
    /* .get_name             = */ ggml_backend_ork_device_get_name,
    /* .get_description      = */ ggml_backend_ork_device_get_description,
    /* .get_memory           = */ ggml_backend_ork_device_get_memory,
    /* .get_type             = */ ggml_backend_ork_device_get_type,
    /* .get_props            = */ ggml_backend_ork_device_get_props,
    /* .init_backend         = */ ggml_backend_ork_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_ork_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_ork_device_get_buffer_type,
    /* .buffer_from_host_ptr = */ ggml_backend_ork_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_ork_device_supports_op,
    /* .supports_buft        = */ ggml_backend_ork_device_supports_buft,
    /* .offload_op           = */ ggml_backend_ork_device_offload_op,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// backend reg interface

static const char * ggml_backend_ork_reg_get_name(ggml_backend_reg_t reg) { return "ORK"; GGML_UNUSED(reg); }
static size_t ggml_backend_ork_reg_get_device_count(ggml_backend_reg_t reg) { return 1; GGML_UNUSED(reg); }
static ggml_backend_dev_t ggml_backend_ork_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    static ggml_backend_device dev = { /* .iface = */ ggml_backend_ork_device_i, /* .reg = */ reg, /* .context = */ nullptr };
    return &dev;
    GGML_UNUSED(index);
}
static const struct ggml_backend_reg_i ggml_backend_ork_reg_i = {
    /* .get_name         = */ ggml_backend_ork_reg_get_name,
    /* .get_device_count = */ ggml_backend_ork_reg_get_device_count,
    /* .get_device       = */ ggml_backend_ork_reg_get_device,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_ork_reg(void) {
    static struct ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_ork_reg_i,
        /* .context     = */ NULL,
    };
    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_ork_reg)
