// ork-driver NPU matmul backend for ggml (Rockchip RK35xx).
//
// Modeled on the BLAS backend: a mul-mat-only accelerator that offloads GGML_OP_MUL_MAT to the
// Rockchip NPU via ork-driver and leaves all other ops to the CPU backend. Uses the int8 (W8A8)
// path: weights are dequantized then per-channel int8-quantized and packed once (cached, NPU-
// resident); activations are per-row int8-quantized each call; the NPU computes the int32 product
// which is dequantized (aScale[m]*bScale[n]) into the fp32 dst. ~1% vs fp32 on real weights, half
// the weight bytes of fp16. (int4/W4A4 + per-group scales is the next step down.)
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
//                        ork_mm_repack_i8 (no IOMMU churn). Only relevant when ORK_MOE_NPU is on.
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
//   ORK_FUSE=1           EXPERIMENTAL. QKV / gate-up fusion (int8). Measured neutral; off.
//   ORK_QUANT=4          EXPERIMENTAL. int4 W4A4 instead of int8 (incoherent; research only).
//   ORK_HADAMARD=1       EXPERIMENTAL. Hadamard-rotated int4 path (with ORK_QUANT=4).
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
//                        length == packer K) and pack via ork_mm_pack_i4a8_im (imatrix-weighted
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

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

extern "C" {
#include "ork_npu.h"
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
// First run with ORK_PERSIST=<path> writes it (one slow pass); later runs mmap it and load the bytes
// straight into DMA — no dequant/quant/tile. Each weight's (K,N,dtype) is re-checked on load AND the
// footer carries ork_pack_format_version() (ork-driver's MAJOR ver): a tile-layout / quant change bumps
// that major, so an incompatible file is rejected wholesale at startup and regenerated (the read path
// falls through to write mode). (K,N,dtype) alone can't catch a tiling change — same-(K,N) blobs from
// an incompatible major have identical size — which is exactly what the token guards.
//
// Per-tensor tier is carried in `dtype` (the field is back-compatible — v1 files only ever wrote dtype==1):
//   dtype == ORKPACK_DT_I8 (1): blob = ork_w_dump bytes (tiled int8), followed by bscale_n bscale floats.
//   dtype == ORKPACK_DT_I4 (4): blob = ork_w_dump_i4a8 bytes (self-describing 'O4N1': K,N,quant_kind +
//                               bscale[N] + nibble store). bscale lives INSIDE the blob → bscale_n==0,
//                               bscale_off unused. Loaded via ork_mm_load_i4a8, runs via ork_mm_run_i8.
// The struct layout is unchanged from v1, so v1 (all-int8) files load unmodified; VERSION bumps to 2 to
// mark files that may contain int4 entries (both versions are accepted on read).
#define ORKPACK_MAGIC   "ORKPK01"
#define ORKPACK_VERSION 3u   // v3 adds ork_fmt (ork-driver pack-compat token = its MAJOR ver) to the footer
#define ORKPACK_DT_I8   1u
#define ORKPACK_DT_I4   4u
struct orkpack_entry  { uint32_t K, N, dtype, bscale_n; uint64_t blob_off, blob_size, bscale_off; };
// ork_fmt = ork_pack_format_version() at write time. A tile-layout / quant change bumps ork-driver's
// MAJOR version, so a stored ork_fmt != this build's => the tiled bytes are incompatible; the file is
// rejected on read and regenerated (the read path falls through to write mode). magic stays last so it
// remains the final 8 bytes of the file regardless of footer growth.
struct orkpack_footer { uint64_t index_off; uint32_t n_entries; uint32_t version; uint32_t ork_fmt; char magic[8]; };

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
};

// One reusable slot in the MoE expert pool: a packed weight whose DMA buffer is reused (repack-in-place)
// across different experts of the SAME shape, so the NPU IOMMU isn't churned/fragmented by alloc+free.
struct ork_moe_slot {
    ork_w * w = nullptr;
    std::vector<float> bscale;
    const void * key = nullptr;       // host ptr of the expert currently packed here (nullptr = empty)
};

struct ggml_backend_ork_context {
    ork_npu * npu = nullptr;
    int qbits = 8;              // 8 = W8A8 (default), 4 = W4A4 (ORK_QUANT=4)
    int hadamard = 0;          // ORK_HADAMARD=1 (with ORK_QUANT=4): per-channel int4 + block-Hadamard rotation
    int no_reuse = 0;          // ORK_NOREUSE=1: disable activation-quant reuse (A/B benchmark)
    int no_cache = 0;          // ORK_NOCACHE=1: re-pack the weight every matmul (A/B benchmark)
    bool hybrid = false;       // use hybrid loading (FFN 4-bit, Attn 8-bit)
    std::vector<float>    f32;   // dequantized src0 plane [N*K] (cache-miss scratch)
    std::vector<int8_t>   bi;    // weights quantized int8 B[K*N] (cache-miss scratch)
    std::vector<int8_t>   ai;    // activations quantized int8 A[M*K]
    std::vector<float>    as;    // per-row activation scale [M]
    std::vector<int32_t>  ci;    // int32 matmul result [M*N] before dequant
    std::vector<float>    arot;  // rotated activation row [K] scratch (Hadamard int4 path)
    // model weights are constant during inference, so pack+quantize each once (NPU-resident) and
    // reuse, keyed by the weight plane's host pointer. The transformer pattern ork-driver is for.
    std::unordered_map<const void *, ork_weight> wcache;
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
        ork_w * wg_f16 = nullptr;                      // gate weight packed fp16 as -S*Wg (negate: fp16 index spreads on neg acc)
        std::vector<int16_t> lut_f16;                  // universal fp16 silu LUT (calibrated once per layer's gate range)
        double f16_out = 0;                            // dequant: silu(gate) = C_out * f16_out
        // ORK_FFN_SILU_CPU_GMAX: per-layer precision policy. High-|gate| layers lose the most to int8 fused
        // silu (fixed-range LUT); when this layer's gmax exceeds the threshold, use the EXACT path instead —
        // per-channel gate matmul on-NPU + fp32 silu on CPU (baseline quality, ~2.3x the fused-silu gate cost).
        bool silu_cpu = false;
        double gmax = 0;                               // this layer's calibrated max|gate| (the sensitivity signal)
    };
    std::unordered_map<const void *, ork_ffn_smooth>   ffncache;       // gate ptr -> smoothed layer state
    uint64_t wcache_tick  = 0;   // monotonic clock for LRU last_use
    // STREAM-POOL tier (ORK_STREAM_POOL=1): RAM-resident inflated-int8 cache w/ cheap map/unmap.
    ork_stream_pool * spool = nullptr;     // created at init when enabled (NULL => fall back to plain wcache)
    size_t   spool_ram_bytes = 0;          // RAM bytes held across all stream entries (RAM-LRU budget)
    long     spool_remaps = 0, spool_ram_evicts = 0, spool_iova_unmaps = 0;  // diagnostics
    // .orkpack persist (ORK_PERSIST=<path>): 0 off, 1 read (mmap'd), 2 write (building a .tmp)
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
    double s_pack    = 0;   // pack-miss: ork_mm_pack_i8 (tile into IOVA dma-buf)
    double s_load    = 0;   // .orkpack hit: ork_mm_load_i8 (Bf full-K rebuild)
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
    size_t domain_bytes[16] = {0};   // resident NPU bytes placed in each domain (report)
    long   mem_create_runtime = 0;   // # of weight packs/loads AFTER the load phase (must stay ~0 = no churn)
    int    load_phase = 1;           // 1 during initial residence fill; cleared at first decode/steady state
    int    domain_cursor = 0;        // byte-balanced fill: current domain being filled (advances as domains near the IOVA cap)
    int    domain_last_layer = -1;   // last blk.N layer index seen by ork_weight_domain (advance domains only at layer boundaries)
};
static ggml_backend_ork_context * g_ork_ctx = nullptr;
static bool g_ork_hybrid_loading = false;
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

// Evict least-recently-used weights (reclaiming IOVA via ork_mm_free) until `need` more bytes fit under
// the budget. Only per-tile-owned weights (int8 / per-channel int4) actually return IOVA; the current
// op's weight is never in the cache yet, so it is never evicted.
static void ork_wcache_evict(ggml_backend_ork_context * ctx, size_t need) {
    // Convert mode (building .orkpack): keep ~0 resident — pack→dump→free each weight (evicted by the next
    // pack). This makes conversion fit ANY model size (≤1 weight in the 4 GiB window) and avoids thrash.
    const size_t budget = ctx->persist_mode == 2 ? 0 : ork_wcache_budget();
    while (ctx->wcache_bytes + need > budget && !ctx->wcache.empty()) {
        auto lru = ctx->wcache.begin();
        for (auto it = ctx->wcache.begin(); it != ctx->wcache.end(); ++it)
            if (it->second.last_use < lru->second.last_use) lru = it;
        ork_mm_free(ctx->npu, lru->second.w);
        ctx->wcache_bytes -= lru->second.bytes;
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

// Open ORK_PERSIST: if the file exists and validates, mmap it for READ (load weights by name); otherwise
// open a <path>.tmp for WRITE (this run packs the model and dumps it, then finalize renames it in).
static void ork_persist_init(ggml_backend_ork_context * ctx) {
    const char * p = getenv("ORK_PERSIST");
    if (!p || !*p) return;
    int fd = open(p, O_RDONLY);
    if (fd >= 0) {
        off_t sz = lseek(fd, 0, SEEK_END);
        if (sz > (off_t) sizeof(orkpack_footer)) {
            void * m = mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);
            if (m != MAP_FAILED) {
                orkpack_footer f; memcpy(&f, (char *) m + sz - sizeof f, sizeof f);
                // Reject (=> regenerate below) if the ork-driver pack format is incompatible: an older
                // footer schema (< v3, no ork_fmt) or a different pack-compat token (a tiling/quant
                // change bumps ork-driver's MAJOR version). Same-(K,N) blobs from an incompatible major
                // are the SAME size, so this token is the only thing that catches them.
                bool magic_ok = memcmp(f.magic, ORKPACK_MAGIC, 8) == 0;
                if (magic_ok && f.version != ORKPACK_VERSION)          // older footer schema (pre-v3, no token)
                    fprintf(stderr, "[ORK PERSIST] %s predates the pack-compat token (footer < v%u) — regenerating\n", p, ORKPACK_VERSION);
                else if (magic_ok && f.ork_fmt != ork_pack_format_version())   // v3 file, but tiling/quant major differs
                    fprintf(stderr, "[ORK PERSIST] %s is stale (pack-compat token %u != this build's %u) — regenerating\n",
                            p, f.ork_fmt, ork_pack_format_version());
                if (memcmp(f.magic, ORKPACK_MAGIC, 8) == 0 && f.version == ORKPACK_VERSION &&
                    f.ork_fmt == ork_pack_format_version() && f.index_off < (uint64_t) sz) {
                    const char * idx = (const char *) m + f.index_off;
                    for (uint32_t i = 0; i < f.n_entries; i++) {
                        uint32_t nl; memcpy(&nl, idx, 4); idx += 4;
                        std::string name(idx, nl); idx += nl;
                        orkpack_entry e; memcpy(&e, idx, sizeof e); idx += sizeof e;
                        ctx->persist_idx.emplace(std::move(name), e);
                    }
                    ctx->persist_map = m; ctx->persist_map_sz = sz; ctx->persist_mode = 1; close(fd);
                    if (getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK PERSIST] read %s (%zu weights) — loading from disk, no re-conversion\n", p, ctx->persist_idx.size());
                    return;
                }
                munmap(m, sz);
            }
        }
        close(fd);
    }
    ctx->persist_final = p; ctx->persist_tmp = std::string(p) + ".tmp";
    ctx->persist_out = fopen(ctx->persist_tmp.c_str(), "wb");
    if (ctx->persist_out) { ctx->persist_mode = 2;
        if (getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK PERSIST] building %s (one-time conversion this run)\n", ctx->persist_tmp.c_str()); }
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
// vector aligns directly with ork_mm_pack_i4a8_im's per-input-channel importance contract.
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
    static int init = 0, i4_ffn = 0, from_src = 1; static long i4_above_bytes = -1;
    if (!init) {
        init = 1;
        const char * a = getenv("ORK_ORKPACK_I4_ABOVE_MB");
        if (a && *a) i4_above_bytes = atoll(a) * 1024 * 1024;
        i4_ffn = getenv("ORK_ORKPACK_I4_FFN") ? 1 : 0;
        const char * fs = getenv("ORK_ORKPACK_TIER_FROM_SRC");   // default ON; "0" disables
        if (fs && fs[0] == '0' && fs[1] == '\0') from_src = 0;
    }
    if ((K % 32) != 0 || (N % 32) != 0) return 8;          // int4 shape constraint → int8 regardless

    // (A0) external tier map (ORK_ORKPACK_TIERMAP) — wins over source-type so an fp16 source can
    // inherit a Q4_K_M's allocation by name. A mapped int4 still respects the shape constraint above.
    int tm = ork_tiermap_lookup(name);
    if (tm == 4 || tm == 8) return tm;

    bool want_i4 = false;
    // (A) source-type-driven default
    if (from_src) {
        double bits = ork_src_type_bits(src_type);
        if (bits >= 0.0 && bits < 5.0) want_i4 = true;     // low-bit source → int4 tier; unknown/high-bit → int8
    }
    // (B) explicit env overrides (force int4)
    if (i4_above_bytes >= 0 && (long) K * N >= i4_above_bytes) want_i4 = true;
    if (i4_ffn && name && (strstr(name, "ffn_") || strstr(name, "exps") ||
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
// ORK_ORKPACK_CPU: force the .orkpack tiling entirely onto the CPU (ork_w_dump_i8_cpu) — no bcreate/NPU
// pack at all, so the big cores (NEON dequant/quant + tile) do all the work and the NPU stays free.
// Turns the whole persist route CPU-only regardless of NPU idle/busy. int4-tier weights fall back to the
// int8 CPU dump (the compact int4 form needs the NPU packer). Off = the idle-gated hybrid.
static bool ork_orkpack_cpu_only() { static const int v = getenv("ORK_ORKPACK_CPU") != nullptr; return v; }

// bi_i8 (optional): the raw int8 weights [K*N, k-major]. When the int8 tier is chosen and no NPU-packed
// ow.w is supplied (the hybrid CPU route — NPU busy, or persistence-only weights that never went
// resident), the blob is tiled on the CPU straight from bi_i8 (ork_w_dump_i8_cpu, byte-identical to the
// NPU pack+dump) — no bcreate/IOVA. The int4 tier's NPU pack (ork_mm_pack_i4a8) is likewise gated on the
// NPU being idle; if it's busy the weight falls through to the int8 CPU dump (correct, just less compact).
static void ork_persist_write(ggml_backend_ork_context * ctx, const char * name, int K, int N,
                              const ork_weight & ow, const float * f32_plane, enum ggml_type src_type,
                              const int8_t * bi_i8 = nullptr) {
    if (ctx->persist_mode != 2 || !ctx->persist_out) return;
    if (!ctx->persist_dumped.insert(name).second) return;   // already dumped — a convert-decode re-pack, don't duplicate

    int tier = f32_plane ? ork_orkpack_tier(name, K, N, src_type) : 8;   // no f32 plane available → int8 only
    if (getenv("ORK_VERBOSE"))
        fprintf(stderr, "[ORK PERSIST] tier %s K=%d N=%d src=%s -> int%d\n",
                name, K, N, ggml_type_name(src_type), tier);
    if (tier == 4 && !ork_orkpack_cpu_only() && !ork_npu_busy(ctx->npu)) {   // int4 uses the NPU packer — only when idle + not CPU-forced
        std::vector<float> bscale_tmp(N);   // pack_i4a8 always writes bscale_out (no NULL check); the dump
        const float * im = ork_imatrix_lookup(name, K);   // per-input-channel importance, length K (or NULL)
        if (im && getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK PERSIST] imatrix %s (K=%d)\n", name, K);
        ork_w * w4 = ork_mm_pack_i4a8_im(ctx->npu, K, N, f32_plane, im, bscale_tmp.data());   // im=NULL → plain absmax (identical to pack_i4a8)
        if (w4) {
            size_t tb = ork_w_dump_i4a8(w4, nullptr, 0);
            if (tb) {
                std::vector<char> tmp(tb);
                ork_w_dump_i4a8(w4, tmp.data(), tb);
                orkpack_entry e; e.K = K; e.N = N; e.dtype = ORKPACK_DT_I4; e.bscale_n = 0;
                e.blob_off = ctx->persist_off; e.blob_size = tb; e.bscale_off = 0;
                fwrite(tmp.data(), 1, tb, ctx->persist_out); ctx->persist_off += tb;
                ctx->persist_built.emplace_back(std::string(name), e);
                ork_mm_free(ctx->npu, w4);
                if (getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK PERSIST] int4 %s K=%d N=%d (%zu B)\n", name, K, N, tb);
                return;
            }
            ork_mm_free(ctx->npu, w4);
        }
        // int4 pack/dump failed → fall through to int8 (never persist a broken entry)
    }
    // int8 tier: tile on the CPU from the raw int8 weights when forced CPU-only or when there's no
    // NPU-packed weight (hybrid CPU route — no NPU/IOVA); otherwise dump the NPU-packed tiles. Both
    // produce byte-identical blobs.
    const bool cpu_dump = (ork_orkpack_cpu_only() && bi_i8) || !ow.w;
    size_t tb = cpu_dump ? ork_w_dump_i8_cpu(ctx->npu, K, N, bi_i8, nullptr, 0)
                         : ork_w_dump(ow.w, nullptr, 0);
    std::vector<char> tmp(tb);
    if (cpu_dump) ork_w_dump_i8_cpu(ctx->npu, K, N, bi_i8, tmp.data(), tb);
    else          ork_w_dump(ow.w, tmp.data(), tb);
    orkpack_entry e; e.K = K; e.N = N; e.dtype = ORKPACK_DT_I8; e.bscale_n = (uint32_t) ow.bscale.size();
    e.blob_off = ctx->persist_off; e.blob_size = tb;
    fwrite(tmp.data(), 1, tb, ctx->persist_out); ctx->persist_off += tb;
    e.bscale_off = ctx->persist_off;
    fwrite(ow.bscale.data(), sizeof(float), ow.bscale.size(), ctx->persist_out);
    ctx->persist_off += ow.bscale.size() * sizeof(float);
    ctx->persist_built.emplace_back(std::string(name), e);
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
    int nthr = (int) sysconf(_SC_NPROCESSORS_ONLN); if (nthr < 1) nthr = 1;
    if (nthr > N) nthr = (int) N;
    auto worker = [&](int n0, int n1) {
        ork_unpin_current_thread();
        for (int n = n0; n < n1; n++) {
            float * frow = f32 + (size_t) n * K;
            if (type == GGML_TYPE_F32) memcpy(frow, x + n*nb01, (size_t) K*sizeof(float));
            else                        to_float(x + n*nb01, frow, K);
            float mx = 1e-9f;
            for (int k = 0; k < K; k++) { float v = fabsf(frow[k]); if (v > mx) mx = v; }
            float scale_val = mx / 127.0f; bscale[n] = scale_val;
            for (int k = 0; k < K; k++) {
                int q = (int) lrintf(frow[k] / scale_val);
                bi[(size_t) k*N + n] = (int8_t) (q > 127 ? 127 : q < -127 ? -127 : q);
            }
        }
    };
    if (nthr <= 1) { worker(0, (int) N); return; }
    std::vector<std::thread> th; int chunk = ((int) N + nthr - 1) / nthr;
    for (int t = 0; t < nthr; t++) { int a = t*chunk, b = std::min((int) N, a+chunk); if (a < b) th.emplace_back(worker, a, b); }
    for (auto & t : th) t.join();
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

    // COMPUTE→DMA PIPELINE. The per-expert work splits into a parallel CPU half (dequant+quant) and a
    // serial NPU/IO half (ork_mm_pack_i8's bcreate + IOMMU-map + bsync DMA, then dump). The serial half
    // is single-stream and leaves every core idle (~70% idle measured). So double-buffer: while THIS
    // thread runs the serial NPU pack/dump of expert i, a helper thread dequant+quants expert i+1 (all
    // cores) into the other buffer. Producer touches no NPU/ctx state → safe alongside the consumer.
    // Consumed strictly in `todo` order, so the .orkpack is bit-identical to the serial version.
    struct Stage { std::vector<float> f32; std::vector<int8_t> bi; std::vector<float> bscale; };
    Stage buf[2];
    for (auto & s : buf) { s.f32.resize((size_t) N * K); s.bi.resize((size_t) K * N); s.bscale.resize(N); }

    auto consume = [&](Stage & s, int e) {
        const std::string key = ork_expert_key(src0->name, e);
        ork_weight ow; ow.bscale = s.bscale;
        // HYBRID gate: experts are persistence-only (not used in the convert forward pass), so tile
        // them on the NPU packer ONLY when the device is idle; while it's serving, tile on the CPU
        // (ork_persist_write's int8 CPU path from s.bi) — no bcreate, no contention with inference.
        if (!ork_orkpack_cpu_only() && !ork_npu_busy(ctx->npu)) ow.w = ork_mm_pack_i8(ctx->npu, K, N, s.bi.data());
        ork_persist_write(ctx, key.c_str(), K, N, ow, s.f32.data(), type, s.bi.data());
        if (ow.w) ork_mm_free(ctx->npu, ow.w);   // resident ~0: free before the next slice
    };

    ork_expert_dequant_quant(src0, todo[0], K, N, type, to_float, buf[0].f32.data(), buf[0].bi.data(), buf[0].bscale.data());
    for (size_t i = 0; i < todo.size(); i++) {
        std::thread prod;
        if (i + 1 < todo.size()) {
            Stage & ns = buf[(i + 1) & 1];
            prod = std::thread(ork_expert_dequant_quant, src0, todo[i + 1], K, N, type, to_float,
                               ns.f32.data(), ns.bi.data(), ns.bscale.data());
        }
        consume(buf[i & 1], todo[i]);
        if (prod.joinable()) prod.join();
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
    memcpy(f.magic, ORKPACK_MAGIC, 8);
    fwrite(&f, sizeof f, 1, ctx->persist_out);
    fflush(ctx->persist_out); fclose(ctx->persist_out); ctx->persist_out = nullptr;
    rename(ctx->persist_tmp.c_str(), ctx->persist_final.c_str());
    if (getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK PERSIST] finalized %s (%u weights)\n", ctx->persist_final.c_str(), f.n_entries);
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
    ork_stream_entry * se = ork_stream_pool_add_i8(ctx->spool, K, N, blob.data(), need);
    if (!se) return false;    // fall back to ow.w (still resident & counted)
    // free the temp IOVA weight (it was counted in wcache_bytes by the caller); the RAM entry replaces it
    ctx->wcache_bytes -= ow.bytes;
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
                      bool allow_evict) {
    // allow_evict: the non-chain path uses one weight at a time, so it may stream-evict the LRU to
    // free IOVA. The chain path needs ALL `count` weights co-resident at submit, so it passes false
    // (matching the original chain pack, which never evicted) — otherwise packing weight i frees the
    // already-packed weight i-1 that the chain still references → use-after-free at submit.
    const double _r0 = ctx->profile ? ork_now_us_e() : 0;
    const char * x = (const char *) src0->data;
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
        auto pit = ctx->persist_idx.find(src0->name);
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
                if (e.dtype == ORKPACK_DT_I4) {
                    if (!no_import) ow.w = ork_mm_load_i4a8_import(ctx->npu, K, N, blob, e.blob_size);   // ZERO-COPY: map .orkpack page-cache pages into IOVA
                    if (!ow.w) ow.w = ork_mm_load_i4a8(ctx->npu, K, N, blob, e.blob_size);  // copy (import off / unavailable)
                    if (ow.w) { const float * bs = ork_w_bscale(ow.w); if (bs) ow.bscale.assign(bs, bs + N); }
                } else {
                    const double _l0 = ctx->profile ? ork_now_us_e() : 0;
                    if (!no_import) ow.w = ork_mm_load_i8_import(ctx->npu, K, N, blob, e.blob_size);     // ZERO-COPY: map .orkpack page-cache pages into IOVA
                    if (!ow.w) ow.w = ork_mm_load_i8(ctx->npu, K, N, blob, e.blob_size); // copy (import off / unavailable)
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
                if (ctx->n_domains > 1 && _dom < 16) ctx->domain_bytes[_dom] += it->second.bytes;
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
    ow.w = ork_mm_pack_i8(ctx->npu, K, N, bi);
    while (!ow.w && (_dom = ork_domain_advance(ctx)) >= 0)  // domain's IOVA full -> next domain, retry
        ow.w = ork_mm_pack_i8(ctx->npu, K, N, bi);
    if (ctx->profile) ctx->s_pack += ork_now_us_e() - _p0;
    if (!ow.w) return ctx->wcache.end();
    it = ctx->wcache.emplace(x, std::move(ow)).first;
    it->second.bytes = ork_w_bytes(it->second.w);
    ctx->wcache_bytes += it->second.bytes;
    if (ctx->n_domains > 1 && _dom < 16) ctx->domain_bytes[_dom] += it->second.bytes;
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
static inline double ork_now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec*1e6 + t.tv_nsec*1e-3; }

// dst = src0 x src1 :  src0 [K=ne00, N=ne01], src1 [K=ne10=ne00, M=ne11], dst [N, M] (row-major [M][N])
// Central dispatcher: pick the right ork-driver entry for a set of independent int8 matmul tasks
// (a single weight, or a chain of matmuls that share src1 — Q/K/V, gate/up — hence data-independent).
//   • 1 task              → ork_mm_run_i8 (does its own K-split / N-tiling / M-scheduler multicore)
//   • N, SAME domain      → ork_mm_run_stream_i8 — round-robin the independent matmuls across the 3 NPU
//                           cores concurrently (the NPU has ONE active IOMMU domain at a time, so RR is
//                           only valid within a domain; layer-aligned placement keeps a chain co-domain)
//   • N, cross-domain     → ork_mm_run_chain_i8 (single-core chain; cross-domain concurrency unsupported)
// On any error, falls back to sequential per-task run_i8. Returns true on success.
static bool ork_dispatch_i8(ggml_backend_ork_context * ctx, std::vector<ork_mm_task_i8> & tasks) {
    if (tasks.empty()) return true;
    int rc;
    if (tasks.size() == 1) {
        rc = ork_mm_run_i8(ctx->npu, tasks[0].w, tasks[0].M, tasks[0].A, tasks[0].C) ? -1 : 0;
    } else {
        bool same_dom = true; const int d0 = ork_w_domain(tasks[0].w);
        for (size_t t = 1; t < tasks.size(); t++) if (ork_w_domain(tasks[t].w) != d0) { same_dom = false; break; }
        rc = same_dom ? ork_mm_run_stream_i8(ctx->npu, tasks.size(), tasks.data())
                      : ork_mm_run_chain_i8 (ctx->npu, tasks.size(), tasks.data());
        if (getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK] dispatch: %zu tasks, same_dom=%d -> %s rc=%d\n",
                                           tasks.size(), (int)same_dom, same_dom ? "run_stream" : "run_chain", rc);
    }
    if (rc != 0) {   // fallback: sequential single-task
        for (size_t t = 0; t < tasks.size(); t++)
            if (ork_mm_run_i8(ctx->npu, tasks[t].w, tasks[t].M, tasks[t].A, tasks[t].C)) return false;
    }
    return true;
}

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
            if (!ork_dispatch_i8(ctx, tasks)) return false;   // 1→run_i8 · N same-domain→run_stream (RR) · N cross→run_chain
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
                ow.w = ork_mm_pack_i4_grouped(ctx->npu, K, N, bi, G);
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
            if (ork_mm_run_i4_grouped(ctx->npu, ow.w, M_padded, ai, as, ow.bscale.data(), d_ptr)) return false;
            if (M != M_padded) {
                memcpy(d, d_ptr, (size_t) M * N * sizeof(float));
            }
        }
    }
    return true;
}


// int4 (W4A4) with PER-CHANNEL scales + a block-Hadamard rotation (ORK_HADAMARD=1). Weights are
// rotated (R·B) and per-channel int4-quantized once at load (cached); activations are rotated (A·R)
// and per-row int4-quantized each matmul; the rotation cancels in fp32 (A·B = (A·R)·(R·B)) but lets
// the coarse per-channel int4 quant stay accurate. Per-channel = full-K SINGLE submit (ork_mm_run_i4),
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

            auto it = ctx->wcache.find(x);
            if (it == ctx->wcache.end()) {
                if (type == GGML_TYPE_F32) {
                    for (int64_t n = 0; n < N; n++) memcpy(f32 + n*K, x + n*nb01, (size_t) K*sizeof(float));
                } else {
                    for (int64_t n = 0; n < N; n++) to_float((const char *) x + n*nb01, f32 + n*K, K);
                }
                ork_weight ow; ow.gsize = 0; ow.bscale.resize((size_t) N);   // per-channel scale ws[n]
                for (int n = 0; n < N; n++) {
                    float * col = f32 + (size_t) n*K;
                    for (int off = 0; off < K; off += b) {
                        ork_fwht_norm(col + off, b);                        // rotate weight column R·B
                    }
                    float mx = 1e-9f;
                    for (int k = 0; k < K; k++) { float v = fabsf(col[k]); if (v > mx) mx = v; }
                    float s = mx / 7.0f; ow.bscale[n] = s;
                    for (int k = 0; k < K; k++) {
                        int q = (int) lrintf(col[k] / s);
                        bi[(size_t) k*N + n] = (int8_t) (q > 7 ? 7 : q < -8 ? -8 : q);
                    }
                }
                ow.w = ork_mm_pack_i4(ctx->npu, K, N, bi);
                if (!ow.w) return false;
                it = ctx->wcache.emplace(x, std::move(ow)).first;
            }
            const ork_weight & ow = it->second;

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
            if (ork_mm_run_i4(ctx->npu, task.w, task.M, task.A, task.C)) return false;    // full-K single submit, int32 C
            #pragma omp parallel for if (M >= 16)
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    d[(size_t) m*N + n] = (float) ci[(size_t) m*N + n] * as[m] * ow.bscale[n];
                }
            }
        }
    }
    return true;
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
    if (it == ctx->wcache.end()) {                       // build + pack the fused weight once
        ork_weight ow; ow.bscale.resize(Ntot);
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
                for (int k = 0; k < K; k++) {            // fused B[k][off+n] = src0_i[n][k]
                    int q = (int) lrintf(f32[(size_t) n*K + k] / s);
                    bi[(size_t) k*Ntot + off[i]+n] = (int8_t) (q > 127 ? 127 : q < -127 ? -127 : q);
                }
            }
        }
        ow.w = ork_mm_pack_i8(ctx->npu, K, Ntot, bi);
        if (!ow.w) return false;
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
    if (ork_mm_run_i8(ctx->npu, ow.w, M_padded, ai, ci)) return false;     // ONE submit for all ng matmuls
    const double t2 = ctx->profile ? ork_now_us() : 0;

    const float * bs = ow.bscale.data();                 // scatter+dequant into each dst
    for (int i = 0; i < ng; i++) {
        const int Ni = (int) g[i]->src[0]->ne[1]; const int o = off[i];
        float * dbase = (float *) g[i]->data;
        for (int m = 0; m < M; m++) {
            const float rs = as[m];
            const int32_t * cr = ci + (size_t) m*Ntot + o;
            float * dr = dbase + (size_t) m*Ni;
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
static void ggml_backend_ork_free(ggml_backend_t backend) {
    ggml_backend_ork_context * ctx = (ggml_backend_ork_context *) backend->context;
    ork_profile_dump(ctx);
    if(getenv("ORK_VERBOSE"))fprintf(stderr, "[ORK DEBUG] ggml_backend_ork_free called!\n");
    ork_persist_finalize(ctx);   // .orkpack: write index+footer, rename .tmp into place (write mode)
    if (ctx->persist_mode) fprintf(stderr, "[ORK PERSIST] this run: loaded %ld from disk, packed %ld\n", ctx->persist_hits, ctx->persist_misses);
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
    if (ctx->persist_map) munmap(ctx->persist_map, ctx->persist_map_sz);
    if (ctx->spool) ork_stream_pool_free(ctx->spool);   // frees all stream entries' RAM dma-bufs
    for (auto & kv : ctx->wcache) ork_w_free(kv.second.w);   // w is NULL for stream-pool entries (no-op)
    for (auto & p : ctx->moe_pools) for (auto & s : p.second) if (s.w) ork_w_free(s.w);   // MoE expert pool
    for (auto & tk : ctx->moe_hot) for (auto & es : tk.second) if (es.second.w) ork_w_free(es.second.w);   // hot-expert partition
    if (ctx->cpu_backend) ggml_backend_free(ctx->cpu_backend);   // PATH (b) cached CPU backend
    for (auto & kv : ctx->pathb_repack) { if (kv.second.buf) ggml_backend_buffer_free(kv.second.buf); if (kv.second.gctx) ggml_free(kv.second.gctx); }
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
            ow.w = ork_mm_pack_i4(ctx->npu, K, N, bi);
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
        ok = ork_mm_run_i4(ctx->npu, tasks[0].w, tasks[0].M, tasks[0].A, tasks[0].C) ? -1 : 0;
    } else {
        ok = ork_mm_run_chain_i4(ctx->npu, tasks.size(), tasks.data());
    }
    
    if (ok != 0) {
        // Fallback to sequential single-task run
        if(getenv("ORK_VERBOSE"))fprintf(stderr, "[ORK] i4 chain failed (%d), falling back to sequential\n", ok); fflush(stderr);
        for (size_t t = 0; t < tasks.size(); t++) {
            if (ork_mm_run_i4(ctx->npu, tasks[t].w, tasks[t].M, tasks[t].A, tasks[t].C)) {
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
            const double _aq0 = ctx->profile ? ork_now_us() : 0;   // LEVER3: time chain-path act-quant (SCALAR — lever2 NEON not applied here!)
            #pragma omp parallel for if (M_padded >= 16)
            for (int m = 0; m < M_padded; m++) {
                if (m < M) {
                    const float * yr = y + (size_t) m*K;
                    int8_t * amr = task_A + (size_t) m*K;
                    float mx = 1e-9f;
                    for (int k = 0; k < K; k++) { float v = fabsf(yr[k]); mx = v > mx ? v : mx; }
                    task_as[m] = mx / 127.0f;
                    const float inv = 127.0f / mx;
                    for (int k = 0; k < K; k++) {
                        float q = yr[k] * inv;
                        int qi = (int) (q + copysignf(0.5f, q));
                        amr[k] = (int8_t) (qi > 127 ? 127 : qi < -127 ? -127 : qi);
                    }
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
    
    if (!ork_dispatch_i8(ctx, tasks)) return false;   // 1→run_i8 · N same-domain→run_stream (RR) · N cross→run_chain

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



// Dequant one expert-weight output channel (row) -> dst[K] for ork_mm_pack_i8_dequant: fuses ggml's
// Q4_K->f32 with ork-driver's int8 quant+tile so the full f32[N][K] is never materialized (kills the
// DRAM round-trip — alloc + write + read-back of N*K floats — that was part of the MoE repack cost).
struct ork_moe_deq_ctx { const char * x; size_t nb01; ggml_to_float_t to_float; bool is_f32; };
static void ork_moe_deq_row(void * vctx, int n, float * dst, int K) {
    const ork_moe_deq_ctx * c = (const ork_moe_deq_ctx *) vctx;
    if (c->is_f32) memcpy(dst, c->x + (size_t) n * c->nb01, (size_t) K * sizeof(float));
    else           c->to_float(c->x + (size_t) n * c->nb01, dst, K);
}

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
    // One cold work-item: dot output rows [n0,n1) of expert e's weight against the quantized activation
    // for (token t, slot j). We split the N output rows into blocks so the thread pool fans across ROWS,
    // not just across experts — critical at decode, where there may be only a few cold experts but each
    // has N=1792..2048 independent output rows. (vec_dot writes dr[n] from the raw quant weight row.)
    struct cold_item { int e, t, j, n0, n1; size_t qoff; };
    auto run_cold_item = [&](const cold_item & ci) {
        const char * xw = (const char *) src0->data + (size_t) ci.e * src0->nb[2];
        const void * qa = qact.data() + ci.qoff;
        float * dr = (float *)(dbase + (size_t) ci.j * dst->nb[1] + (size_t) ci.t * dst->nb[2]);
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
            slot.w = (pe->dtype==ORKPACK_DT_I4) ? ork_mm_load_i4a8(ctx->npu,K,N,blob,pe->blob_size)
                                                : ork_mm_load_i8  (ctx->npu,K,N,blob,pe->blob_size);
            if (!slot.w) return (ggml_backend_ork_context::ork_hot_slot *) nullptr;
            if (pe->dtype==ORKPACK_DT_I4){ const float*b=ork_w_bscale(slot.w); if(b) memcpy(bsc.data(),b,N*sizeof(float)); }
            else memcpy(bsc.data(), (const char*)ctx->persist_map + pe->bscale_off, N*sizeof(float));
            ctx->persist_hits++;
        } else {
            if (ctx->persist_mode==1) ctx->persist_misses++;
            ork_moe_deq_ctx dq = { (const char*)x, (size_t) src0->nb[1], to_float, type==GGML_TYPE_F32 };
            slot.w = ork_mm_pack_i8_dequant(ctx->npu, K, N, ork_moe_deq_row, &dq, bsc.data());
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
                int rc = ork_mm_run_stream_i8(ctx->npu, Snpu, tasks.data());
                if (rc) { rc = 0; for (int x = 0; x < Snpu && rc == 0; x++) rc = ork_mm_run_i8(ctx->npu, tasks[x].w, tasks[x].M, tasks[x].A, tasks[x].C); }
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

    const size_t eff_cap = batched ? (size_t) -1 /*budget-limited only*/ : hot_K;
    std::vector<int> hot_e; std::vector<ggml_backend_ork_context::ork_hot_slot *> hot_s;
    for (auto & kv : buckets) {
        const int e = kv.first;
        const void * x = (const char *) src0->data + (size_t) e * src0->nb[2];
        bool resident = hotmap.count(x) && hotmap[x].w;
        // PATH B: conforming K always NPU-eligible; non-conforming K only when this expert has enough
        // routed rows (M_e) to amortize the submit (the prefill / batched-verify regime). allk forces all.
        const int M_e = (int) kv.second.size();
        const bool npu_ok = conforming_k || allk || (M_e >= down_minM);
        // admit to NPU if the shape conforms AND (resident, or the pool has headroom under eff_cap)
        ggml_backend_ork_context::ork_hot_slot * s =
            (npu_ok && (resident || hotmap.size() < eff_cap)) ? get_hot(e, eff_cap) : nullptr;
        if (s) { hot_e.push_back(e); hot_s.push_back(s); }
        else   { cpu_expert(e, kv.second); }   // non-conforming or budget/cap-full -> CPU (deferred; run threaded below)
    }

    // Pre-quantize all needed token activations single-threaded (qact grows; not thread-safe), then build
    // a flat list of (expert, token, slot, row-block) work-items and fan THOSE across the pool. Splitting
    // each expert's N output rows into ROW_BLK-sized blocks keeps every thread busy even when only one or
    // two experts went cold this step (the decode case) — the old per-expert fan-out left N-row parallelism
    // on the table and serialized a lone cold expert's 1792 vec_dot calls onto a single core.
    if (!cold.empty()) {
        const double cd0 = ctx->profile ? ork_now_us() : 0;
        const int n_cold = (int) cold.size();
        std::vector<cold_item> items;
        const int ROW_BLK = 256;
        for (auto & ce : cold) {
            const int e = ce.first;
            for (auto & pr : *ce.second) {
                const size_t qoff = quant_tok(pr.first, pr.second);   // populate qact (single-threaded)
                for (int n0 = 0; n0 < N; n0 += ROW_BLK)
                    items.push_back(cold_item{ e, pr.first, pr.second, n0, n0+ROW_BLK<N ? n0+ROW_BLK : N, qoff });
            }
        }
        const int n_items = (int) items.size();
        unsigned hw = std::thread::hardware_concurrency();
        int nthr = (int) (getenv("ORK_MOE_COLD_THREADS") ? atoi(getenv("ORK_MOE_COLD_THREADS")) : (hw ? hw/2 : 4));
        if (nthr < 1) nthr = 1; if (nthr > n_items) nthr = n_items;
        if (nthr <= 1) {
            for (auto & ci : items) run_cold_item(ci);
        } else {
            std::atomic<int> next(0);
            auto worker = [&]() { int i; while ((i = next.fetch_add(1)) < n_items) run_cold_item(items[i]); };
            std::vector<std::thread> th; th.reserve(nthr-1);
            for (int w = 0; w < nthr-1; w++) th.emplace_back(worker);
            worker();
            for (auto & t : th) t.join();
        }
        ctx->moe_cold_cpu += n_cold;
        if (ctx->profile) { ctx->moe_cold += ork_now_us() - cd0; ctx->moe_cold_calls += n_cold; }
    }

    if (ctx->profile && ctx->moe_calls < 4) fprintf(stderr, "[ork MoE-DIM] K=%d N=%d S=%d type=%d (chain-envelope K%%512==0&&K<=4096: %s)\n", K, N, (int)hot_e.size(), (int)type, (K%512==0&&K<=4096)?"YES":"no");
    const int S = (int) hot_e.size();
    if (S == 0) { if (ctx->profile) { ctx->t_run += ork_now_us() - t0; ctx->n_mm++; } return true; }

    // Pack the hot experts' routed rows into one chained submit (run_chain_i8; per-task run_i8 fallback
    // for the K=1792 down-proj that sits outside the chain envelope).
    size_t total_rows = 0; for (int e : hot_e) total_rows += buckets[e].size();
    std::vector<int8_t>  bigA((size_t) total_rows * K);
    std::vector<int32_t> bigC((size_t) total_rows * N);
    std::vector<float>   as_row(total_rows);
    std::vector<ork_mm_task_i8> tasks(S);
    std::vector<size_t> offs(S);
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
        tasks[x].w = hot_s[x]->w; tasks[x].M = cnt; tasks[x].A = Ae; tasks[x].C = bigC.data() + off * N;
        offs[x] = off; off += cnt;
    }
    // M2 change #2: dispatch the S independent hot experts via the ASYNC ROUND-ROBIN STREAM
    // (run_stream_i8, ~2.5x cross-core even when weights are resident) instead of the single-core
    // PC-chain. run_stream_i8 rejects any task whose K is outside the full-K envelope (K%512==0 &&
    // K<=4096) with rc=-3, so non-conforming shapes (e.g. LFM down-proj K=1792) fall through to the
    // per-task run_i8 path below — same correctness, just no cross-core dispatch for those.
    // ORK_MOE_STREAM=0 reverts to the chain path (A/B comparison).
    static const bool use_stream = !(getenv("ORK_MOE_STREAM") && atoi(getenv("ORK_MOE_STREAM")) == 0);
    const double ch0 = ctx->profile ? ork_now_us() : 0;
    int crc = use_stream ? ork_mm_run_stream_i8(ctx->npu, S, tasks.data())
                         : ork_mm_run_chain_i8 (ctx->npu, S, tasks.data());
    const double ch1 = ctx->profile ? ork_now_us() : 0;   // [VERIFY] split stream/chain vs fallback
    if (crc) { crc = 0; for (int x = 0; x < S && crc == 0; x++) crc = ork_mm_run_i8(ctx->npu, tasks[x].w, tasks[x].M, tasks[x].A, tasks[x].C);
               if (ctx->profile) { ctx->moe_fallback_t += ork_now_us() - ch1; ctx->moe_fallback_calls++; } }
    if (ctx->profile) { ctx->moe_chain += ork_now_us() - ch0; ctx->moe_calls++; ctx->moe_chain_S_sum += S; }
    if (crc) { if(getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK] mul_mat_id partition run FAIL rc=%d S=%d K=%d N=%d\n", crc, S, K, N); return false; }

    const double sc0 = ctx->profile ? ork_now_us() : 0;
    for (int x = 0; x < S; x++) {
        std::vector<std::pair<int,int>> & ent = buckets[hot_e[x]];
        const int cnt = (int) ent.size(); const size_t o = offs[x];
        const float * bs = hot_s[x]->bscale.data();
        for (int i = 0; i < cnt; i++) {
            const int t = ent[i].first, j = ent[i].second;
            const int32_t * cr = bigC.data() + (o + i) * N;
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
// computed on the NPU in fp16 via ork_npu_ewmul_f16. fp16 is bit-exact for this
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
    int rc = ork_npu_ewmul_f16(ctx->npu, ha.data(), hb.data(), M, N, ho.data(), &us);
    if (rc == 0) {
        for (int64_t i = 0; i < ne; i++) o[i] = (float) ho[i];
        return true;
    }
    // CPU fallback (fp32, exact) — keeps the graph correct if the NPU declined/failed.
    for (int64_t i = 0; i < ne; i++) o[i] = a[i] * b[i];
    return true;
}

// Residual ADD: dst = src0 + src1 (same-shape, contiguous, f32) via ork_npu_add_f16.
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
    int rc = ork_npu_add_f16(ctx->npu, ha.data(), hb.data(), M, N, ho.data(), &us);
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
    int rc = gelu ? ork_npu_gelu_i8(ctx->npu, qi.data(), M, N, in_scale, out_scale, qo.data(), &us)
                  : ork_npu_silu_i8(ctx->npu, qi.data(), M, N, in_scale, out_scale, qo.data(), &us);
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
// bi[k*N+n] layout ork_mm_pack_i8 expects. Simpler than ork_resolve_weight_i8 (no stream/orkpack/domain
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
    ork_w * w = ork_mm_pack_i8(ctx->npu, K, N, bi.data());
    while (!w && (dom = ork_domain_advance(ctx)) >= 0) w = ork_mm_pack_i8(ctx->npu, K, N, bi.data());
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
    int dom = ork_weight_domain(ctx, (size_t) K * N, layer); ork_npu_set_pack_domain(ctx->npu, dom);
    ork_w * w = ork_mm_pack_i8(ctx->npu, K, N, bi.data());
    while (!w && (dom = ork_domain_advance(ctx)) >= 0) w = ork_mm_pack_i8(ctx->npu, K, N, bi.data());
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
static bool ork_ffn_prep(ggml_backend_ork_context * ctx, ggml_backend_ork_context::ork_ffn_smooth & fc,
        const struct ggml_tensor * Wg, const struct ggml_tensor * Wu, const struct ggml_tensor * Wd,
        const float * xf, int M, int K, int Nff, int Kd) {
    // HYBRID chain: only the GATE is forced per-tensor (its fused SiLU has a scalar SDP output scale).
    // up + down run the STANDARD per-channel host-dequant path (near-baseline quality) — the RE finding was
    // that RKNN itself does matmul per-channel requant on the host, not in the SDP regcmd. So prep only needs
    // the gate: SmoothQuant the gate input (x outliers -> gate weight), pack the smoothed gate per-tensor,
    // calibrate static s_x + s_silu, build the fused-SiLU LUT.
    // PER-DOMAIN FUSION: the fused path uses a PER-TENSOR gate (fc.wg, packed below) and NEVER the per-channel
    // gate — but the per-channel gate was loaded during the normal load phase into the SAME domain, doubling
    // this layer's gate storage. Under multi-domain that overflows the domain (PRIME_FD_TO_HANDLE fails on a
    // later import -> partial down-proj -> multi-core submit reads unmapped IOVA -> errno 110/22 -> soft-reset;
    // the exact trace). Evict the per-channel gate now (reclaim its IOVA) BEFORE packing fc.wg, so the domain
    // holds ONE gate, not two. Safe: the fused handler resolves only up/down per-channel, never the gate again.
    // Skip stream-pool-backed entries (se != nullptr) — the spool manages their IOVA separately.
    if (ctx->n_domains > 1) {
        auto git = ctx->wcache.find(Wg->data);
        if (git != ctx->wcache.end() && !git->second.se && git->second.w) {
            ork_mm_free(ctx->npu, git->second.w);
            ctx->wcache_bytes -= git->second.bytes;
            ctx->wcache.erase(git);
        }
    }
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
    fc.wg = ork_pack_pt_f32(ctx, wgf.data(), K, Nff, fc.s.data(), &fc.sg, ork_layer_of(Wg->name));
    if (!fc.wg) { if (getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK FFN-CHAIN] prep FAIL %s: fc.wg pack (K=%d N=%d dom=%d)\n", Wg->name, K, Nff, ork_weight_domain(ctx,(size_t)K*Nff,ork_layer_of(Wg->name))); return false; }
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
    // ORK_FFN_SILU_CPU_GMAX: strategic per-layer precision — layers whose gate range exceeds the threshold use
    // exact CPU fp32 silu (per-channel gate); the rest keep the fast int8 fused silu. Tune the ratio via the
    // threshold (lower = more layers exact = higher quality, lower speed).
    if (const char * t = getenv("ORK_FFN_SILU_CPU_GMAX")) fc.silu_cpu = (gmax_gate > atof(t));
    // ORK_FFN_GATE_F16: precise fp16 gate — build the fp16 SiLU LUT for this layer's gate range and pack the
    // gate weight as -S*Wg (fp16). silu output = C*f16_out at fp16 precision (no int8 activation quant).
    if (getenv("ORK_FFN_GATE_F16")) {
        fc.lut_f16.resize(1030); double S = 0, R = 0, os = 0;
        if (ork_mm_build_f16_silu_lut(ctx->npu, gmax_gate, fc.lut_f16.data(), &S, &R, &os)) return false;
        fc.f16_out = os;
        std::vector<ork_f16> wgh((size_t) Nff * K);   // pack -S*Wg as fp16 in [K][N] layout (ork_mm_pack convention)
        for (int n = 0; n < Nff; n++) for (int k = 0; k < K; k++)
            wgh[(size_t) k*Nff + n] = (ork_f16) (-(double) S * wgf[(size_t) n*K + k]);
        int dom = ork_weight_domain(ctx, (size_t) K * Nff * 2, ork_layer_of(Wg->name)); ork_npu_set_pack_domain(ctx->npu, dom);
        fc.wg_f16 = ork_mm_pack(ctx->npu, K, Nff, wgh.data());
        while (!fc.wg_f16 && (dom = ork_domain_advance(ctx)) >= 0) fc.wg_f16 = ork_mm_pack(ctx->npu, K, Nff, wgh.data());
        if (!fc.wg_f16) return false;
        if (getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK GATE-F16] prep %s: gmax=%.2f S=%.2f out=%.3e\n", Wg->name, gmax_gate, S, os);
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
//   gate:  ork_mm_run_i8_silu (gate matmul + fused SiLU int8-out, per-layer LUT)  -> silu_i8[M,Nff]
//   up:    ork_mm_run_i8_out8 (up matmul int8-out)                                -> up_i8[M,Nff]
//   glu:   ork_npu_ewmul_i8   (silu_i8 * up_i8 / 128)                             -> glu_i8[M,Nff]
//   down:  ork_mm_run_i8      (down matmul, int8-in)                              -> down_i32[M,Kd]
// then dequant down -> dst fp32. PER-TENSOR scales throughout (the fused SiLU stage's scalar R can't carry
// per-row/per-channel). CPU fp32 fallback (recompute the whole inner) on ANY error, so output is always valid.
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

    const float * xf = (const float *) x->data;
    // one-time per-layer SmoothQuant prep (smoothing + smoothed-weight pack + static-scale calibration + LUT)
    auto & fc = ctx->ffncache[Wg->data];
    if (!fc.ready) { if (!ork_ffn_prep(ctx, fc, Wg, Wu, Wd, xf, M, K, Nff, Kd)) return false; }

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
    // resolve per-channel up/down weights (standard wcache path)
    auto itu = ork_resolve_weight_i8(ctx, Wu, K, Nff, Wu->nb[1], Wu->type, ggml_get_type_traits(Wu->type)->to_float, true);
    if (itu == ctx->wcache.end()) return false;
    const ork_weight & owu = itu->second; const float * bsu = owu.bscale.data();
    auto itd = ork_resolve_weight_i8(ctx, Wd, Nff, Kd, Wd->nb[1], Wd->type, ggml_get_type_traits(Wd->type)->to_float, true);
    if (itd == ctx->wcache.end()) return false;
    const ork_weight & owd = itd->second; const float * bsd = owd.bscale.data();
    // PER-DOMAIN FUSION SAFETY (multi-domain): the three matmul weights (fused gate fc.wg, up owu.w, down
    // owd.w) MUST reside in the SAME IOMMU domain — a fused submit binds buffers in one active domain, so a
    // cross-domain weight would fault/soft-reset the NPU. Layer-aligned assignment (domain=layer/domain_layers)
    // normally guarantees co-residence; verify at RUNTIME and fall back to the CPU recompute if it ever differs.
    if (ctx->n_domains > 1) {
        int dg = ork_w_domain(fc.wg), du = ork_w_domain(owu.w), dd = ork_w_domain(owd.w);
        if (dg != du || du != dd) {
            if (getenv("ORK_VERBOSE"))
                fprintf(stderr, "[ORK FFN-CHAIN] %s: cross-domain weights (gate=%d up=%d down=%d) — per-node fallback\n",
                        Wg->name, dg, du, dd);
            return false;
        }
    }

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
    // Phase 1: up (per-channel, per-row) -> up_f fp32
    for (int m0 = 0; m0 < M && ok; m0 += MT) { int mc = std::min(MT, M - m0);
        if (ork_mm_run_i8(ctx->npu, owu.w, mc, xr_i8.data() + (size_t) m0*K, acc_i32.data())) { ok = false; break; }
        for (int r = 0; r < mc; r++) { const int32_t * ur = acc_i32.data() + (size_t) r*Nff;
            float * uo = up_f.data() + (size_t)(m0+r)*Nff; const float rs = as_x[m0+r];
            for (int n = 0; n < Nff; n++) uo[n] = rs * bsu[n] * (float) ur[n]; } }
    // Phase 2: silu_f fp32
    if (ok && fc.silu_cpu && ablate < 0) {
        // EXACT gate (strategic high-gmax layer): per-channel gate matmul on-NPU + fp32 silu on CPU. Same recipe
        // as the up projection (per-channel weight, per-row x), so quality matches baseline; only the fused-silu
        // stage is skipped for this layer. Reuses xr_i8 + as_x (per-row original x, already computed above).
        auto itg = ork_resolve_weight_i8(ctx, Wg, K, Nff, Wg->nb[1], Wg->type, ggml_get_type_traits(Wg->type)->to_float, true);
        if (itg == ctx->wcache.end()) { ok = false; }
        else { const ork_weight & owg = itg->second; const float * bsg = owg.bscale.data();
            for (int m0 = 0; m0 < M && ok; m0 += MT) { int mc = std::min(MT, M - m0);
                if (ork_mm_run_i8(ctx->npu, owg.w, mc, xr_i8.data() + (size_t) m0*K, acc_i32.data())) { ok = false; break; }
                for (int r = 0; r < mc; r++) { const int32_t * ar = acc_i32.data() + (size_t) r*Nff;
                    float * so = silu_f.data() + (size_t)(m0+r)*Nff; const float rs = as_x[m0+r];
                    for (int n = 0; n < Nff; n++) { float g = (float) ar[n] * rs * bsg[n]; so[n] = g / (1.0f + expf(-g)); } } } }
    } else if (ok && fc.wg_f16 && ablate < 0) {
        // ORK_FFN_GATE_F16: precise fp16 gate matmul + fused fp16 SiLU. x -> fp16 (cast), gate weight is -S*Wg
        // (baked in prep), so acc = -S*gate spreads the fp16 LUT. silu(gate) = C_out * fc.f16_out (no int8 quant).
        std::vector<ork_f16> xh((size_t) M * K);
        for (size_t i = 0; i < (size_t) M*K; i++) xh[i] = (ork_f16) xf[i];
        for (int m0 = 0; m0 < M && ok; m0 += MT) { int mc = std::min(MT, M - m0);
            if (ork_mm_run_f16_silu(ctx->npu, fc.wg_f16, mc, xh.data() + (size_t) m0*K, silu_f.data() + (size_t) m0*Nff,
                                    0, 0xffffc000u, 0x56391100u, fc.lut_f16.data(), 1030)) { ok = false; break; } }
        for (size_t i = 0; i < (size_t) M*Nff; i++) silu_f[i] *= (float) fc.f16_out;
    } else if (ok && ablate < 0) {
        // normal fused path: silu on NPU -> int8 -> *s_silu
        std::vector<int8_t> silu_i8((size_t) M * Nff);
        for (int m0 = 0; m0 < M && ok; m0 += MT) { int mc = std::min(MT, M - m0);
            if (ork_mm_run_i8_silu(ctx->npu, fc.wg, mc, xi.data() + (size_t) m0*K, silu_i8.data() + (size_t) m0*Nff,
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
            if (ork_mm_run_i8(ctx->npu, gw, mc, gx + (size_t) m0*K, acc_i32.data())) { ok = false; break; }
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
            if (ork_mm_run_i8(ctx->npu, owd.w, mc, glr_i8.data() + (size_t) m0*Nff, down_i32.data() + (size_t) m0*Kd)) { ok = false; break; } }
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
// Quantize per-tensor to int8, silu(gate) via ork_npu_silu_i8, (x)up via ork_npu_ewmul_i8 (gain 1/128 so
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
    int r1 = ork_npu_silu_i8(ctx->npu, (const signed char*)gi.data(), M, N, s_gate, s_silu, (signed char*)si.data(), &us);
    int r2 = r1 ? -1 : ork_npu_ewmul_i8(ctx->npu, ui.data(), si.data(), M, N, 0x4000, 21, glu.data(), &us);
    if (r1 == 0 && r2 == 0) {
        const double s_glu = s_up * s_silu * 128.0;
        for (int64_t i = 0; i < ne; i++) o[i] = (float)(glu[i] * s_glu);
        return true;
    }
    for (int64_t i = 0; i < ne; i++) { float x = gate[i]; o[i] = (x/(1.0f+expf(-x))) * up[i]; }  // CPU fallback
    return true;
}

static enum ggml_status ggml_backend_ork_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
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
    // QKV/gate-up fusion: implemented + correct, but measured SLOWER on RK3588 (decode 9.4->6.4
    // tok/s) — one wide multi-core matmul + strided scatter costs more than the 2 saved submits, i.e.
    // the NPU per-matmul cost scales with work, it's not a fixed floor fusion can amortize. Off by
    // default; opt in with ORK_FUSE=1 to experiment (may differ on larger models / tuned scatter).
    const int fuse = (ctx->qbits == 8) && (getenv("ORK_FUSE") != nullptr);
    // FUSED FFN chain requires ALL of a layer's FFN weights (gate/up/down) co-resident in ONE IOMMU domain:
    // each ork_mm_run_* submit binds buffers in the active domain, and a submit touching a buffer reserved in
    // a DIFFERENT domain faults the NPU (soft reset). Enabled when EITHER single-domain (n_domains<=1) OR
    // LAYER-ALIGNED multi-domain (domain_layers>0), which places every layer's weights — including the lazily
    // packed fused gate — in ONE domain (domain=layer/domain_layers). Even so, the handler makes a RUNTIME
    // co-domain check (ork_w_domain of gate/up/down) and falls back to per-node if they ever differ, so a
    // cross-domain fused submit can never fire (wedge-safe). PER-DOMAIN FUSION for >4GiB models.
    const bool ffn_chain = ork_ffn_chain_on() && ctx->qbits == 8 && (ctx->n_domains <= 1 || ctx->domain_layers > 0);
    if (getenv("ORK_DUMP_GRAPH")) { static int dumped = 0;
        bool has_ffn = false, has_glu = false;
        for (int i = 0; i < cgraph->n_nodes; i++) { struct ggml_tensor * n = cgraph->nodes[i];
            if (n->op == GGML_OP_GLU) has_glu = true;
            if (n->src[0] && (strstr(n->src[0]->name,"ffn_gate")||strstr(n->src[0]->name,"ffn_down"))) has_ffn = true; }
        if ((has_ffn || has_glu) && dumped++ < 3) {
            fprintf(stderr, "[ORK GRAPH] FFN subgraph, %d nodes (has_ffn=%d has_glu=%d):\n", cgraph->n_nodes, has_ffn, has_glu);
            for (int i = 0; i < cgraph->n_nodes && i < 40; i++) { struct ggml_tensor * n = cgraph->nodes[i];
                fprintf(stderr, "  [%2d] %-12s src0=%-28s src1=%-20s ne=[%ld,%ld]\n", i, ggml_op_name(n->op),
                        n->src[0]?n->src[0]->name:"-", n->src[1]?n->src[1]->name:"-", (long)n->ne[0], (long)n->ne[1]);
            } fflush(stderr); } }
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        // ORK_FFN_CHAIN: recognize the fused-SwiGLU FFN inner and run it as ONE round-trip-free int8 chain
        // (int8 intermediates never touch fp32). Consumes all 4 nodes; falls through to per-node on any miss.
        if (ffn_chain && node->op == GGML_OP_MUL_MAT) {
            struct ggml_tensor *g,*u,*gl,*dn; int last;
            if (ork_ffn_chain_match(cgraph, i, &g, &u, &gl, &dn, &last)) {
                if (!ggml_backend_ork_ffn_swiglu_chain(ctx, g, u, gl, dn)) return GGML_STATUS_FAILED;
                i = last;      // skip past gate/up/GLU/down — all handled by the chain
                continue;
            }
        }
        switch (node->op) {
            case GGML_OP_MUL_MAT: {
                std::vector<struct ggml_tensor *> chain_nodes;
                ork_chain_type type = get_node_chain_type(ctx, node);

                // STREAM-POOL mode handles each MUL_MAT individually (the per-node int8 path is
                // stream-pool-aware; the multi-weight chain handler is not — it needs distinct weights
                // co-resident under allow_evict=false, which the RAM/IOVA tiering can't guarantee).
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
                    if (fuse && node->ne[2] == 1 && node->ne[3] == 1) {
                        while (i + ng < cgraph->n_nodes && ng < 16) {
                            struct ggml_tensor * nj = cgraph->nodes[i + ng];
                            if (nj->op == GGML_OP_MUL_MAT && nj->src[1] == node->src[1] &&
                                nj->src[0]->ne[0] == node->src[0]->ne[0] && nj->ne[2] == 1 && nj->ne[3] == 1)
                                grp[ng++] = nj;
                            else break;
                        }
                    }
                    if (ng >= 2) {
                        if (!ggml_backend_ork_mul_mat_group_i8(ctx, grp, ng)) return GGML_STATUS_FAILED;
                        i += ng - 1;
                    } else {
                        const char * name = node->src[0]->name;
                        bool is_ffn = strstr(name, "ffn_") || ork_is_expert(name);
                        bool is_attn = strstr(name, "attn_q") || strstr(name, "attn_k") || strstr(name, "attn_v") || strstr(name, "attn_output");
                        
                        int target_qbits = ctx->qbits;
                        if (ctx->hybrid) {
                            if (is_ffn) target_qbits = 4;
                            else if (is_attn) target_qbits = 8;
                        }

                        if (target_qbits == 4
                               ? (ctx->hadamard ? !ggml_backend_ork_mul_mat_i4_hadamard(ctx, node)
                                                : !ggml_backend_ork_mul_mat_i4(ctx, node))
                               : !ggml_backend_ork_mul_mat_i8(ctx, node)) {
                            return GGML_STATUS_FAILED;
                        }
                    }
                }
                break;
            }
            case GGML_OP_MUL_MAT_ID: {
                if (!ggml_backend_ork_mul_mat_id_i8(ctx, node)) return GGML_STATUS_FAILED;
                break;
            }
            case GGML_OP_MUL: {
                if (!ggml_backend_ork_mul_f16(ctx, node)) return GGML_STATUS_FAILED;
                break;
            }
            case GGML_OP_ADD: {
                if (!ggml_backend_ork_add_f16(ctx, node)) return GGML_STATUS_FAILED;
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
    return GGML_STATUS_SUCCESS;
    GGML_UNUSED(backend);
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
    ork_npu * npu = ork_npu_init();
    if (!npu) { GGML_LOG_ERROR("%s: ork_npu_init failed (no NPU / no perms)\n", __func__); return NULL; }
    ggml_backend_ork_context * ctx = new ggml_backend_ork_context;
    ctx->npu = npu;
    g_ork_ctx = ctx;
    ork_persist_init(ctx);   // .orkpack: read (fast load) if present, else build it this run
    const char * q = getenv("ORK_QUANT");
    ctx->qbits = (q && q[0] == '4') ? 4 : 8;   // ORK_QUANT=4 -> W4A4; default (unset/8) -> W8A8
    ctx->profile = getenv("ORK_PROFILE") != nullptr;
    if (ctx->profile) atexit(ork_profile_atexit);   // LEVER3: dump under llama-bench (no backend free)
    ctx->no_reuse = getenv("ORK_NOREUSE") != nullptr;
    ctx->no_cache = getenv("ORK_NOCACHE") != nullptr;
    ctx->hybrid = g_ork_hybrid_loading || getenv("ORK_HYBRID") != nullptr;
    ctx->hadamard = (ctx->qbits == 4) && getenv("ORK_HADAMARD") != nullptr;
    ctx->phase_evict = env_enabled("ORK_MOE_PHASE_EVICT");   // #1 phase-aware backbone eviction (default OFF)
    // MULTI-DOMAIN RESIDENCE: ORK_DOMAINS>1 spreads weights across that many IOMMU domains (each with its
    // own ~4 GiB IOVA window) so a >4 GiB model stays fully resident with NO streaming/churn. ORK_DOMAIN_LAYERS
    // sets layers-per-domain (0 = auto from a 28-layer assumption). Pass a large ORK_WCACHE_BUDGET_MB so the
    // residence never evicts.
    { const char * nd = getenv("ORK_DOMAINS");
      if (nd) {                                                  // explicit override (debug/pinning)
          ctx->n_domains = atoi(nd);
          if (ctx->n_domains < 1)  ctx->n_domains = 1;
          if (ctx->n_domains > 16) ctx->n_domains = 16;
      } else if (!ctx->persist_idx.empty()) {
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
          size_t total = 0; for (const auto & kv : ctx->persist_idx) total += kv.second.blob_size;
          const size_t cap = (size_t) 3000 * 1024 * 1024;
          size_t inflated = total * 21 / 10;                       // resident Bb+Bf estimate (~2.1x int8 blob)
          long nd2 = (long) ((inflated + cap - 1) / cap);          // ceil(inflated / 3.0 GiB) = min domains needed
          ctx->n_domains = nd2 < 1 ? 1 : (nd2 > 16 ? 16 : (int) nd2);
          // EVEN packing: cap each domain at inflated/n_domains (+ small slack) instead of the 3.0 GiB ceiling,
          // so domains fill UNIFORMLY (7B: ~2.55 GiB each x5, ~1.5 GiB headroom) rather than 3.0/3.0/3.0/3.0/0.26
          // (early domains tight, last overflowing). Never exceeds the 3.0 GiB hard cap (inflated/n <= 3.0 by
          // construction). +64 MiB slack absorbs per-weight granularity so the last domain isn't forced over.
          if (ctx->n_domains > 1)
              ctx->domain_fill_cap = inflated / (size_t) ctx->n_domains + (size_t) 64 * 1024 * 1024;
          // LAYER-ALIGNED domains: assign each transformer layer's weights to ONE domain (domain = layer/
          // domain_layers) so a per-layer fused FFN chain never crosses a domain -> enables PER-DOMAIN
          // FUSION for >4GiB models. domain_layers = ceil(n_layers / n_domains) spreads layers ~evenly;
          // n_layers = max "blk.N" index + 1 over the .orkpack index. Non-layer NPU weights (lm_head) -> the
          // last (under-filled) domain. When 0 (no index) ork_weight_domain falls back to byte-balanced fill.
          if (ctx->n_domains > 1 && ctx->domain_layers == 0) {
              int max_layer = -1;
              for (const auto & kv : ctx->persist_idx) { int l = ork_layer_of(kv.first.c_str()); if (l > max_layer) max_layer = l; }
              if (max_layer >= 0) ctx->domain_layers = (max_layer + 1 + ctx->n_domains - 1) / ctx->n_domains;
          }
          if (getenv("ORK_VERBOSE"))
              fprintf(stderr, "[ORK] auto n_domains=%d (domain_layers=%d) from .orkpack footprint %.2f GiB int8 (x2.1 Bb+Bf; EVEN fill %.2f GiB/domain)\n",
                      ctx->n_domains, ctx->domain_layers,
                      total / (1024.0 * 1024.0 * 1024.0), ctx->domain_fill_cap / (1024.0 * 1024.0 * 1024.0));
      } else {
          // No .orkpack index (live-pack / write mode): footprint unknown up front (weights arrive one matmul
          // at a time). Keep a domain ceiling; ork_weight_domain() fills only as many as the resident set needs.
          ctx->n_domains = 8;
      }
      const char * dl = getenv("ORK_DOMAIN_LAYERS"); ctx->domain_layers = dl ? atoi(dl) : 0; }
    // Residency = ork-driver's MULTI-DOMAIN mechanism (weights resident across up to 16 IOMMU domains,
    // each its own ~4 GiB IOVA window; dom_activate zero-copy-swaps the active domain per submit). This
    // resides up to ~64 GiB with no streaming — every practical model. The stream pool (RAM-hold + dma-buf
    // map/unmap) is only needed BEYOND that; leave it off (NULL) so the domain path is used.
    ctx->spool = nullptr;
    fprintf(stderr, "[ork] residency: multi-domain (up to %d domains x ~4 GiB IOVA, dom_activate swap)\n", ctx->n_domains);
    // One-line version banner to stderr — visible even under llama-bench (which suppresses
    // GGML_LOG_INFO). Cheap, once per backend init. ork_npu_version() = semver (+git hash if built
    // with one). Makes "which build is this?" answerable from any benchmark/run log.
    fprintf(stderr, "[ork] ork-driver %s (W%dA%d%s)\n", ork_npu_version(),
            ctx->qbits, ctx->qbits, ctx->hadamard ? "+Had" : "");
    GGML_LOG_INFO("%s: ork backend ready (ork-driver %s, %sW%dA%d%s)\n", __func__, ork_npu_version(),
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
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    switch (op->op) {
        case GGML_OP_NONE: case GGML_OP_RESHAPE: case GGML_OP_VIEW:
        case GGML_OP_PERMUTE: case GGML_OP_TRANSPOSE:
            return true;
        case GGML_OP_MUL_MAT: {
            const int64_t K = src0->ne[0], N = op->ne[0], M = op->ne[1];
            // Explicitly block output and lm_head (vocabulary projection) layers from offloading to NPU.
            // These layers are extremely wide (e.g. N=151936), causing massive DMA buffer allocation and
            // packing overhead, which can trigger NPU driver IOVA allocation failures and kernel hangs.
            const char * name = src0->name;
            if(getenv("ORK_VERBOSE"))fprintf(stderr, "[ORK DEBUG supports_op] name='%s' K=%ld N=%ld M=%ld\n", name, (long)K, (long)N, (long)M);
            fflush(stderr);
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
            bool hadamard = g_ork_ctx ? g_ork_ctx->hadamard : (getenv("ORK_HADAMARD") != nullptr);
            bool is_grouped = (src0->type == GGML_TYPE_Q4_0  ||
                               src0->type == GGML_TYPE_Q4_1  ||
                               src0->type == GGML_TYPE_Q4_K  ||
                               src0->type == GGML_TYPE_IQ4_NL ||
                               src0->type == GGML_TYPE_IQ4_XS) && !hadamard;
            int threshold = is_expert ? 1 : (is_grouped ? min_m : (min_m > 32 ? 32 : min_m));
            if (target_qbits == 8) threshold = 1; // i8 chaining makes M=1 decode fast on NPU
            // EXPERIMENT #1 (ORK_MOE_PHASE_EVICT): at DECODE (M==1) DECLINE the dense backbone matmuls so
            // the scheduler routes them to CPU (bandwidth-bound, cheap at M=1) — this frees the ~2.8 GiB of
            // IOVA the backbone otherwise pins, handing it to the MoE hot-expert cache. Experts go through
            // MUL_MAT_ID (a separate case, still accepted). MUL_MAT here is the dense/attn backbone (the
            // _exps tensors never reach this case), so declining at M==1 is exactly the backbone-at-decode.
            {
                static const int pe = env_enabled("ORK_MOE_PHASE_EVICT");
                if (pe && M == 1 && op->ne[2] == 1 && op->ne[3] == 1) return false;
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
            if (!env_enabled("ORK_MOE_NPU") && !env_enabled("ORK_NO_EXPERT_REPACK")) return false;
            {   // one-time loud warning when the experimental MoE-on-NPU path is actually enabled
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
            // fp16 SwiGLU multiply / residual add (ork_npu_ewmul_f16 / add_f16). EXPERIMENTAL,
            // default-off. MUL under ORK_PPU_OPS (bit-exact fp16); ADD needs ORK_PPU_ADD (fp16
            // residual accumulation caveat). Only the same-shape contiguous f32 case (SwiGLU
            // gate*up, residual x+y); broadcast MUL (norm-weight, RoPE) stays on CPU.
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
            // int8 SILU / GELU (ork_npu_silu_i8 / gelu_i8). EXPERIMENTAL, per-op opt-in
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
            if (!ork_ppu_glu_on() && !ork_ffn_chain_on()) return false;
            if (ggml_get_glu_op(op) != GGML_GLU_OP_SWIGLU) return false;
            if (!src0 || !src1) return false;                        // split form only (two inputs)
            if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) return false;
            if (!ggml_are_same_shape(src0, src1) || !ggml_are_same_shape(src0, op)) return false;
            if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(op)) return false;
            const int64_t N = op->ne[0], ne = ggml_nelements(op), M = ne / N;
            // ORK_FFN_CHAIN consumes the GLU INSIDE the fused chain handler (the ewmul runs on CPU/tiled), so
            // it is NOT bound by the standalone ork_npu_ewmul_i8 N<=8192 cap. Claim support at any N so the
            // FFN's gate/up/GLU/down land in ONE ork subgraph and the chain matcher can fuse them — otherwise
            // the 7B's Nff=18944 GLU is rejected, the scheduler splits the FFN across backends, and the 4
            // nodes never share a graph_compute (chain can never fire). Standalone GLU (ORK_PPU_GLU) keeps cap.
            if (ork_ffn_chain_on()) return N >= 16 && (N & 15) == 0 && M >= 1 && M <= 8192;
            return N >= 16 && N <= 8192 && (N & 15) == 0 && M >= ork_ppu_minm() && M <= 8192;
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
