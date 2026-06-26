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
//   ORK_MOE_NPU=1        EXPERIMENTAL. Offload MoE experts (MUL_MAT_ID) to the NPU via the
//                        group-by-expert int8 handler. Measured a NET LOSS on RK3588 (decode ~60x,
//                        prefill ~80x slower — the per-expert submit floor dominates), so default OFF
//                        = experts on CPU. Requires the matching repack-buffer exclusion in
//                        ggml-cpu/repack.cpp (gated on the same flag). Legacy alias: ORK_NO_EXPERT_REPACK.
//   ORK_MOE_CACHE=<n>    Resident expert-pool slots PER SHAPE (default 384); reused round-robin via
//                        ork_mm_repack_i8 (no IOMMU churn). Only relevant when ORK_MOE_NPU is on.
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
#include "gguf.h"

#include <vector>
#include <map>
#include <algorithm>
#include <cstring>
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
#include <string>
#include <fcntl.h>
#include <unistd.h>

// ---- .orkpack persist format: a self-populating on-disk cache of pre-tiled (mixed int8 / int4) weights ----
// File: [ blobs: per weight, packed bytes then (int8 only) N bscale floats ][ index ][ footer@EOF ].
// First run with ORK_PERSIST=<path> writes it (one slow pass); later runs mmap it and load the bytes
// straight into DMA — no dequant/quant/tile. Each weight's (K,N,dtype) is re-checked on load, so a
// stale/mismatched file can never feed wrong weights (a mismatch just falls back to packing).
//
// Per-tensor tier is carried in `dtype` (the field is back-compatible — v1 files only ever wrote dtype==1):
//   dtype == ORKPACK_DT_I8 (1): blob = ork_w_dump bytes (tiled int8), followed by bscale_n bscale floats.
//   dtype == ORKPACK_DT_I4 (4): blob = ork_w_dump_i4a8 bytes (self-describing 'O4N1': K,N,quant_kind +
//                               bscale[N] + nibble store). bscale lives INSIDE the blob → bscale_n==0,
//                               bscale_off unused. Loaded via ork_mm_load_i4a8, runs via ork_mm_run_i8.
// The struct layout is unchanged from v1, so v1 (all-int8) files load unmodified; VERSION bumps to 2 to
// mark files that may contain int4 entries (both versions are accepted on read).
#define ORKPACK_MAGIC   "ORKPK01"
#define ORKPACK_VERSION 2u
#define ORKPACK_DT_I8   1u
#define ORKPACK_DT_I4   4u
struct orkpack_entry  { uint32_t K, N, dtype, bscale_n; uint64_t blob_off, blob_size, bscale_off; };
struct orkpack_footer { uint64_t index_off; uint32_t n_entries; uint32_t version; char magic[8]; };

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
// Layer-streaming: the NPU IOMMU addresses only ~4 GiB at once (rk_iommu is 32-bit), so a >4 GiB model
// can't keep every weight resident. Cap wcache resident bytes at a budget and evict the LRU weight
// (reclaiming IOVA via ork_mm_free) to make room. In one prefill pass each weight packs once → amortizes.
static size_t ork_wcache_budget() {
    static size_t b = 0;
    if (!b) { const char * e = getenv("ORK_WCACHE_BUDGET_MB"); b = (size_t)(e ? atoll(e) : 3072) * 1024 * 1024; }
    return b;
}

// a packed quantized weight + its scales, kept NPU-resident and reused.
//   int8 (W8A8):  gsize==0, bscale [N]            (per output channel)
//   int4 (W4A4):  gsize==G,  bscale [(K/G)*N]      (per K-group, per channel)
struct ork_weight {
    ork_w * w = nullptr;
    std::vector<float> bscale;
    int gsize = 0;
    size_t   bytes = 0;       // resident NPU bytes (for the streaming LRU budget)
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
    bool hybrid = false;       // ORK_HYBRID: per-weight W8A8/W4A4 by SIZE (see ork_pick_qbits)
    size_t hybrid_w4_above = 0;// ORK_HYBRID_W4_ABOVE_MB (bytes): weights whose int8 footprint K*N >= this run W4A4
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
    uint64_t wcache_tick  = 0;   // monotonic clock for LRU last_use
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
    double t_run_dec = 0, t_run_pf = 0; long n_dec = 0, n_pf = 0, m_pf = 0;
    // MoE chained-handler phase breakdown (ORK_PROFILE): where the 0.97 t/s goes.
    double moe_prequant = 0, moe_pack = 0, moe_gather = 0, moe_chain = 0, moe_scatter = 0; long moe_calls = 0;
    double moe_deq = 0, moe_quant = 0;   // pack/repack sub-split: Q4_K->f32 dequant vs f32->int8 quant+tile
    double moe_cold = 0; long moe_cold_calls = 0;   // cold-expert CPU GEMV (threaded ggml vec_dot) wall time
};
static ggml_backend_ork_context * g_ork_ctx = nullptr;
static bool g_ork_hybrid_loading = false;

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
                if (memcmp(f.magic, ORKPACK_MAGIC, 8) == 0 && (f.version == 1u || f.version == 2u) && f.index_off < (uint64_t) sz) {
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
static void ork_persist_write(ggml_backend_ork_context * ctx, const char * name, int K, int N,
                              const ork_weight & ow, const float * f32_plane, enum ggml_type src_type) {
    if (ctx->persist_mode != 2 || !ctx->persist_out) return;
    if (!ctx->persist_dumped.insert(name).second) return;   // already dumped — a convert-decode re-pack, don't duplicate

    int tier = f32_plane ? ork_orkpack_tier(name, K, N, src_type) : 8;   // no f32 plane available → int8 only
    if (getenv("ORK_VERBOSE"))
        fprintf(stderr, "[ORK PERSIST] tier %s K=%d N=%d src=%s -> int%d\n",
                name, K, N, ggml_type_name(src_type), tier);
    if (tier == 4) {
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
    size_t tb = ork_w_dump(ow.w, nullptr, 0);
    std::vector<char> tmp(tb);
    ork_w_dump(ow.w, tmp.data(), tb);
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
static void ork_persist_write_experts(ggml_backend_ork_context * ctx, const struct ggml_tensor * src0,
                                      int K, int N, enum ggml_type type, ggml_to_float_t to_float) {
    if (ctx->persist_mode != 2 || !ctx->persist_out) return;
    const int n_expert = (int) src0->ne[2];
    std::vector<float> f32((size_t) N * K);
    for (int e = 0; e < n_expert; e++) {
        const std::string key = ork_expert_key(src0->name, e);
        if (ctx->persist_dumped.count(key)) continue;   // already dumped this slice
        const char * x = (const char *) src0->data + (size_t) e * src0->nb[2];
        const size_t nb01 = src0->nb[1];
        if (type == GGML_TYPE_F32) {
            for (int64_t n = 0; n < N; n++) memcpy(f32.data() + n*K, x + n*nb01, (size_t) K*sizeof(float));
        } else {
            for (int64_t n = 0; n < N; n++) to_float(x + n*nb01, f32.data() + n*K, K);
        }
        // pack a temporary int8 weight from the f32 plane (per-channel absmax, mirrors the dense path) so
        // ork_persist_write's int8 tier can dump ow.w; the int4 tier ignores ow.w and re-packs from f32.
        ork_weight ow; ow.bscale.resize(N);
        std::vector<int8_t> bi((size_t) K * N);
        for (int n = 0; n < N; n++) {
            float mx = 1e-9f;
            for (int k = 0; k < K; k++) { float v = fabsf(f32[(size_t) n*K + k]); if (v > mx) mx = v; }
            float scale_val = mx / 127.0f; ow.bscale[n] = scale_val;
            for (int k = 0; k < K; k++) {
                int q = (int) lrintf(f32[(size_t) n*K + k] / scale_val);
                bi[(size_t) k*N + n] = (int8_t) (q > 127 ? 127 : q < -127 ? -127 : q);
            }
        }
        ow.w = ork_mm_pack_i8(ctx->npu, K, N, bi.data());
        if (!ow.w) { if (getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK PERSIST] expert pack FAIL %s\n", key.c_str()); continue; }
        ork_persist_write(ctx, key.c_str(), K, N, ow, f32.data(), type);
        ork_mm_free(ctx->npu, ow.w);   // resident ~0: free before the next slice
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
static std::unordered_map<const void *, ork_weight>::iterator
ork_resolve_weight_i8(ggml_backend_ork_context * ctx, const struct ggml_tensor * src0,
                      int K, int N, size_t nb01, enum ggml_type type, ggml_to_float_t to_float,
                      bool allow_evict) {
    // allow_evict: the non-chain path uses one weight at a time, so it may stream-evict the LRU to
    // free IOVA. The chain path needs ALL `count` weights co-resident at submit, so it passes false
    // (matching the original chain pack, which never evicted) — otherwise packing weight i frees the
    // already-packed weight i-1 that the chain still references → use-after-free at submit.
    const char * x = (const char *) src0->data;
    auto it = ctx->wcache.find(x);
    if (it != ctx->wcache.end()) return it;

    if (ctx->persist_mode == 1) {
        // .orkpack hit: load pre-tiled bytes straight into DMA (no dequant/quant/tile). Per-weight
        // (K,N,dtype) is re-checked so a stale file can't feed wrong weights — mismatch → pack below.
        auto pit = ctx->persist_idx.find(src0->name);
        if (pit != ctx->persist_idx.end() && pit->second.K == (uint32_t) K && pit->second.N == (uint32_t) N &&
            (pit->second.dtype == ORKPACK_DT_I8 || pit->second.dtype == ORKPACK_DT_I4)) {
            const orkpack_entry & e = pit->second;
            if (allow_evict) ork_wcache_evict(ctx, (size_t) K * N);
            ork_weight ow;
            const char * blob = (const char *) ctx->persist_map + e.blob_off;
            if (e.dtype == ORKPACK_DT_I4) {
                ow.w = ork_mm_load_i4a8(ctx->npu, K, N, blob, e.blob_size);
                if (ow.w) { const float * bs = ork_w_bscale(ow.w); if (bs) ow.bscale.assign(bs, bs + N); }
            } else {
                ow.w = ork_mm_load_i8(ctx->npu, K, N, blob, e.blob_size);
                if (ow.w) { const float * bs = (const float *) ((const char *) ctx->persist_map + e.bscale_off);
                            ow.bscale.assign(bs, bs + e.bscale_n); }
            }
            if (ow.w) {
                it = ctx->wcache.emplace(x, std::move(ow)).first;
                it->second.bytes = ork_w_bytes(it->second.w);
                ctx->wcache_bytes += it->second.bytes;
                ctx->persist_hits++;
                if (getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK PERSIST] %s %s K=%d N=%d\n",
                    e.dtype == ORKPACK_DT_I4 ? "int4-load" : "int8-load", src0->name, K, N);
                ork_evict_src(x, (size_t) N * nb01);
                return it;
            }
        }
    }

    // pack-miss: dequant -> per-channel int8 quant -> pack -> (write mode) persist
    if (ctx->persist_mode) ctx->persist_misses++;
    ctx->f32.resize((size_t) N * K);
    ctx->bi .resize((size_t) K * N);
    float  * f32 = ctx->f32.data();
    int8_t * bi  = ctx->bi.data();
    if (type == GGML_TYPE_F32) {
        for (int64_t n = 0; n < N; n++) memcpy(f32 + n*K, x + n*nb01, (size_t) K*sizeof(float));
    } else {
        for (int64_t n = 0; n < N; n++) to_float((const char *) x + n*nb01, f32 + n*K, K);
    }
    ork_weight ow; ow.bscale.resize(N);
    for (int n = 0; n < N; n++) {
        float mx = 1e-9f;
        for (int k = 0; k < K; k++) { float v = fabsf(f32[(size_t) n*K + k]); if (v > mx) mx = v; }
        float scale_val = mx / 127.0f; ow.bscale[n] = scale_val;
        for (int k = 0; k < K; k++) {
            int q = (int) lrintf(f32[(size_t) n*K + k] / scale_val);
            bi[(size_t) k*N + n] = (int8_t) (q > 127 ? 127 : q < -127 ? -127 : q);
        }
    }
    if (allow_evict) ork_wcache_evict(ctx, (size_t) K * N);   // stream: free IOVA room before packing (int8 ~K*N bytes)
    ow.w = ork_mm_pack_i8(ctx->npu, K, N, bi);
    if (!ow.w) return ctx->wcache.end();
    it = ctx->wcache.emplace(x, std::move(ow)).first;
    it->second.bytes = ork_w_bytes(it->second.w);
    ctx->wcache_bytes += it->second.bytes;
    ork_persist_write(ctx, src0->name, K, N, it->second, ctx->f32.data(), type);   // .orkpack: dump for next time (f32 plane enables int4 tier; src type drives tier)
    ork_evict_src(x, (size_t) N * nb01);   // source plane now dead weight (custom loader)
    if (getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK DEBUG] packed weight, wcache=%zu resident=%.0fMB, K=%d, N=%d, x=%p\n",
        ctx->wcache.size(), ctx->wcache_bytes/1e6, K, N, (const void *) x);
    return it;
}

void ggml_backend_ork_set_hybrid(bool use_hybrid) {
    g_ork_hybrid_loading = use_hybrid;
}
static inline double ork_now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec*1e6 + t.tv_nsec*1e-3; }

// dst = src0 x src1 :  src0 [K=ne00, N=ne01], src1 [K=ne10=ne00, M=ne11], dst [N, M] (row-major [M][N])
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
                #pragma omp parallel for if (M_padded >= 16)
                for (int m = 0; m < M_padded; m++) {
                    if (m < M) {
                        const float * yr = y + (size_t) m*K;
                        int8_t * amr = ar + (size_t) m*K;
                        float mx = 1e-9f;
                        for (int k = 0; k < K; k++) { float v = fabsf(yr[k]); mx = v > mx ? v : mx; }
                        asr[m] = mx / 127.0f;
                        const float inv = 127.0f / mx;
                        for (int k = 0; k < K; k++) {
                            float q = yr[k] * inv;
                            int qi = (int) (q + copysignf(0.5f, q));
                            amr[k] = (int8_t) (qi > 127 ? 127 : qi < -127 ? -127 : qi);
                        }
                    } else {
                        memset(ar + (size_t) m*K, 0, K);
                        asr[m] = 0.0f;
                    }
                }
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
        }

        const double t1 = ctx->profile ? ork_now_us() : 0;

        if(getenv("ORK_VERBOSE"))fprintf(stderr, "[ORK] i8 chain: M=%d, tasks=%zu (S=%d, K=%d, N=%d)\n", M, tasks.size(), S, K, N);
        fflush(stderr);
        int ok = -1;
        if (tasks.size() == 1) {
            ok = ork_mm_run_i8(ctx->npu, tasks[0].w, tasks[0].M, tasks[0].A, tasks[0].C);
        } else {
            ok = ork_mm_run_chain_i8(ctx->npu, tasks.size(), tasks.data());
            if (ok != 0) {
                // Fallback to sequential single-task run
                for (size_t t = 0; t < tasks.size(); t++) {
                    if (ork_mm_run_i8(ctx->npu, tasks[t].w, tasks[t].M, tasks[t].A, tasks[t].C)) {
                        return false;
                    }
                }
            }
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

static void ggml_backend_ork_free(ggml_backend_t backend) {
    ggml_backend_ork_context * ctx = (ggml_backend_ork_context *) backend->context;
    if (ctx->profile && ctx->n_mm) {
        double tot = ctx->t_quant + ctx->t_run + ctx->t_deq;
        GGML_LOG_INFO("ork profile: %ld matmuls | quant %.0fms (%.0f%%) run %.0fms (%.0f%%) dequant %.0fms (%.0f%%) | %.1f us/matmul (run %.1f us)\n",
            ctx->n_mm, ctx->t_quant/1e3, 100*ctx->t_quant/tot, ctx->t_run/1e3, 100*ctx->t_run/tot,
            ctx->t_deq/1e3, 100*ctx->t_deq/tot, tot/ctx->n_mm, ctx->t_run/ctx->n_mm);
        if (ctx->n_dec) GGML_LOG_INFO("ork profile: decode  (M=1)  %ld matmuls, run %.1f us/matmul\n", ctx->n_dec, ctx->t_run_dec/ctx->n_dec);
        if (ctx->n_pf)  GGML_LOG_INFO("ork profile: prefill (M>1) %ld matmuls, avgM %.1f, run %.1f us/matmul (%.2f us/row)\n",
            ctx->n_pf, (double)ctx->m_pf/ctx->n_pf, ctx->t_run_pf/ctx->n_pf, ctx->t_run_pf/ctx->m_pf);
        if (ctx->moe_calls) {
            double mt = ctx->moe_prequant + ctx->moe_pack + ctx->moe_gather + ctx->moe_chain + ctx->moe_scatter;
            fprintf(stderr, "[ork MoE-chain] %ld calls, %.0fms total | prequant %.0fms (%.0f%%) pack/repack %.0fms (%.0f%%) gather %.0fms (%.0f%%) chain-submit %.0fms (%.0f%%) scatter %.0fms (%.0f%%)\n",
                ctx->moe_calls, mt/1e3,
                ctx->moe_prequant/1e3, 100*ctx->moe_prequant/mt, ctx->moe_pack/1e3, 100*ctx->moe_pack/mt,
                ctx->moe_gather/1e3, 100*ctx->moe_gather/mt, ctx->moe_chain/1e3, 100*ctx->moe_chain/mt,
                ctx->moe_scatter/1e3, 100*ctx->moe_scatter/mt);
            fprintf(stderr, "[ork MoE-chain] pack split: Q4_K->f32 dequant %.0fms (%.0f%%) | f32->int8 quant+tile %.0fms (%.0f%%)\n",
                ctx->moe_deq/1e3, 100*ctx->moe_deq/ctx->moe_pack, ctx->moe_quant/1e3, 100*ctx->moe_quant/ctx->moe_pack);
        }
        if (ctx->moe_cold_calls)
            fprintf(stderr, "[ork MoE-cold] %ld cold-expert GEMV calls (threaded ggml vec_dot) | %.0fms total | %.1f us/expert\n",
                ctx->moe_cold_calls, ctx->moe_cold/1e3, ctx->moe_cold/ctx->moe_cold_calls);
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
    if (ctx->persist_map) munmap(ctx->persist_map, ctx->persist_map_sz);
    for (auto & kv : ctx->wcache) ork_w_free(kv.second.w);
    for (auto & p : ctx->moe_pools) for (auto & s : p.second) if (s.w) ork_w_free(s.w);   // MoE expert pool
    for (auto & tk : ctx->moe_hot) for (auto & es : tk.second) if (es.second.w) ork_w_free(es.second.w);   // hot-expert partition
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

// Hybrid precision policy (ORK_HYBRID): choose W8A8 vs W4A4 per weight by SIZE. A weight whose int8-
// resident footprint (K*N bytes) is >= ctx->hybrid_w4_above runs W4A4 — that's the FFN / expert bulk,
// where int4 halves both resident bytes AND streaming bandwidth (the binding constraints for large
// models against the ~4 GiB NPU IOVA window + KV-cache headroom). Smaller, numerically-sensitive
// weights (attention, gates, norms-adjacent projections) stay W8A8, where the byte cost is trivial and
// accuracy matters more. This generalizes the old FFN=4/attn=8 name heuristic: on a standard transformer
// it lands on the same split, but it is architecture-agnostic and tunable via ORK_HYBRID_W4_ABOVE_MB
// (default 8 MB; set 0 to force all-W8A8 under hybrid). With ORK_HYBRID off, ctx->qbits is used verbatim.
static int ork_pick_qbits(const ggml_backend_ork_context * ctx, int64_t K, int64_t N) {
    if (!ctx->hybrid) return ctx->qbits;
    return (ctx->hybrid_w4_above && (size_t) (K * N) >= ctx->hybrid_w4_above) ? 4 : 8;
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

    int target_qbits = ork_pick_qbits(ctx, K, N);
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
    
    int ok = 0;
    if (tasks.size() == 1) {
        ok = ork_mm_run_i8(ctx->npu, tasks[0].w, tasks[0].M, tasks[0].A, tasks[0].C) ? -1 : 0;
    } else {
        ok = ork_mm_run_chain_i8(ctx->npu, tasks.size(), tasks.data());
    }
    
    if (ok != 0) {
        // Fallback to sequential single-task run
        if(getenv("ORK_VERBOSE"))fprintf(stderr, "[ORK] i8 chain failed (%d), falling back to sequential\n", ok); fflush(stderr);
        for (size_t t = 0; t < tasks.size(); t++) {
            if (ork_mm_run_i8(ctx->npu, tasks[t].w, tasks[t].M, tasks[t].A, tasks[t].C)) {
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
    // ===== M1b/M3 STATIC HOT-EXPERT PARTITION =====
    // Per LAYER-TENSOR cap K (ORK_MOE_HOT, default 8). The hottest K experts of THIS `_exps` tensor stay
    // resident on the NPU (LRU within the cap, IOVA freed on evict); every other (cold) expert this step
    // is computed on the CPU (dequant + f32 GEMV, the same math the convert path uses). Residency is thus
    // bounded to K * (#_exps tensors) * per-proj bytes — for LFM2.5 top-8 = ~1.9 GiB, well under 4 GiB —
    // so the IOMMU never EFAULTs (the live-path soft-reset) regardless of how many experts the router hits.
    static const size_t hot_K = getenv("ORK_MOE_HOT") ? (size_t) atoi(getenv("ORK_MOE_HOT")) : 8;
    static const size_t hot_budget = (size_t)((getenv("ORK_MOE_HOT_GIB") ? atof(getenv("ORK_MOE_HOT_GIB")) : 2.5) * 1024.0*1024.0*1024.0);
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
    auto quant_tok = [&](int t, int jslot) -> const void * {
        const int key = bcast ? t : (t * 100000 + jslot);
        auto qit = tok_q.find(key);
        if (qit != tok_q.end()) return qact.data() + qit->second;
        const size_t off = qact.size(); qact.resize(off + vdt_row);
        const float * y = (const float *)((const char *) src1->data + (size_t)(bcast?0:jslot)*src1->nb[1] + (size_t) t*src1->nb[2]);
        if (vdt == GGML_TYPE_F32) memcpy(qact.data()+off, y, (size_t)K*sizeof(float));
        else vdt_tt->from_float(y, qact.data()+off, K);
        tok_q[key] = off;
        return qact.data() + off;
    };

    // Collect cold (expert, entries) pairs; run them all in one threaded fan-out after the split.
    std::vector<std::pair<int, std::vector<std::pair<int,int>> *>> cold;
    auto cpu_expert = [&](int e, std::vector<std::pair<int,int>> & ent) { cold.push_back({e, &ent}); };
    // Run one cold expert: for each routed (token,slot), dot all N weight rows against the quantized act.
    auto run_cold_expert = [&](int e, std::vector<std::pair<int,int>> & ent) {
        const char * xw = (const char *) src0->data + (size_t) e * src0->nb[2];
        for (auto & pr : ent) {
            const int t = pr.first, j = pr.second;
            const void * qa = quant_tok(t, j);                    // pre-quantized activation (read-only)
            float * dr = (float *)(dbase + (size_t) j * dst->nb[1] + (size_t) t * dst->nb[2]);
            for (int n = 0; n < N; n++)
                vec_dot(K, dr + n, 0, xw + (size_t) n * row_bytes, 0, qa, 0, 1);
        }
    };

    // Make expert e resident in this tensor's hot pool, packing it live on first touch (or loading from
    // the .orkpack if present). Evicts the LRU resident expert (freeing IOVA) when the pool is at cap.
    // Returns the slot, or nullptr if it could not be made resident (caller falls back to CPU).
    auto get_hot = [&](int e) -> ggml_backend_ork_context::ork_hot_slot * {
        const void * x = (const char *) src0->data + (size_t) e * src0->nb[2];
        auto it = hotmap.find(x);
        if (it != hotmap.end() && it->second.w) { it->second.last_use = ++ctx->moe_hot_tick; ctx->moe_hot_hits++; return &it->second; }
        // Global IOVA budget gate FIRST: the backbone wcache + hot experts share the 4 GiB window. If this
        // expert won't fit under the weight-budget AND evicting our own LRU wouldn't free enough, refuse
        // (route to CPU) — without freeing a resident hot expert or EFAULTing the MEM_CREATE ioctl. (When
        // at per-tensor cap an eviction below frees exactly K*N, so a same-shape swap always fits.)
        const size_t need = (size_t) K * N;
        const bool will_evict = hotmap.size() >= hot_K;
        const size_t after_evict = ctx->moe_hot_bytes - (will_evict ? need : 0);
        if (after_evict + need > hot_budget) return (ggml_backend_ork_context::ork_hot_slot *) nullptr;
        // admitted: evict LRU within THIS tensor's pool until under the per-tensor count cap
        while (hotmap.size() >= hot_K) {
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
        slot.bscale = std::move(bsc); slot.key = x; slot.last_use = ++ctx->moe_hot_tick;
        ctx->moe_hot_bytes += (size_t) K * N;
        if (ctx->moe_hot_bytes > ctx->moe_hot_peak) ctx->moe_hot_peak = ctx->moe_hot_bytes;
        ctx->moe_hot_hits++;
        auto res = hotmap.emplace(x, std::move(slot));
        return &res.first->second;
    };

    // Split active experts: those already resident, or that fit/evict into the per-tensor cap -> NPU.
    // The rest this step -> CPU. Prefer keeping already-resident experts (the hot set) on the NPU.
    std::vector<int> hot_e; std::vector<ggml_backend_ork_context::ork_hot_slot *> hot_s;
    for (auto & kv : buckets) {
        const int e = kv.first;
        const void * x = (const char *) src0->data + (size_t) e * src0->nb[2];
        bool resident = hotmap.count(x) && hotmap[x].w;
        // admit to NPU if resident, or if the pool has headroom / LRU room (cap enforced inside get_hot)
        ggml_backend_ork_context::ork_hot_slot * s = (resident || hotmap.size() < hot_K) ? get_hot(e) : nullptr;
        if (s) { hot_e.push_back(e); hot_s.push_back(s); }
        else   { cpu_expert(e, kv.second); }   // cold -> CPU (deferred; run threaded below)
    }

    // Fan the cold experts across a thread pool (each expert is independent). Pre-quantize all needed
    // token activations single-threaded first (qact vector grows; not thread-safe), then dot in parallel.
    if (!cold.empty()) {
        const double cd0 = ctx->profile ? ork_now_us() : 0;
        for (auto & ce : cold) for (auto & pr : *ce.second) quant_tok(pr.first, pr.second);   // populate qact
        const int n_cold = (int) cold.size();
        unsigned hw = std::thread::hardware_concurrency();
        int nthr = (int) (getenv("ORK_MOE_COLD_THREADS") ? atoi(getenv("ORK_MOE_COLD_THREADS")) : (hw ? hw/2 : 4));
        if (nthr < 1) nthr = 1; if (nthr > n_cold) nthr = n_cold;
        if (nthr <= 1) {
            for (auto & ce : cold) run_cold_expert(ce.first, *ce.second);
        } else {
            std::atomic<int> next(0);
            auto worker = [&]() { int i; while ((i = next.fetch_add(1)) < n_cold) run_cold_expert(cold[i].first, *cold[i].second); };
            std::vector<std::thread> th; th.reserve(nthr-1);
            for (int w = 0; w < nthr-1; w++) th.emplace_back(worker);
            worker();
            for (auto & t : th) t.join();
        }
        ctx->moe_cold_cpu += n_cold;
        if (ctx->profile) { ctx->moe_cold += ork_now_us() - cd0; ctx->moe_cold_calls += n_cold; }
    }

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
    const double ch0 = ctx->profile ? ork_now_us() : 0;
    int crc = ork_mm_run_chain_i8(ctx->npu, S, tasks.data());
    if (crc) { crc = 0; for (int x = 0; x < S && crc == 0; x++) crc = ork_mm_run_i8(ctx->npu, tasks[x].w, tasks[x].M, tasks[x].A, tasks[x].C); }
    if (ctx->profile) { ctx->moe_chain += ork_now_us() - ch0; ctx->moe_calls++; }
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

static enum ggml_status ggml_backend_ork_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_ork_context * ctx = (ggml_backend_ork_context *) backend->context;
    ctx->last_src1 = nullptr;
    if(getenv("ORK_VERBOSE"))fprintf(stderr, "[ORK] START graph_compute, %d nodes\n", cgraph->n_nodes); fflush(stderr);
    // QKV/gate-up fusion: implemented + correct, but measured SLOWER on RK3588 (decode 9.4->6.4
    // tok/s) — one wide multi-core matmul + strided scatter costs more than the 2 saved submits, i.e.
    // the NPU per-matmul cost scales with work, it's not a fixed floor fusion can amortize. Off by
    // default; opt in with ORK_FUSE=1 to experiment (may differ on larger models / tuned scatter).
    const int fuse = (ctx->qbits == 8) && (getenv("ORK_FUSE") != nullptr);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        switch (node->op) {
            case GGML_OP_MUL_MAT: {
                std::vector<struct ggml_tensor *> chain_nodes;
                ork_chain_type type = get_node_chain_type(ctx, node);
                
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
                        int target_qbits = ork_pick_qbits(ctx, node->src[0]->ne[0], node->src[0]->ne[1]);

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
    ctx->no_reuse = getenv("ORK_NOREUSE") != nullptr;
    ctx->no_cache = getenv("ORK_NOCACHE") != nullptr;
    ctx->hybrid = g_ork_hybrid_loading || getenv("ORK_HYBRID") != nullptr;
    {   // size-aware hybrid threshold (see ork_pick_qbits): weights with int8 footprint >= this go W4A4
        const char * t = getenv("ORK_HYBRID_W4_ABOVE_MB");
        ctx->hybrid_w4_above = (size_t)(t ? atoll(t) : 8) * 1024 * 1024;
    }
    ctx->hadamard = (ctx->qbits == 4) && getenv("ORK_HADAMARD") != nullptr;
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
static ggml_backend_buffer_type_t ggml_backend_ork_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cpu_buffer_type(); GGML_UNUSED(dev);
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
            if (strstr(name, "output") || strstr(name, "lm_head") || N > 16384) {
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
            bool pass_m_threshold = (M >= threshold || (M > 1 && (op->ne[2] > 1 || op->ne[3] > 1)));

            return pass_m_threshold &&
                   ggml_is_contiguous(src0) && ggml_is_contiguous(src1) &&
                   src1->type == GGML_TYPE_F32 &&
                   K % 32 == 0 && N % 64 == 0 &&           // K%32; N%64 satisfies both int8 (%32) and int4 (%64)
                   K >= 32 &&
                   src1->ne[2] % src0->ne[2] == 0 &&
                   src1->ne[3] % src0->ne[3] == 0 &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->to_float != NULL);
        }
        case GGML_OP_MUL_MAT_ID: {
            // MoE expert offload (group-by-expert, int8). EXPERIMENTAL — measured a net loss for both
            // decode (~60x) and prefill (~80x) on RK3588 (per-expert submit floor dominates), so it is
            // OFF by default. Enable with ORK_MOE_NPU=1 (legacy alias: ORK_NO_EXPERT_REPACK=1) — the same
            // flag must un-gate the repack-buffer exclusion in ggml-cpu/repack.cpp for experts to route.
            if (!getenv("ORK_MOE_NPU") && !getenv("ORK_NO_EXPERT_REPACK")) return false;
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
