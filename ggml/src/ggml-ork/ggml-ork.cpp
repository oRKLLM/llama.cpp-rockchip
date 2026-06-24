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
// ===================================================================================

#include "ggml-impl.h"
#include "ggml-ork.h"
#include "ggml-backend-impl.h"

#include <vector>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <ctime>
#include <utility>
#include <unordered_map>
#include <deque>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

extern "C" {
#include "ork_npu.h"
}

// a packed quantized weight + its scales, kept NPU-resident and reused.
//   int8 (W8A8):  gsize==0, bscale [N]            (per output channel)
//   int4 (W4A4):  gsize==G,  bscale [(K/G)*N]      (per K-group, per channel)
struct ork_weight {
    ork_w * w = nullptr;
    std::vector<float> bscale;
    int gsize = 0;
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
    // MoE expert weights are too numerous to keep ALL packed NPU-resident (the IOMMU exhausts ~2k).
    // Fixed pool PER SHAPE: a bounded set of slots allocated once, reused round-robin via repack-in-place
    // (NO alloc/free → no IOMMU fragmentation). Dense/attn weights stay in wcache (resident forever).
    std::unordered_map<int64_t, std::deque<ork_moe_slot>> moe_pools;  // shape (K<<32|N) -> slots
    std::unordered_map<int64_t, size_t>                   moe_rr;     // shape -> round-robin cursor
    std::unordered_map<const void *, ork_moe_slot *>      moe_loc;    // expert host ptr -> its slot
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
};
static ggml_backend_ork_context * g_ork_ctx = nullptr;
static bool g_ork_hybrid_loading = false;

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

            // weight: check cache / pack
            auto it = ctx->wcache.find(x);
            if (it == ctx->wcache.end()) {
                float * f32 = ctx->f32.data();
                int8_t * bi = ctx->bi.data();
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
                ow.w = ork_mm_pack_i8(ctx->npu, K, N, bi);
                if (!ow.w) return false;
                it = ctx->wcache.emplace(x, std::move(ow)).first;
                if(getenv("ORK_VERBOSE"))fprintf(stderr, "[ORK DEBUG] packed weight, wcache size=%zu, K=%d, N=%d, x=%p\n", ctx->wcache.size(), K, N, x);
            }
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
    for (auto & kv : ctx->wcache) ork_w_free(kv.second.w);
    for (auto & p : ctx->moe_pools) for (auto & s : p.second) if (s.w) ork_w_free(s.w);   // MoE expert pool
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
            ow.w = ork_mm_pack_i8(ctx->npu, K, N, bi);
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

    // Resident-expert budget PER SHAPE (count). Slots are allocated once then reused round-robin via
    // repack-in-place (NO alloc/free) — keeps the packed set bounded WITHOUT fragmenting the NPU IOMMU
    // (which is what killed the free+realloc LRU). Tunable via ORK_MOE_CACHE.
    static const size_t moe_cap = getenv("ORK_MOE_CACHE") ? (size_t) atoi(getenv("ORK_MOE_CACHE")) : 384;
    const int64_t shape = ((int64_t) K << 32) | (uint32_t) N;
    auto pack_expert = [&](const char * x) -> ork_moe_slot * {
        auto loc = ctx->moe_loc.find(x);
        if (loc != ctx->moe_loc.end() && loc->second->key == x) return loc->second;   // resident hit
        // FUSED Q4_K->int8: dequant each expert channel into ork-driver's reused K-scratch + NEON
        // quant+tile in one pass — the full f32[N][K] is never materialized, killing its DRAM round-trip.
        // pack/repack_i8_dequant call back into ork_moe_deq_row (ggml to_float) per channel, fill bscale[].
        std::vector<float> bsc(N);
        ork_moe_deq_ctx dq = { x, (size_t) src0->nb[1], to_float, type == GGML_TYPE_F32 };
        const double d1 = ctx->profile ? ork_now_us() : 0;
        std::deque<ork_moe_slot> & pool = ctx->moe_pools[shape];
        ork_moe_slot * s;
        if (pool.size() < moe_cap) {                       // grow pool: one-time alloc for this slot
            pool.emplace_back();
            s = &pool.back();
            s->w = ork_mm_pack_i8_dequant(ctx->npu, K, N, ork_moe_deq_row, &dq, bsc.data());
            if (!s->w) { pool.pop_back(); return (ork_moe_slot *) nullptr; }
        } else {                                           // reuse a slot in place (no alloc/free)
            s = &pool[ctx->moe_rr[shape]++ % moe_cap];
            if (s->key) ctx->moe_loc.erase(s->key);        // evict previous occupant
            if (ork_mm_repack_i8_dequant(ctx->npu, s->w, K, N, ork_moe_deq_row, &dq, bsc.data()) != 0) return (ork_moe_slot *) nullptr;
        }
        if (ctx->profile) ctx->moe_quant += ork_now_us() - d1;   // fused dequant+quant+tile (moe_deq now folded in)
        s->bscale = std::move(bsc);
        s->key = x;
        ctx->moe_loc[x] = s;
        return s;
    };

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

    // GROUP TOKENS BY EXPERT: bucket every (token t, slot j) under expert ids[j,t], then run ONE matmul
    // per active expert over all its tokens (M=count). For PREFILL this amortizes the submit floor + the
    // expert-weight read across many tokens (the real MoE-on-NPU win); for decode (M=1) it degenerates
    // to one matmul per selected expert.
    std::unordered_map<int, std::vector<std::pair<int,int>>> buckets;
    for (int t = 0; t < n_tokens; t++) {
        const int32_t * idp = (const int32_t *)((const char *) ids->data + (size_t) t * ids->nb[1]);
        for (int j = 0; j < n_used; j++) buckets[idp[j]].push_back(std::make_pair(t, j));
    }
    // CHAIN all active experts of this projection into ONE submit (ork_mm_run_chain_i8): each expert is
    // one chain task with M=count rows (the chain M-tiles internally and uses each expert's Bf full-K
    // weight). This collapses the per-expert submit floor — the MoE-prefill win. Pack each expert into a
    // contiguous row block of bigA/bigC. All active experts (<=128/projection << pool cap) stay resident
    // through the submit, so round-robin repack never evicts a weight the chain still needs.
    std::vector<std::pair<int, std::vector<std::pair<int,int>>*>> active;
    size_t total_rows = 0;
    for (auto & kv : buckets) { active.push_back(std::make_pair(kv.first, &kv.second)); total_rows += kv.second.size(); }
    const int S = (int) active.size();
    if (S == 0) { if (ctx->profile) { ctx->t_run += ork_now_us() - t0; ctx->n_mm++; } return true; }

    std::vector<int8_t>          bigA((size_t) total_rows * K);
    std::vector<int32_t>         bigC((size_t) total_rows * N);
    std::vector<float>           as_row(total_rows);
    std::vector<ork_mm_task_i8>  tasks(S);
    std::vector<ork_moe_slot *>  slots(S);
    std::vector<size_t>          offs(S);

    size_t off = 0;
    for (int x = 0; x < S; x++) {
        const int e = active[x].first; std::vector<std::pair<int,int>> & ent = *active[x].second;
        const int cnt = (int) ent.size();
        const double pk0 = ctx->profile ? ork_now_us() : 0;
        ork_moe_slot * s = pack_expert((const char *) src0->data + (size_t) e * src0->nb[2]);
        if (ctx->profile) ctx->moe_pack += ork_now_us() - pk0;
        if (!s) { if(getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK] mul_mat_id PACK FAIL expert=%d K=%d N=%d\n", e, K, N); return false; }
        slots[x] = s; offs[x] = off;
        const double g0 = ctx->profile ? ork_now_us() : 0;
        int8_t * Ae = bigA.data() + off * K;
        for (int i = 0; i < cnt; i++) {
            const int t = ent[i].first, j = ent[i].second;
            int8_t * ar = Ae + (size_t) i * K;
            if (bcast) { memcpy(ar, A_all.data() + (size_t) t * K, (size_t) K); as_row[off + i] = as_all[t]; }
            else {
                const float * y = (const float *)((const char *) src1->data + (size_t) j*src1->nb[1] + (size_t) t*src1->nb[2]);
                float mx=1e-9f; for (int k=0;k<K;k++){ float v=fabsf(y[k]); if(v>mx)mx=v; }
                as_row[off + i]=mx/127.0f; float ainv=127.0f/mx;
                for (int k=0;k<K;k++){ int qi=(int)lrintf(y[k]*ainv); ar[k]=(int8_t)(qi>127?127:qi<-127?-127:qi); }
            }
        }
        if (ctx->profile) ctx->moe_gather += ork_now_us() - g0;
        tasks[x].w = s->w; tasks[x].M = cnt; tasks[x].A = Ae; tasks[x].C = bigC.data() + off * N;
        off += cnt;
    }

    const double ch0 = ctx->profile ? ork_now_us() : 0;
    int crc = ork_mm_run_chain_i8(ctx->npu, S, tasks.data());
    if (ctx->profile) { ctx->moe_chain += ork_now_us() - ch0; ctx->moe_calls++; }
    if (crc) { if(getenv("ORK_VERBOSE")) fprintf(stderr, "[ORK] mul_mat_id run_chain_i8 FAIL rc=%d S=%d rows=%zu K=%d N=%d\n", crc, S, total_rows, K, N); return false; }

    const double sc0 = ctx->profile ? ork_now_us() : 0;
    for (int x = 0; x < S; x++) {
        std::vector<std::pair<int,int>> & ent = *active[x].second;
        const int cnt = (int) ent.size(); const size_t o = offs[x];
        const float * bs = slots[x]->bscale.data();
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
    const char * q = getenv("ORK_QUANT");
    ctx->qbits = (q && q[0] == '4') ? 4 : 8;   // ORK_QUANT=4 -> W4A4; default (unset/8) -> W8A8
    ctx->profile = getenv("ORK_PROFILE") != nullptr;
    ctx->no_reuse = getenv("ORK_NOREUSE") != nullptr;
    ctx->no_cache = getenv("ORK_NOCACHE") != nullptr;
    ctx->hybrid = g_ork_hybrid_loading || getenv("ORK_HYBRID") != nullptr;
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
