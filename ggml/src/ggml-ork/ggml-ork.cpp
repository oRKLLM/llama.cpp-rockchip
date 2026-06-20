// ork-driver NPU matmul backend for ggml (Rockchip RK35xx).
//
// Modeled on the BLAS backend: a mul-mat-only accelerator that offloads GGML_OP_MUL_MAT to the
// Rockchip NPU via ork-driver and leaves all other ops to the CPU backend. Uses the int8 (W8A8)
// path: weights are dequantized then per-channel int8-quantized and packed once (cached, NPU-
// resident); activations are per-row int8-quantized each call; the NPU computes the int32 product
// which is dequantized (aScale[m]*bScale[n]) into the fp32 dst. ~1% vs fp32 on real weights, half
// the weight bytes of fp16. (int4/W4A4 + per-group scales is the next step down.)

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
    // reuse the quantized activation across consecutive matmuls that share the same src1 input
    // (Q/K/V off the normed hidden state; FFN gate/up off the same x) — skips redundant per-matmul
    // activation int8-quant. Holds for the data in ctx->ai/as while last_* matches.
    const void * last_src1 = nullptr; int last_M = 0, last_K = 0;
    // ORK_PROFILE=1: accumulate where time goes, report on free (split decode M=1 vs prefill M>1)
    double t_quant = 0, t_run = 0, t_deq = 0; long n_mm = 0; int profile = 0;
    double t_run_dec = 0, t_run_pf = 0; long n_dec = 0, n_pf = 0, m_pf = 0;
};
static ggml_backend_ork_context * g_ork_ctx = nullptr;
static bool g_ork_hybrid_loading = false;

void ggml_backend_ork_set_hybrid(bool use_hybrid) {
    g_ork_hybrid_loading = use_hybrid;
}
static inline double ork_now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec*1e6 + t.tv_nsec*1e-3; }

// dst = src0 x src1 :  src0 [K=ne00, N=ne01], src1 [K=ne10=ne00, M=ne11], dst [N, M] (row-major [M][N])
static bool ggml_backend_ork_mul_mat_i8(ggml_backend_ork_context * ctx, struct ggml_tensor * dst) {
    fprintf(stderr, "[ORK] START mul_mat_i8\n"); fflush(stderr);
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

        ctx->ai .resize((size_t) chunk_size * M * K);
        ctx->as .resize((size_t) chunk_size * M);
        ctx->ci .resize((size_t) chunk_size * M * N);

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
            }
            const ork_weight & ow = it->second;

            // activation: per-row int8 quant
            int8_t * ar = ai + t * M * K;
            float * asr = as + t * M;
            for (int m = 0; m < M; m++) {
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
            }
            tasks.push_back({
                ow.w,
                M,
                ai + t * M * K,
                ci + t * M * N
            });
        }

        const double t1 = ctx->profile ? ork_now_us() : 0;

        fprintf(stderr, "[ORK] i8 chain: M=%d, tasks=%zu (S=%d, K=%d, N=%d)\n", M, tasks.size(), S, K, N);
        fflush(stderr);
        int ok = ork_mm_run_chain_i8(ctx->npu, tasks.size(), tasks.data());
        if (ok != 0) {
            // Fallback to sequential single-task run
            for (size_t t = 0; t < tasks.size(); t++) {
                if (ork_mm_run_i8(ctx->npu, tasks[t].w, tasks[t].M, tasks[t].A, tasks[t].C)) {
                    return false;
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
            const float * asr = as + t * M;
            const int32_t * ctr = ci + t * M * N;

            for (int m = 0; m < M; m++) {
                const float rs = asr[m];
                const int32_t * cr = ctr + (size_t) m*N;
                float * dr = d + (size_t) m*N;
                for (int n = 0; n < N; n++) dr[n] = rs * bs[n] * (float) cr[n];
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
    fprintf(stderr, "[ORK] START mul_mat_i4\n"); fflush(stderr);
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

    ctx->f32.resize((size_t) N * K);
    ctx->bi .resize((size_t) K * N);
    ctx->ai .resize((size_t) M * K);
    ctx->as .resize((size_t) M * NG);                 // per-row, per-group activation scale
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

            // activations: per-row, per-group int4 quant
            for (int m = 0; m < M; m++)
                for (int g = 0; g < NG; g++) {
                    float mx = 1e-9f;
                    for (int j = 0; j < G; j++) { float v = fabsf(y[(size_t) m*K + g*G + j]); if (v > mx) mx = v; }
                    float s = mx / 7.0f; as[(size_t) m*NG + g] = s;
                    for (int j = 0; j < G; j++) {
                        int q = (int) lrintf(y[(size_t) m*K + g*G + j] / s);
                        ai[(size_t) m*K + g*G + j] = (int8_t) (q > 7 ? 7 : q < -8 ? -8 : q);
                    }
                }

            // grouped run dequantizes per group into the fp32 dst directly
            fprintf(stderr, "[ORK] i4 grouped: M=%d, K=%d, N=%d, G=%d\n", M, K, N, G);
            fflush(stderr);
            if (ork_mm_run_i4_grouped(ctx->npu, ow.w, M, ai, as, ow.bscale.data(), d)) return false;
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
    fprintf(stderr, "[ORK] START mul_mat_i4_hadamard\n"); fflush(stderr);
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type type = src0->type;
    const int K = (int) ne00, N = (int) ne01, M = (int) ne11;
    const int b = K & (-K);                          // largest power-of-2 block dividing K (full FWHT if K is pow2)

    const int64_t r2 = ne12/ne02, r3 = ne13/ne03;
    const auto * tt = ggml_get_type_traits(type);
    ggml_to_float_t const to_float = tt->to_float;

    ctx->f32.resize((size_t) N * K);
    ctx->bi .resize((size_t) K * N);
    ctx->ai .resize((size_t) M * K);
    ctx->as .resize((size_t) M);
    ctx->ci .resize((size_t) M * N);
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

            // activations: rotate each row (A·R), per-row int4 quant
            for (int m = 0; m < M; m++) {
                memcpy(arow, y + (size_t) m*K, (size_t) K*sizeof(float));
                for (int off = 0; off < K; off += b) {
                    ork_fwht_norm(arow + off, b);
                }
                float mx = 1e-9f;
                for (int k = 0; k < K; k++) { float v = fabsf(arow[k]); if (v > mx) mx = v; }
                float s = mx / 7.0f; as[m] = s;
                for (int k = 0; k < K; k++) {
                    int q = (int) lrintf(arow[k] / s);
                    ai[(size_t) m*K + k] = (int8_t) (q > 7 ? 7 : q < -8 ? -8 : q);
                }
            }

            ork_mm_task_i4 task = { ow.w, M, ai, ci };
            fprintf(stderr, "[ORK] i4 chain: M=%d, K=%d, N=%d\n", M, K, N);
            fflush(stderr);
            if (ork_mm_run_i4(ctx->npu, task.w, task.M, task.A, task.C)) return false;    // full-K single submit, int32 C
            for (int m = 0; m < M; m++)
                for (int n = 0; n < N; n++)
                    d[(size_t) m*N + n] = (float) ci[(size_t) m*N + n] * as[m] * ow.bscale[n];
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
    fprintf(stderr, "[ORK] START mul_mat_group_i8 ng=%d\n", ng); fflush(stderr);
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

    ctx->ai.resize((size_t) M*K); ctx->as.resize(M); ctx->ci.resize((size_t) M*Ntot);
    ctx->last_src1 = nullptr;                            // group overwrote ctx->ai — kill reuse cache
    int8_t * ai = ctx->ai.data(); float * as = ctx->as.data(); int32_t * ci = ctx->ci.data();
    const float * y = (const float *) src1->data;
    for (int m = 0; m < M; m++) {                        // quantize the shared activation once
        const float * yr = y + (size_t) m*K; int8_t * ar = ai + (size_t) m*K;
        float mx = 1e-9f;
        for (int k = 0; k < K; k++) { float v = fabsf(yr[k]); mx = v > mx ? v : mx; }
        as[m] = mx / 127.0f; const float inv = 127.0f / mx;
        for (int k = 0; k < K; k++) { float q = yr[k]*inv; int qi = (int)(q + copysignf(0.5f, q));
            ar[k] = (int8_t)(qi > 127 ? 127 : qi < -127 ? -127 : qi); }
    }
    const double t1 = ctx->profile ? ork_now_us() : 0;
    fprintf(stderr, "[ORK] mul_mat_id i8: M=%d, K=%d, N=%d (ng=%d)\n", M, K, Ntot, ng);
    fflush(stderr);
    if (ork_mm_run_i8(ctx->npu, ow.w, M, ai, ci)) return false;     // ONE submit for all ng matmuls
    const double t2 = ctx->profile ? ork_now_us() : 0;

    const float * bs = ow.bscale.data();                 // scatter+dequant into each dst
    for (int i = 0; i < ng; i++) {
        const int Ni = (int) g[i]->src[0]->ne[1]; const int o = off[i];
        float * dbase = (float *) g[i]->data;
        for (int m = 0; m < M; m++) { const float rs = as[m]; const int32_t * cr = ci + (size_t) m*Ntot + o;
            float * dr = dbase + (size_t) m*Ni; for (int n = 0; n < Ni; n++) dr[n] = rs * bs[o+n] * (float) cr[n]; }
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
    for (auto & kv : ctx->wcache) ork_w_free(kv.second.w);
    if (ctx->npu) ork_npu_free(ctx->npu);
    delete ctx;
    g_ork_ctx = nullptr;
    delete backend;
}

enum ork_chain_type {
    ORK_CHAIN_NONE,
    ORK_CHAIN_I8
};

static ork_chain_type get_node_chain_type(ggml_backend_ork_context * ctx, struct ggml_tensor * node) {
    if (node->op != GGML_OP_MUL_MAT) {
        return ORK_CHAIN_NONE;
    }
    const char * name = node->src[0]->name;
    bool is_ffn = strstr(name, "ffn_") || strstr(name, "expert");
    bool is_attn = strstr(name, "attn_q") || strstr(name, "attn_k") || strstr(name, "attn_v") || strstr(name, "attn_output");
    
    int target_qbits = ctx->qbits;
    if (ctx->hybrid) {
        if (is_ffn) target_qbits = 4;
        else if (is_attn) target_qbits = 8;
    }
    if (target_qbits == 8) {
        return ORK_CHAIN_I8;
    }
    return ORK_CHAIN_NONE;
}

static bool ggml_backend_ork_mul_mat_chain_i8(ggml_backend_ork_context * ctx, struct ggml_tensor ** nodes, int count) {
    fprintf(stderr, "[ORK] START mul_mat_chain_i8, count=%d\n", count); fflush(stderr);

    size_t total_ai_size = 0;
    size_t total_as_size = 0;
    size_t total_ci_size = 0;
    for (int i = 0; i < count; i++) {
        struct ggml_tensor * dst = nodes[i];
        int K = dst->src[0]->ne[0];
        int N = dst->src[0]->ne[1];
        int M = dst->src[1]->ne[1];
        total_ai_size += (size_t)M * K;
        total_as_size += (size_t)M;
        total_ci_size += (size_t)M * N;
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
            for (int m = 0; m < M; m++) {
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
            }
            chain_act_cache[src1->data] = {task_A, task_as};
            ai_offset += (size_t)M * K;
            as_offset += (size_t)M;
        }

        tasks.push_back({
            ow.w,
            M,
            task_A,
            ci_base + ci_offset
        });

        ci_offset += (size_t)M * N;
    }

    const double t1 = ctx->profile ? ork_now_us() : 0;

    fprintf(stderr, "[ORK] i8 chain submit: tasks=%zu\n", tasks.size());
    fflush(stderr);
    int ok = ork_mm_run_chain_i8(ctx->npu, tasks.size(), tasks.data());
    if (ok != 0) {
        // Fallback to sequential single-task run
        fprintf(stderr, "[ORK] i8 chain failed (%d), falling back to sequential\n", ok); fflush(stderr);
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

        for (int m = 0; m < M; m++) {
            const float rs = task_as[m];
            const int32_t * cr = ctr + (size_t) m*N;
            float * dr = d + (size_t) m*N;
            for (int n = 0; n < N; n++) dr[n] = rs * bs[n] * (float) cr[n];
        }

        ci_offset += (size_t)M * N;
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



static enum ggml_status ggml_backend_ork_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_ork_context * ctx = (ggml_backend_ork_context *) backend->context;
    fprintf(stderr, "[ORK] START graph_compute, %d nodes\n", cgraph->n_nodes); fflush(stderr);
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
                        bool is_ffn = strstr(name, "ffn_") || strstr(name, "expert");
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
    ctx->qbits = (q && q[0] == '8') ? 8 : 4;
    ctx->profile = getenv("ORK_PROFILE") != nullptr;
    ctx->no_reuse = getenv("ORK_NOREUSE") != nullptr;
    ctx->no_cache = getenv("ORK_NOCACHE") != nullptr;
    ctx->hybrid = g_ork_hybrid_loading || getenv("ORK_HYBRID") != nullptr;
    ctx->hadamard = (ctx->qbits == 4) && getenv("ORK_HADAMARD") != nullptr;
    GGML_LOG_INFO("%s: ork backend ready (%sW%dA%d%s)\n", __func__,
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
            if (strstr(name, "output") || strstr(name, "lm_head")) {
                return false;
            }
            // Measured (RK3588, Qwen3-1.7B-w8a8): the ~365us/matmul NPU submit floor makes per-token
            // DECODE (M=1) a net LOSS vs CPU (4.7 vs 9.4 tok/s) — ~197 submits/token at 365us each is
            // ~72ms before any compute benefit, and M=1 matmuls are tiny. PREFILL (large M) is the
            // opposite: M>1 amortizes the floor over many rows, so NPU wins (39.6 vs 13.6 tok/s).
            // Gate on M (the token/batch dim) ONLY — NOT N. The old `M>=min || N>=min` always passed
            // because every weight has a large N, dragging M=1 decode onto the NPU. ORK_MINM tunes it.
            static const int min_m = getenv("ORK_MINM") ? atoi(getenv("ORK_MINM")) : 32;
            int target_qbits = g_ork_ctx ? g_ork_ctx->qbits : ((getenv("ORK_QUANT") && getenv("ORK_QUANT")[0] == '8') ? 8 : 4);
            bool hybrid = g_ork_ctx ? g_ork_ctx->hybrid : (g_ork_hybrid_loading || getenv("ORK_HYBRID") != nullptr);
            const char * name_src = src0->name;
            bool is_expert = strstr(name_src, "expert") != nullptr;
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
            int threshold = is_expert ? 1 : min_m;
            bool pass_m_threshold = (M >= threshold || op->ne[2] > 1 || op->ne[3] > 1);

            return pass_m_threshold &&
                   ggml_is_contiguous(src0) && ggml_is_contiguous(src1) &&
                   src1->type == GGML_TYPE_F32 &&
                   K % 32 == 0 && N % 64 == 0 &&           // K%32; N%64 satisfies both int8 (%32) and int4 (%64)
                   K >= 32 &&
                   src1->ne[2] % src0->ne[2] == 0 &&
                   src1->ne[3] % src0->ne[3] == 0 &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->to_float != NULL);
        }
        default:
            return false;
    }
    GGML_UNUSED(dev);
}

static bool ggml_backend_ork_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft); GGML_UNUSED(dev);
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
    /* .offload_op           = */ NULL,
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
