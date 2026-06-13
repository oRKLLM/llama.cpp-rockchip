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
#include <cstring>
#include <cmath>
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
    std::vector<float>    f32;   // dequantized src0 plane [N*K] (cache-miss scratch)
    std::vector<int8_t>   bi;    // weights quantized int8 B[K*N] (cache-miss scratch)
    std::vector<int8_t>   ai;    // activations quantized int8 A[M*K]
    std::vector<float>    as;    // per-row activation scale [M]
    std::vector<int32_t>  ci;    // int32 matmul result [M*N] before dequant
    // model weights are constant during inference, so pack+quantize each once (NPU-resident) and
    // reuse, keyed by the weight plane's host pointer. The transformer pattern ork-driver is for.
    std::unordered_map<const void *, ork_weight> wcache;
};

// dst = src0 x src1 :  src0 [K=ne00, N=ne01], src1 [K=ne10=ne00, M=ne11], dst [N, M] (row-major [M][N])
static bool ggml_backend_ork_mul_mat_i8(ggml_backend_ork_context * ctx, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type type = src0->type;
    const int K = (int) ne00, N = (int) ne01, M = (int) ne11;

    const int64_t r2 = ne12/ne02;        // broadcast factors (e.g. GQA / attention)
    const int64_t r3 = ne13/ne03;

    const auto * tt = ggml_get_type_traits(type);
    ggml_to_float_t const to_float = tt->to_float;

    ctx->f32.resize((size_t) N * K);
    ctx->bi .resize((size_t) K * N);
    ctx->ai .resize((size_t) M * K);
    ctx->as .resize((size_t) M);
    ctx->ci .resize((size_t) M * N);
    float   * f32 = ctx->f32.data();
    int8_t  * bi  = ctx->bi.data();
    int8_t  * ai  = ctx->ai.data();
    float   * as  = ctx->as.data();
    int32_t * ci  = ctx->ci.data();

    for (int64_t i13 = 0; i13 < ne13; i13++) {
        for (int64_t i12 = 0; i12 < ne12; i12++) {
            const int64_t i03 = i13/r3;
            const int64_t i02 = i12/r2;

            const char  * x = (const char *) src0->data + i02*nb02 + i03*nb03;   // [N][K], type
            const float * y = (const float *)((const char *) src1->data + i12*nb12 + i13*nb13); // [M][K] f32
                  float * d = (      float *)((      char *)  dst->data + i12*nb2  + i13*nb3);   // [M][N] f32

            // weight: dequant -> per-channel int8 quant -> pack, ONCE (cached, NPU-resident)
            auto it = ctx->wcache.find(x);
            if (it == ctx->wcache.end()) {
                if (type == GGML_TYPE_F32) {
                    for (int64_t n = 0; n < N; n++) memcpy(f32 + n*K, x + n*nb01, (size_t) K*sizeof(float));
                } else {
                    for (int64_t n = 0; n < N; n++) to_float((const char *) x + n*nb01, f32 + n*K, K);
                }
                ork_weight ow; ow.bscale.resize(N);
                for (int n = 0; n < N; n++) {
                    float mx = 1e-9f;
                    for (int k = 0; k < K; k++) { float v = fabsf(f32[(size_t) n*K + k]); if (v > mx) mx = v; }
                    float s = mx / 127.0f; ow.bscale[n] = s;
                    for (int k = 0; k < K; k++) {           // B[k][n] = src0[n][k]
                        int q = (int) lrintf(f32[(size_t) n*K + k] / s);
                        bi[(size_t) k*N + n] = (int8_t) (q > 127 ? 127 : q < -127 ? -127 : q);
                    }
                }
                ow.w = ork_mm_pack_i8(ctx->npu, K, N, bi);
                if (!ow.w) return false;
                it = ctx->wcache.emplace(x, std::move(ow)).first;
            }
            const ork_weight & ow = it->second;

            // activation: per-row int8 quant. SIMD-vectorizable inner loops (precomputed inverse
            // scale + branchless round, no per-element divide or lrintf). NOT threaded: the per-call
            // work is small relative to NPU submit latency, and OpenMP region overhead net-hurt here.
            for (int m = 0; m < M; m++) {
                const float * yr = y + (size_t) m*K;
                int8_t * ar = ai + (size_t) m*K;
                float mx = 1e-9f;
                for (int k = 0; k < K; k++) { float v = fabsf(yr[k]); mx = v > mx ? v : mx; }
                as[m] = mx / 127.0f;
                const float inv = 127.0f / mx;
                for (int k = 0; k < K; k++) {
                    float q = yr[k] * inv;
                    int qi = (int) (q + copysignf(0.5f, q));
                    ar[k] = (int8_t) (qi > 127 ? 127 : qi < -127 ? -127 : qi);
                }
            }

            if (ork_mm_run_i8(ctx->npu, ow.w, M, ai, ci)) return false;

            // dequant int32 -> fp32 dst: d[m][n] = aScale[m]*bScale[n]*C[m][n] (SIMD over n)
            const float * bs = ow.bscale.data();
            for (int m = 0; m < M; m++) {
                const float rs = as[m];
                const int32_t * cr = ci + (size_t) m*N;
                float * dr = d + (size_t) m*N;
                for (int n = 0; n < N; n++) dr[n] = rs * bs[n] * (float) cr[n];
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
            if (ork_mm_run_i4_grouped(ctx->npu, ow.w, M, ai, as, ow.bscale.data(), d)) return false;
        }
    }
    return true;
}

// backend interface

static const char * ggml_backend_ork_get_name(ggml_backend_t backend) { return "ORK"; GGML_UNUSED(backend); }

static void ggml_backend_ork_free(ggml_backend_t backend) {
    ggml_backend_ork_context * ctx = (ggml_backend_ork_context *) backend->context;
    for (auto & kv : ctx->wcache) ork_w_free(kv.second.w);
    if (ctx->npu) ork_npu_free(ctx->npu);
    delete ctx;
    delete backend;
}

static enum ggml_status ggml_backend_ork_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_ork_context * ctx = (ggml_backend_ork_context *) backend->context;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        switch (node->op) {
            case GGML_OP_MUL_MAT:
                if (ctx->qbits == 4 ? !ggml_backend_ork_mul_mat_i4(ctx, node)
                                    : !ggml_backend_ork_mul_mat_i8(ctx, node)) return GGML_STATUS_FAILED;
                break;
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
    /* .get_name           = */ ggml_backend_ork_get_name,
    /* .free               = */ ggml_backend_ork_free,
    /* .set_tensor_async   = */ NULL,
    /* .get_tensor_async   = */ NULL,
    /* .cpy_tensor_async   = */ NULL,
    /* .synchronize        = */ NULL,
    /* .graph_plan_create  = */ NULL,
    /* .graph_plan_free    = */ NULL,
    /* .graph_plan_update  = */ NULL,
    /* .graph_plan_compute = */ NULL,
    /* .graph_compute      = */ ggml_backend_ork_graph_compute,
    /* .event_record       = */ NULL,
    /* .event_wait         = */ NULL,
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
    const char * q = getenv("ORK_QUANT");
    ctx->qbits = (q && q[0] == '4') ? 4 : 8;
    GGML_LOG_INFO("%s: ork backend ready (W%dA%d)\n", __func__, ctx->qbits, ctx->qbits);
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
static void ggml_backend_ork_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) { *free = 0; *total = 0; GGML_UNUSED(dev); }
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
            const int64_t min_batch = 32;                 // NPU only pays off on large matmuls
            return ggml_is_contiguous(src0) && ggml_is_contiguous(src1) &&
                   src1->type == GGML_TYPE_F32 &&
                   K % 32 == 0 && N % 64 == 0 &&           // K%32; N%64 satisfies both int8 (%32) and int4 (%64)
                   (M >= min_batch || N >= min_batch) && K >= min_batch &&
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
    /* .get_host_buffer_type = */ NULL,
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
