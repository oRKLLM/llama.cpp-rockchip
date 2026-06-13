// ork-driver NPU matmul backend for ggml (Rockchip RK35xx).
//
// Modeled on the BLAS backend: a mul-mat-only accelerator that offloads GGML_OP_MUL_MAT to the
// Rockchip NPU via ork-driver and leaves all other ops to the CPU backend. First cut uses the fp16
// path (W16A16): weights are dequantized to fp16, activations converted F32->fp16, the matmul runs
// on the NPU, and the fp32 result is written straight into dst. Correct and accurate; weight packing
// happens per call (to be cached later). int8/int4 fast paths can be added on top.

#include "ggml-impl.h"
#include "ggml-ork.h"
#include "ggml-backend-impl.h"

#include <vector>
#include <cstring>

extern "C" {
#include "ork_npu.h"
}

struct ggml_backend_ork_context {
    ork_npu * npu = nullptr;
    std::vector<float>   f32;   // dequantized src0 plane [N*K]
    std::vector<ork_f16> bf;    // weights as fp16 B[K*N]
    std::vector<ork_f16> af;    // activations as fp16 A[M*K]
};

// dst = src0 x src1 :  src0 [K=ne00, N=ne01], src1 [K=ne10=ne00, M=ne11], dst [N, M] (row-major [M][N])
static bool ggml_backend_ork_mul_mat(ggml_backend_ork_context * ctx, struct ggml_tensor * dst) {
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
    ctx->bf .resize((size_t) K * N);
    ctx->af .resize((size_t) M * K);
    float   * f32 = ctx->f32.data();
    ork_f16 * bf  = ctx->bf.data();
    ork_f16 * af  = ctx->af.data();

    for (int64_t i13 = 0; i13 < ne13; i13++) {
        for (int64_t i12 = 0; i12 < ne12; i12++) {
            const int64_t i03 = i13/r3;
            const int64_t i02 = i12/r2;

            const char  * x = (const char *) src0->data + i02*nb02 + i03*nb03;   // [N][K], type
            const float * y = (const float *)((const char *) src1->data + i12*nb12 + i13*nb13); // [M][K] f32
                  float * d = (      float *)((      char *)  dst->data + i12*nb2  + i13*nb3);   // [M][N] f32

            // src0 -> fp32 [N][K]
            if (type == GGML_TYPE_F32) {
                for (int64_t n = 0; n < N; n++) memcpy(f32 + n*K, x + n*nb01, (size_t) K*sizeof(float));
            } else {
                for (int64_t n = 0; n < N; n++) to_float((const char *) x + n*nb01, f32 + n*K, K);
            }
            // -> fp16 B[K][N]   (B[k][n] = src0[n][k])
            for (int n = 0; n < N; n++)
                for (int k = 0; k < K; k++) bf[(size_t) k*N + n] = (ork_f16) f32[(size_t) n*K + k];
            // src1 -> fp16 A[M][K]
            for (int64_t i = 0; i < (int64_t) M*K; i++) af[i] = (ork_f16) y[i];

            ork_w * w = ork_mm_pack(ctx->npu, K, N, bf);
            if (!w) return false;
            int rc = ork_mm_run(ctx->npu, w, M, af, d);
            ork_w_free(w);
            if (rc) return false;
        }
    }
    return true;
}

// backend interface

static const char * ggml_backend_ork_get_name(ggml_backend_t backend) { return "ORK"; GGML_UNUSED(backend); }

static void ggml_backend_ork_free(ggml_backend_t backend) {
    ggml_backend_ork_context * ctx = (ggml_backend_ork_context *) backend->context;
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
                if (!ggml_backend_ork_mul_mat(ctx, node)) return GGML_STATUS_FAILED;
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
                   K % 32 == 0 && N % 16 == 0 &&           // ork fp16 path: K%32, N%16
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
