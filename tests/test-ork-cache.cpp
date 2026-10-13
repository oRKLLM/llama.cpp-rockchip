#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <cmath>
#include <string>

int main() {
    // load and enumerate backends
    ggml_backend_load_all();

    // Find the ORK backend
    ggml_backend_dev_t ork_dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (std::string(ggml_backend_dev_name(dev)) == "ORK") {
            ork_dev = dev;
            break;
        }
    }

    if (!ork_dev) {
        printf("ORK backend not found or not compiled. Skipping test.\n");
        return 0;
    }

    ggml_backend_t ork_backend = ggml_backend_dev_init(ork_dev, NULL);
    if (!ork_backend) {
        printf("Failed to initialize ORK backend. Skipping test.\n");
        return 0;
    }

    printf("ORK backend initialized successfully.\n");

    // Initialize GGML context
    struct ggml_init_params params = {
        /* .mem_size   = */ 1024*1024,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true,
    };
    struct ggml_context * ctx = ggml_init(params);

    int K = 128;
    int N = 128;
    int M = 1;

    // Create weight tensor W (N x K) and input tensor X (K x M)
    struct ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    struct ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor * dst = ggml_mul_mat(ctx, W, X);

    // Allocate backend buffers
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, ork_backend);
    assert(buf != nullptr);

    // Set weights of W to constant 0.5f
    std::vector<float> W_data(K * N, 0.5f);
    ggml_backend_tensor_set(W, W_data.data(), 0, W_data.size() * sizeof(float));

    // Run 1: Set X to 1.0f
    std::vector<float> X_data1(K * M, 1.0f);
    ggml_backend_tensor_set(X, X_data1.data(), 0, X_data1.size() * sizeof(float));

    // Build compute graph
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(gf, dst);

    // Compute Run 1
    ggml_backend_graph_compute(ork_backend, gf);

    // Read result 1
    std::vector<float> dst_data1(N * M);
    ggml_backend_tensor_get(dst, dst_data1.data(), 0, dst_data1.size() * sizeof(float));

    printf("Run 1 (X = 1.0): dst[0] = %f\n", dst_data1[0]);
    // W_data is 0.5, X is 1.0, so dst[i] should be K * 0.5 * 1.0 = 128 * 0.5 * 1.0 = 64.0
    float expected1 = K * 0.5f * 1.0f;
    printf("Expected Run 1: %f\n", expected1);

    // Run 2: Change X to 2.0f
    std::vector<float> X_data2(K * M, 2.0f);
    ggml_backend_tensor_set(X, X_data2.data(), 0, X_data2.size() * sizeof(float));

    // Compute Run 2 (reuses the same graph, backend state, and activation tensors)
    ggml_backend_graph_compute(ork_backend, gf);

    // Read result 2
    std::vector<float> dst_data2(N * M);
    ggml_backend_tensor_get(dst, dst_data2.data(), 0, dst_data2.size() * sizeof(float));

    printf("Run 2 (X = 2.0): dst[0] = %f\n", dst_data2[0]);
    float expected2 = K * 0.5f * 2.0f;
    printf("Expected Run 2: %f\n", expected2);

    // Clean up
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(ork_backend);

    // Check correctness.
    // If the activation cache was not cleared, Run 2 would reuse the cached quantized activations
    // from Run 1, resulting in output close to expected1 (64.0) instead of expected2 (128.0).
    float diff1 = std::abs(dst_data1[0] - expected1);
    float diff2 = std::abs(dst_data2[0] - expected2);

    if (diff1 > 1e-3f || diff2 > 1e-3f) {
        printf("Regression test FAILED! Outputs did not match mathematical expectations.\n");
        return 1;
    }

    printf("Regression test PASSED!\n");
    return 0;
}
