// ork_xstream — validate the ggml-ork ASYNC CROSS-STREAM path (Slice 1).
// Stream A = an NPU-heavy MUL_MAT graph run via ggml_backend_ork_graph_compute_async (returns immediately).
// Stream B = real CPU work on the main thread (a fp32 GEMV, stand-in for a speculative draft's routing math).
// Asserts (1) the NPU output is correct and (2) overlap wall ~= max(A_npu, B_cpu), NOT the sum (free overlap).
//
// Build ON THE BOARD against the already-built ggml libs (no CMake change):
//   g++ -O2 -std=c++17 -I ggml/include -o ork_xstream tools/ork_xstream.cpp \
//       -L build/bin -lggml -lggml-base -lggml-ork -Wl,-rpath,build/bin
//   sudo env ORK_MM_TIMEOUT=4000 LD_LIBRARY_PATH=build/bin ./ork_xstream [cpu_reps]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-ork.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <sched.h>
#include <pthread.h>

// ORK_XS_LITTLE=1: pin the "draft" CPU work to the little A55 cores (0-3), leaving the big A76 cores (4-7)
// for the NPU matmul's dispatch/quant helper threads — the core-partitioning test for free overlap.
static void pin_little_if_requested() {
    if (!getenv("ORK_XS_LITTLE")) return;
    cpu_set_t s; CPU_ZERO(&s); for (int c = 0; c < 4; c++) CPU_SET(c, &s);
    pthread_setaffinity_np(pthread_self(), sizeof(s), &s);
}

static double now_us() {
    using namespace std::chrono;
    return duration<double, std::micro>(steady_clock::now().time_since_epoch()).count();
}

static volatile float g_sink = 0;
static void cpu_work(int reps, std::vector<float> & W, std::vector<float> & x, std::vector<float> & y) {
    const int RN = 512;                                  // 1 MB matrix -> real DRAM traffic (BW-contention probe)
    for (int r = 0; r < reps; r++) {
        for (int a = 0; a < RN; a++) { float acc = 0; const float * wr = &W[(size_t) a * RN];
            for (int b = 0; b < RN; b++) acc += wr[b] * x[b]; y[a] = acc; }
        x[0] = y[RN - 1] * 1e-9f;                         // cross-rep dependency -> no dead-code elimination
    }
    g_sink += x[0];
}

int main(int argc, char ** argv) {
    const int M = 64, K = 4096, N = 4096;                // NPU op ~2.7ms (stands in for a heavy verify layer)
    const int reps = argc > 1 ? atoi(argv[1]) : 16;
    const int iters = 8;

    pin_little_if_requested();                           // optionally keep the "draft" off the NPU's big cores
    ggml_backend_t ork = ggml_backend_ork_init();
    if (!ork) { printf("ork init failed\n"); return 2; }

    // ---- one-node MUL_MAT graph: out[N,M] = W[K,N]^T . A[K,M] ----
    size_t mem = 8 * ggml_tensor_overhead() + ggml_graph_overhead() + 4096;
    struct ggml_init_params ip = { mem, NULL, true };    // no_alloc: backend buffer holds tensor data
    struct ggml_context * ctx = ggml_init(ip);
    struct ggml_tensor * Wt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N); ggml_set_name(Wt, "weight.w");
    struct ggml_tensor * At = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M); ggml_set_name(At, "act");
    struct ggml_tensor * Ct = ggml_mul_mat(ctx, Wt, At);                    ggml_set_name(Ct, "out");
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, Ct);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, ork);
    if (!buf) { printf("alloc failed\n"); return 2; }

    std::vector<float> Wd((size_t) K * N), Ad((size_t) K * M);
    for (size_t i = 0; i < Wd.size(); i++) Wd[i] = 1.0f / 128.0f;           // out = sum_k (1/128)*1 = K/128 = 32
    for (size_t i = 0; i < Ad.size(); i++) Ad[i] = 1.0f;
    ggml_backend_tensor_set(Wt, Wd.data(), 0, Wd.size() * sizeof(float));
    ggml_backend_tensor_set(At, Ad.data(), 0, Ad.size() * sizeof(float));

    std::vector<float> cW(512 * 512), cx(512, 1.0f), cy(512);
    for (size_t i = 0; i < cW.size(); i++) cW[i] = (float) (((unsigned) i * 2654435761u) >> 28) * 0.01f;

    ggml_backend_ork_graph_compute_async(ork, gf); ggml_backend_ork_synchronize(ork);   // warm

    double t0 = now_us();                                                  // (1) NPU solo
    for (int i = 0; i < iters; i++) { ggml_backend_ork_graph_compute_async(ork, gf); ggml_backend_ork_synchronize(ork); }
    double npu_solo = (now_us() - t0) / iters;

    t0 = now_us();                                                         // (2) CPU solo
    for (int i = 0; i < iters; i++) cpu_work(reps, cW, cx, cy);
    double cpu_solo = (now_us() - t0) / iters;

    t0 = now_us();                                                         // (3) OVERLAP: CPU work in the NPU's shadow
    for (int i = 0; i < iters; i++) { ggml_backend_ork_graph_compute_async(ork, gf); cpu_work(reps, cW, cx, cy); ggml_backend_ork_synchronize(ork); }
    double overlap = (now_us() - t0) / iters;

    std::vector<float> Cd((size_t) N * M);
    ggml_backend_tensor_get(Ct, Cd.data(), 0, Cd.size() * sizeof(float));
    const double expected = (double) K / 128.0;
    double maxerr = 0; for (size_t i = 0; i < Cd.size(); i++) { double e = fabs(Cd[i] - expected); if (e > maxerr) maxerr = e; }

    const double sum = npu_solo + cpu_solo, mn = npu_solo < cpu_solo ? npu_solo : cpu_solo;
    double hidden = 100.0 * (sum - overlap) / (mn > 0 ? mn : 1); if (hidden > 100) hidden = 100; if (hidden < 0) hidden = 0;
    const bool ok = maxerr < expected * 0.05 + 1.0;
    printf("ggml-ork async cross-stream (M=%d K=%d N=%d, cpu_reps=%d):\n", M, K, N, reps);
    printf("  npu_solo=%.1fus  cpu_solo=%.1fus  overlap=%.1fus  sum=%.1fus  hidden=%.1f%%\n", npu_solo, cpu_solo, overlap, sum, hidden);
    printf("  NPU out: expected~%.2f  maxerr=%.3f -> %s\n", expected, maxerr, ok ? "CORRECT" : "WRONG");
    printf("  %s\n", (ok && hidden > 70) ? "PASS: async cross-stream overlap is free + correct through ggml-ork"
                                         : "CHECK: see numbers above");
    (void) g_sink;
    ggml_backend_buffer_free(buf); ggml_free(ctx); ggml_backend_free(ork);
    return ok ? 0 : 1;
}
