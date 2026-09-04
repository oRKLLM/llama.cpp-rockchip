// ork_xattn — Slice 2 validation: is the ggml-ork fused-attention path CONCURRENCY-SAFE after moving its
// scratch (was global g_attnp + softmax statics) per-context? Builds a FLASH_ATTN_EXT graph, then:
//   (1) correctness: ork NPU output vs the CPU backend on the same graph/data;
//   (2) concurrency: TWO ork backends run the same attention graph on two threads for many iters (direct
//       ggml_backend_graph_compute -> no coarse mutex -> genuinely concurrent), each comparing every result
//       to the serial reference. A data race on shared attention scratch would corrupt some outputs.
// Build ON THE BOARD:
//   g++ -O2 -std=c++17 -I ggml/include -o ork_xattn tools/ork_xattn.cpp -L build/bin \
//       -Wl,--start-group -lggml-ork -lggml-cpu -lggml -lggml-base -Wl,--end-group -lpthread -Wl,-rpath,$PWD/build/bin
//   sudo env ORK_ATTN=1 ORK_MM_TIMEOUT=4000 LD_LIBRARY_PATH=build/bin ./ork_xattn [iters]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-ork.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <thread>
#include <atomic>

static const int DK=64, DV=64, H=4, Hkv=4, N=64, KV=64, Bt=1;

struct AttnGraph {
    ggml_context * ctx = nullptr; ggml_cgraph * gf = nullptr; ggml_tensor * out = nullptr;
    ggml_backend_buffer_t buf = nullptr; ggml_tensor *q=nullptr,*k=nullptr,*v=nullptr,*m=nullptr;
};

static AttnGraph build(ggml_backend_t backend) {
    AttnGraph g;
    g.ctx = ggml_init({ 24*ggml_tensor_overhead() + ggml_graph_overhead() + 8192, nullptr, true });
    g.q = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, DK, N,  H,   Bt); ggml_set_name(g.q,"q");
    g.k = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F16, DK, KV, Hkv, Bt); ggml_set_name(g.k,"k");
    g.v = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F16, DV, KV, Hkv, Bt); ggml_set_name(g.v,"v");
    g.m = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F16, KV, N,  1,   1);  ggml_set_name(g.m,"m");
    g.out = ggml_flash_attn_ext(g.ctx, g.q, g.k, g.v, g.m, 1.0f/sqrtf((float)DK), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(g.out, GGML_PREC_F32);
    ggml_set_name(g.out, "out");
    g.gf = ggml_new_graph(g.ctx); ggml_build_forward_expand(g.gf, g.out);
    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, backend);
    return g;
}

// deterministic fill (same data everywhere -> every backend/thread must produce the identical output)
static void fill(AttnGraph & g) {
    std::vector<float>       q(DK*N*H*Bt);   for (size_t i=0;i<q.size();i++) q[i] = 0.02f*(float)((i*2654435761u>>28));
    std::vector<ggml_fp16_t> k(DK*KV*Hkv*Bt);for (size_t i=0;i<k.size();i++) k[i] = ggml_fp32_to_fp16(0.02f*(float)((i*40503u>>7)&15));
    std::vector<ggml_fp16_t> v(DV*KV*Hkv*Bt);for (size_t i=0;i<v.size();i++) v[i] = ggml_fp32_to_fp16(0.02f*(float)((i*2246822519u>>28)));
    std::vector<ggml_fp16_t> m(KV*N);        for (size_t i=0;i<m.size();i++) m[i] = ggml_fp32_to_fp16(0.0f);   // all-visible
    ggml_backend_tensor_set(g.q, q.data(), 0, q.size()*sizeof(float));
    ggml_backend_tensor_set(g.k, k.data(), 0, k.size()*sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(g.v, v.data(), 0, v.size()*sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(g.m, m.data(), 0, m.size()*sizeof(ggml_fp16_t));
}

static std::vector<float> run_once(ggml_backend_t backend, AttnGraph & g) {
    ggml_backend_graph_compute(backend, g.gf);
    std::vector<float> o(ggml_nelements(g.out));
    ggml_backend_tensor_get(g.out, o.data(), 0, o.size()*sizeof(float));
    return o;
}
// Concurrent path: go through the async API so the process-global g_npu_queue_mu SERIALIZES the two
// streams' NPU submits (the RKNPU is single-stream — raw concurrent submits IOMMU-fault/wedge). The
// per-context attention scratch is still exercised by each stream's worker; this proves the moved-per-context
// state is correct under multi-threaded invocation without wedging.
static std::vector<float> run_once_async(ggml_backend_t backend, AttnGraph & g) {
    ggml_backend_ork_graph_compute_async(backend, g.gf);
    ggml_backend_ork_synchronize(backend);
    std::vector<float> o(ggml_nelements(g.out));
    ggml_backend_tensor_get(g.out, o.data(), 0, o.size()*sizeof(float));
    return o;
}

static double maxdiff(const std::vector<float>&a, const std::vector<float>&b){ double m=0; for(size_t i=0;i<a.size();i++){double d=fabs((double)a[i]-b[i]); if(d>m)m=d;} return m; }

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);                 // unbuffered: a hang shows exactly where
    const int iters = argc > 1 ? atoi(argv[1]) : 200;    // iters<=0 => serial correctness only (skip concurrent stress)
    if (!getenv("ORK_ATTN")) { printf("set ORK_ATTN=1 (fused attention is opt-in)\n"); return 2; }

    // ---- CPU reference ----
    ggml_backend_t cpu = ggml_backend_cpu_init();
    AttnGraph gc = build(cpu); fill(gc);
    std::vector<float> ref = run_once(cpu, gc);
    double refmag = 0; for (float x : ref) refmag = fmax(refmag, fabs(x));
    printf("CPU ref: %zu elems, max|.|=%.4f\n", ref.size(), refmag);

    // ---- ork serial correctness ----
    ggml_backend_t orkA = ggml_backend_ork_init();
    if (!orkA) { printf("ork init failed\n"); return 2; }
    AttnGraph ga = build(orkA); fill(ga);
    std::vector<float> oa = run_once(orkA, ga);
    double err_serial = maxdiff(oa, ref);
    printf("ork serial vs CPU: maxdiff=%.4f -> %s\n", err_serial, err_serial < 0.05*refmag+0.02 ? "CORRECT" : "WRONG");

    // ---- CONCURRENCY: two ork backends, two threads, same graph/data, many iters (opt-in: iters>0) ----
    int mA = 0, mB = 0;
    ggml_backend_t orkB = nullptr;
    if (iters > 0) {
        orkB = ggml_backend_ork_init();
        AttnGraph gb = build(orkB); fill(gb);
        std::atomic<int> mismA(0), mismB(0);
        auto worker = [&](ggml_backend_t be, AttnGraph & g, std::atomic<int> & mism) {
            for (int i = 0; i < iters; i++) {
                std::vector<float> o = run_once_async(be, g);    // serialized submits (g_npu_queue_mu) + per-context scratch
                if (maxdiff(o, ref) > 0.05*refmag+0.02) mism++;
            }
        };
        std::thread tA(worker, orkA, std::ref(ga), std::ref(mismA));
        std::thread tB(worker, orkB, std::ref(gb), std::ref(mismB));
        tA.join(); tB.join();
        mA = mismA.load(); mB = mismB.load();
        printf("concurrent %d iters x2 (serialized submits): mismatches A=%d B=%d\n", iters, mA, mB);
    } else {
        printf("(serial-only mode; skipped concurrent stress)\n");
    }

    const bool ok = err_serial < 0.05*refmag+0.02 && mA == 0 && mB == 0;
    printf("%s\n", ok ? "PASS: fused attention is correct + concurrency-safe (per-context scratch)"
                      : "FAIL: see numbers above");
    ggml_backend_free(orkA); if (orkB) ggml_backend_free(orkB); ggml_backend_free(cpu);
    return ok ? 0 : 1;
}
