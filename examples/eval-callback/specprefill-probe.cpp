// specprefill-probe — de-risk prototype for the SpecPrefill port on the open llama.cpp stack.
//
// Proves + prototypes the ONE non-native piece: intercepting per-token attention weights
// (the named tensor `kq_soft_max`, one per layer, materialized only when flash-attn is OFF)
// via ggml_backend_sched_eval_callback, then reducing them into a per-prompt-token
// importance score the way SpecPrefill does:
//
//   importance[prompt token j] = mean over the N look-ahead query rows of
//                                 ( max over {heads, layers} softmax[key=j, qrow, head, layer] )
//
// CPU-only. No NPU submit. Run with -ngl 0 and flash-attn OFF (-fa off).
//
// Usage:
//   llama-specprefill-probe -m model.gguf -ngl 0 -fa off -p "<prompt>" [-n N_lookahead]
//
// The look-ahead set = the last prompt query row + N greedily-decoded continuation rows.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "sampling.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ---- importance accumulator, filled by the eval callback ----------------------------------
struct sp_state {
    int     n_prompt = 0;      // number of prompt tokens (keys we score)
    // per-prompt-key running reduction for the CURRENT decode batch:
    //   batch_max[qrow][j] = max over {head,layer} softmax[key j, qrow, head]  (this batch only)
    // We collect across all layers for the current batch, then after the decode the run loop
    // folds the chosen "look-ahead" query rows into the global accumulator.
    int     batch_nqrow = 0;
    std::vector<float> batch_max;   // size batch_nqrow * n_prompt, init -1

    // global accumulator over look-ahead rows:
    std::vector<double> imp_sum;    // size n_prompt
    int     n_lookahead_rows = 0;

    void begin_batch(int nqrow) {
        batch_nqrow = nqrow;
        batch_max.assign((size_t)nqrow * n_prompt, -1.0f);
    }
};

static bool sp_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (sp_state *) user_data;
    if (ask) {
        // only interested in kq_soft_max-* tensors
        return std::strncmp(t->name, "kq_soft_max", 11) == 0;
    }
    if (std::strncmp(t->name, "kq_soft_max", 11) != 0) {
        return true;
    }
    // kq_soft_max ne = {n_kv, n_qrow, n_head, 1}, f32
    const int64_t n_kv   = t->ne[0];
    const int64_t n_qrow = t->ne[1];
    const int64_t n_head = t->ne[2];
    if (t->type != GGML_TYPE_F32) {
        return true;
    }
    // pull data host-side (CPU backend: already host, but be safe)
    static std::vector<uint8_t> buf;
    const uint8_t * data;
    if (ggml_backend_buffer_is_host(t->buffer)) {
        data = (const uint8_t *) t->data;
    } else {
        buf.resize(ggml_nbytes(t));
        ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
        data = buf.data();
    }
    if (st->batch_nqrow != (int) n_qrow) {
        // batch shape changed unexpectedly; (re)size to be safe
        st->begin_batch((int) n_qrow);
    }
    const int P = st->n_prompt;
    // Prompt keys occupy the first P positions of the key axis (n_kv >= P after the prompt is cached).
    const int64_t jmax = std::min<int64_t>(P, n_kv);
    for (int64_t h = 0; h < n_head; ++h) {
        for (int64_t q = 0; q < n_qrow; ++q) {
            const uint8_t * row = data + h * t->nb[2] + q * t->nb[1];
            float * acc = st->batch_max.data() + (size_t) q * P;
            for (int64_t j = 0; j < jmax; ++j) {
                const float v = *(const float *) (row + j * t->nb[0]);
                if (v > acc[j]) acc[j] = v;
            }
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    params.n_predict = 8;     // default N look-ahead steps
    params.prompt    = "The capital of France is";

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    const int N_lookahead = params.n_predict > 0 ? params.n_predict : 8;

    sp_state st;
    params.cb_eval           = sp_cb_eval;
    params.cb_eval_user_data = &st;
    params.warmup            = false;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (!model || !ctx) {
        LOG_ERR("failed to init model/ctx\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> prompt = common_tokenize(ctx, params.prompt, add_bos, true);
    if (prompt.empty()) { LOG_ERR("empty prompt\n"); return 1; }

    const int P = (int) prompt.size();
    st.n_prompt  = P;
    st.imp_sum.assign(P, 0.0);
    st.n_lookahead_rows = 0;

    LOG_INF("prompt tokens P = %d, N_lookahead = %d\n", P, N_lookahead);

    // ---- prefill the whole prompt (single decode, n_qrow = P) ----
    st.begin_batch(P);
    if (llama_decode(ctx, llama_batch_get_one(prompt.data(), P))) {
        LOG_ERR("prefill decode failed\n"); return 1;
    }
    // The LAST prompt query row (index P-1) is the first look-ahead row (it attends over the whole prompt
    // and predicts the next token). Fold it in.
    {
        const float * row = st.batch_max.data() + (size_t)(P - 1) * P;
        for (int j = 0; j < P; ++j) st.imp_sum[j] += row[j];
        st.n_lookahead_rows++;
    }

    // greedy continuation: N more look-ahead query rows, one decode each
    int n_past = P;
    // pick argmax of last logits
    auto greedy_next = [&]() -> llama_token {
        const float * logits = llama_get_logits_ith(ctx, -1);
        const int n_vocab = llama_vocab_n_tokens(vocab);
        int best = 0; float bv = logits[0];
        for (int i = 1; i < n_vocab; ++i) if (logits[i] > bv) { bv = logits[i]; best = i; }
        return best;
    };

    for (int step = 0; step < N_lookahead; ++step) {
        llama_token tok = greedy_next();
        st.begin_batch(1);
        if (llama_decode(ctx, llama_batch_get_one(&tok, 1))) {
            LOG_ERR("decode step %d failed\n", step); return 1;
        }
        n_past++;
        // single query row -> fold it (its attention onto the first P prompt keys)
        const float * row = st.batch_max.data();  // qrow 0
        for (int j = 0; j < P; ++j) st.imp_sum[j] += row[j];
        st.n_lookahead_rows++;
    }

    // ---- final per-prompt-token importance = mean over look-ahead rows ----
    LOG("\n=== SpecPrefill per-prompt-token importance (max over heads+layers, mean over %d look-ahead rows) ===\n",
        st.n_lookahead_rows);
    std::vector<std::pair<float,int>> ranked;
    ranked.reserve(P);
    for (int j = 0; j < P; ++j) {
        float imp = (float) (st.imp_sum[j] / std::max(1, st.n_lookahead_rows));
        ranked.push_back({imp, j});
        char piece[256];
        int n = llama_token_to_piece(vocab, prompt[j], piece, sizeof(piece), 0, true);
        std::string s(piece, n > 0 ? n : 0);
        // make whitespace visible
        std::string vis;
        for (char c : s) { if (c == '\n') vis += "\\n"; else if (c == '\t') vis += "\\t"; else vis += c; }
        LOG("  [%3d] imp=%.5f  tok=%-6d  '%s'\n", j, imp, prompt[j], vis.c_str());
    }

    std::sort(ranked.begin(), ranked.end(), [](auto&a, auto&b){ return a.first > b.first; });
    LOG("\n--- top by importance ---\n");
    for (int r = 0; r < std::min(12, P); ++r) {
        int j = ranked[r].second;
        char piece[256];
        int n = llama_token_to_piece(vocab, prompt[j], piece, sizeof(piece), 0, true);
        std::string s(piece, n > 0 ? n : 0);
        std::string vis; for (char c : s) { if (c=='\n') vis+="\\n"; else if (c=='\t') vis+="\\t"; else vis+=c; }
        LOG("  #%-2d  pos=%3d  imp=%.5f  '%s'\n", r+1, j, ranked[r].first, vis.c_str());
    }

    LOG("\n");
    llama_backend_free();
    return 0;
}
