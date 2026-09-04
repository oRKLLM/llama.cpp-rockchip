// specprefill — end-to-end SpecPrefill TTFT harness on the open llama.cpp stack.
//
// Three pieces, as proven feasible by examples/eval-callback/specprefill-probe.cpp:
//
//  (1) IMPORTANCE: run a SMALL draft model (FA-OFF, dense) over the prompt + N greedy
//      look-ahead steps. Intercept the named `kq_soft_max` tensor per layer via the
//      eval callback. Reduce: importance[j] = mean over the N+1 look-ahead query rows of
//      ( max over {heads, layers} softmax[key=j, qrow] ).  (reuses the probe's logic)
//
//  (2) PRUNE: chunk the prompt, average importance per chunk, keep the top-K chunks at a
//      keep ratio r. The BOS token and the last chunk are always kept.
//
//  (3) PRUNED PREFILL: build the BIG target's prefill batch with ONLY the kept tokens, each
//      at its ORIGINAL position (batch.pos[] with gaps — native to llama.cpp), then decode a
//      few continuation tokens starting from the original max position + 1.
//
// MEASURE: full-prompt target prefill TTFT vs (draft importance pass + pruned target prefill)
// at each keep ratio. Also emit the target's answer (greedy) for quality/needle checking.
//
// Models are loaded SEQUENTIALLY (draft -> free -> target) so the 22GB target and the 1.7B
// draft don't co-reside; the draft is fully freed before the target prefill is timed.
//
// Usage:
//   llama-specprefill --model-draft DRAFT.gguf --model TARGET.gguf -ngl 0 -fa off \
//       -f prompt.txt [-n N_lookahead] [--sp-chunk C] [--sp-answer A]
//
// Keep ratios are fixed at {1.0 (full baseline), 0.7, 0.5, 0.3}; the harness runs all of them.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "sampling.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// SP_NATIVE_POS=1 -> keep kept tokens at ORIGINAL positions (gapped). NOTE: this build's batch
// validator rejects gapped positions ("positions are not continuous"); default 0 re-packs dense.
static int g_native_pos = 0;

// Gate the eval callback: when false, the `ask` phase returns false so the scheduler is NOT asked
// for kq_soft_max and inserts no per-node sync barrier (huge speedup on the bulk prompt prefill,
// whose query rows we don't fold anyway). Set true only around the look-ahead submits we reduce.
static bool g_fold_active = false;

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

// ---- importance accumulator (eval callback), per the probe ----
struct sp_state {
    int n_prompt = 0;
    int batch_nqrow = 0;
    // We only ever need ONE query row per decode batch (the look-ahead row): for the prompt
    // prefill it's the LAST row (P-1); for each decode batch it's row 0. So we reduce only that
    // row into batch_max[0..P) — O(n_kv) per head/layer instead of O(n_qrow*n_kv) (the latter is
    // ~2B ops for a 2k prompt and dominated runtime).
    int only_qrow = -1;              // which query row to reduce (-1 => set in begin_batch)
    std::vector<float>  batch_max;   // size n_prompt, max over {head,layer} for the chosen row
    std::vector<double> imp_sum;     // n_prompt
    int n_lookahead_rows = 0;
    void begin_batch(int nqrow) {
        batch_nqrow = nqrow;
        only_qrow   = nqrow - 1;     // last row of this batch
        batch_max.assign((size_t) n_prompt, -1.0f);
    }
};

static bool sp_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (sp_state *) user_data;
    if (ask) {
        return g_fold_active && std::strncmp(t->name, "kq_soft_max", 11) == 0;
    }
    if (std::strncmp(t->name, "kq_soft_max", 11) != 0) return true;
    const int64_t n_kv   = t->ne[0];
    const int64_t n_qrow = t->ne[1];
    const int64_t n_head = t->ne[2];
    if (t->type != GGML_TYPE_F32) return true;

    static std::vector<uint8_t> buf;
    const uint8_t * data;
    if (ggml_backend_buffer_is_host(t->buffer)) {
        data = (const uint8_t *) t->data;
    } else {
        buf.resize(ggml_nbytes(t));
        ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
        data = buf.data();
    }
    if (st->batch_nqrow != (int) n_qrow) st->begin_batch((int) n_qrow);
    const int P = st->n_prompt;
    const int64_t jmax = std::min<int64_t>(P, n_kv);
    const int64_t q = st->only_qrow;           // reduce only the look-ahead query row
    if (q < 0 || q >= n_qrow) return true;
    float * acc = st->batch_max.data();
    for (int64_t h = 0; h < n_head; ++h) {
        const uint8_t * row = data + h * t->nb[2] + q * t->nb[1];
        for (int64_t j = 0; j < jmax; ++j) {
            const float v = *(const float *) (row + j * t->nb[0]);
            if (v > acc[j]) acc[j] = v;
        }
    }
    return true;
}

// greedy argmax of last logits
static llama_token greedy_next(llama_context * ctx, const llama_vocab * vocab) {
    const float * logits = llama_get_logits_ith(ctx, -1);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    int best = 0; float bv = logits[0];
    for (int i = 1; i < n_vocab; ++i) if (logits[i] > bv) { bv = logits[i]; best = i; }
    return best;
}

static std::string vis_piece(const llama_vocab * vocab, llama_token tok) {
    char piece[256];
    int n = llama_token_to_piece(vocab, tok, piece, sizeof(piece), 0, true);
    std::string s(piece, n > 0 ? n : 0), out;
    for (char c : s) { if (c=='\n') out+="\\n"; else if (c=='\t') out+="\\t"; else out+=c; }
    return out;
}

int main(int argc, char ** argv) {
    common_params params;
    params.n_predict = 8;              // N look-ahead steps for importance
    params.sampling.temp = 0.0f;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }
    g_native_pos = getenv("SP_NATIVE_POS") != nullptr;

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    // ensure context is large enough for a long prompt + look-ahead + answer, and that a single
    // full-prompt prefill fits in one LOGICAL batch (the importance pass + the full-prompt baseline
    // submit all tokens at once). Keep the PHYSICAL ubatch small (512): FA-off attention is O(ubatch
    // * n_kv) per layer, so a giant ubatch blows up compute+memory (a 4096 ubatch made a 1.7B use
    // ~7GB and stall for >10 min). With ubatch=512 the callback fires per-ubatch; we read batch_max
    // only after the whole prefill, so it correctly holds the last ubatch's last-row reduction (the
    // global look-ahead row P-1).
    if (params.n_ctx    < 8192) params.n_ctx    = 8192;
    if (params.n_batch  < 4096) params.n_batch  = 4096;
    if (params.n_ubatch == 0 || params.n_ubatch > 512) params.n_ubatch = 512;

    const int  N_lookahead = params.n_predict > 0 ? params.n_predict : 8;
    const int  CHUNK       = 16;       // prompt chunk size for pruning granularity
    const int  N_ANSWER    = 48;       // continuation tokens to decode on the target (quality)
    const std::string draft_path  = params.speculative.draft.mparams.path;
    const std::string target_path = params.model.path;
    if (draft_path.empty() || target_path.empty()) {
        LOG_ERR("need --model-draft and --model\n");
        return 1;
    }

    // We need the prompt tokens. Tokenize using the TARGET vocab (same family; positions
    // are what matter and the two share the Qwen3 tokenizer). We'll tokenize once we have a
    // model loaded; do it with the draft first (same tokenizer).

    // =========================== (1) IMPORTANCE PASS (draft) ===========================
    std::vector<llama_token> prompt;
    std::vector<double> importance;
    int P = 0;
    double t_importance_ms = 0.0;
    {
        common_params dp = params;
        dp.model.path = draft_path;
        dp.cb_eval = sp_cb_eval;
        dp.warmup = false;
        dp.n_predict = N_lookahead;
        // dp.cb_eval_user_data set after we know P
        // force FA off for the draft (kq_soft_max only materializes FA-off)
        dp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

        // first load to tokenize
        sp_state st;
        dp.cb_eval_user_data = &st;
        auto dinit = common_init_from_params(dp);
        auto * dmodel = dinit->model();
        auto * dctx   = dinit->context();
        if (!dmodel || !dctx) { LOG_ERR("draft init failed\n"); return 1; }
        const llama_vocab * dvocab = llama_model_get_vocab(dmodel);
        const bool add_bos = llama_vocab_get_add_bos(dvocab);
        prompt = common_tokenize(dctx, params.prompt, add_bos, true);
        if (prompt.empty()) { LOG_ERR("empty prompt\n"); return 1; }
        P = (int) prompt.size();
        st.n_prompt = P;
        st.imp_sum.assign(P, 0.0);
        st.n_lookahead_rows = 0;
        LOG_INF("prompt tokens P = %d, N_lookahead = %d, chunk = %d\n", P, N_lookahead, CHUNK);

        auto t0 = clk::now();
        // Prefill tokens [0..P-2] WITHOUT folding (we don't need their query rows; the callback
        // would reduce per-ubatch last rows which aren't the global look-ahead row). Then submit
        // token P-1 ALONE (n_qrow=1) so the callback reduces exactly its attention over all keys
        // -> the first look-ahead row. This keeps every folded batch n_qrow==1 (trivially correct)
        // and lets the bulk prefill stream through small ubatches cheaply.
        st.only_qrow = -1;  // disable folding during the bulk prefill
        g_fold_active = false;  // do not ask for kq_soft_max during the bulk prefill (no sync)
        if (P > 1) {
            if (llama_decode(dctx, llama_batch_get_one(prompt.data(), P - 1))) {
                LOG_ERR("draft bulk prefill failed\n"); return 1;
            }
        }
        g_fold_active = true;   // now reduce the look-ahead rows
        { // token P-1 alone = first look-ahead row
            st.begin_batch(1);
            llama_token last = prompt[P - 1];
            if (llama_decode(dctx, llama_batch_get_one(&last, 1))) {
                LOG_ERR("draft last-token decode failed\n"); return 1;
            }
            const float * row = st.batch_max.data();
            for (int j = 0; j < P; ++j) st.imp_sum[j] += row[j];
            st.n_lookahead_rows++;
        }
        for (int step = 0; step < N_lookahead; ++step) {
            llama_token tok = greedy_next(dctx, dvocab);
            st.begin_batch(1);
            if (llama_decode(dctx, llama_batch_get_one(&tok, 1))) {
                LOG_ERR("draft lookahead %d failed\n", step); return 1;
            }
            const float * row = st.batch_max.data();
            for (int j = 0; j < P; ++j) st.imp_sum[j] += row[j];
            st.n_lookahead_rows++;
        }
        t_importance_ms = ms_since(t0);

        importance.assign(P, 0.0);
        for (int j = 0; j < P; ++j) importance[j] = st.imp_sum[j] / std::max(1, st.n_lookahead_rows);
        LOG_INF("draft importance pass: %.1f ms (%d look-ahead rows)\n", t_importance_ms, st.n_lookahead_rows);
        // dinit goes out of scope here -> draft model + ctx freed before target loads
    }

    // =========================== chunk-level importance ===========================
    const int n_chunks = (P + CHUNK - 1) / CHUNK;
    std::vector<double> chunk_imp(n_chunks, 0.0);
    std::vector<int>    chunk_cnt(n_chunks, 0);
    for (int j = 0; j < P; ++j) { int c = j / CHUNK; chunk_imp[c] += importance[j]; chunk_cnt[c]++; }
    for (int c = 0; c < n_chunks; ++c) if (chunk_cnt[c]) chunk_imp[c] /= chunk_cnt[c];

    // =========================== load TARGET once, run all ratios ===========================
    common_params tp = params;
    tp.model.path = target_path;
    tp.warmup = false;
    // The TARGET does NOT need FA-off (only the draft importance pass reads kq_soft_max). Pruned
    // prefill uses dense contiguous positions, so FA works. SP_TGT_FA=off forces FA-off for A/B.
    {
        const char * f = getenv("SP_TGT_FA");
        tp.flash_attn_type = (f && std::strcmp(f, "off") == 0)
            ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }
    tp.cb_eval = nullptr;  // no callback on the target
    tp.cb_eval_user_data = nullptr;
    auto tinit = common_init_from_params(tp);
    auto * tmodel = tinit->model();
    auto * tctx   = tinit->context();
    if (!tmodel || !tctx) { LOG_ERR("target init failed\n"); return 1; }
    const llama_vocab * tvocab = llama_model_get_vocab(tmodel);
    const int n_batch = (int) llama_n_batch(tctx);

    // helper: build a kept-token list for ratio r (BOS + last chunk always kept), prefill the
    // target on exactly those tokens at ORIGINAL positions, decode N_ANSWER greedy tokens, time
    // the prefill (TTFT for the pruned case), and print the answer + last-token logit head.
    auto run_ratio = [&](double r, bool full) -> std::pair<double,std::string> {
        // memory KV reset
        llama_memory_clear(llama_get_memory(tctx), true);

        std::vector<int> keep;  // prompt indices to keep, in order
        if (full) {
            keep.resize(P);
            for (int j = 0; j < P; ++j) keep[j] = j;
        } else {
            int keep_chunks = std::max(1, (int) std::lround(r * n_chunks));
            // rank chunks by importance desc, take top keep_chunks; always include chunk 0 (BOS)
            std::vector<int> order(n_chunks);
            for (int c = 0; c < n_chunks; ++c) order[c] = c;
            std::sort(order.begin(), order.end(), [&](int a, int b){ return chunk_imp[a] > chunk_imp[b]; });
            std::vector<char> sel(n_chunks, 0);
            for (int i = 0; i < keep_chunks && i < n_chunks; ++i) sel[order[i]] = 1;
            sel[0] = 1;               // always keep BOS chunk
            sel[n_chunks-1] = 1;      // always keep last chunk (the question / most recent)
            for (int j = 0; j < P; ++j) if (sel[j / CHUNK]) keep.push_back(j);
        }

        // Position assignment for the kept tokens:
        //  - native_pos: keep each token at its ORIGINAL position (pos[] with gaps). This is the
        //    "native gaps" idea, BUT this llama.cpp build's batch validator requires a sequence's
        //    positions to be contiguous (src/llama-batch.cpp "positions are not continuous"), so
        //    gapped positions are REJECTED -> decode-fail. Kept as a measurable, honest fallback.
        //  - dense (default): re-pack kept tokens to consecutive positions 0..K-1, preserving order
        //    (standard prefix-pruning / KV-eviction layout). The answer continues from position K.
        const int K = (int) keep.size();
        // prefill in n_batch-sized sub-batches, only the last kept token gets logits
        auto t0 = clk::now();
        llama_batch batch = llama_batch_init(n_batch, 0, 1);
        for (size_t i = 0; i < keep.size(); ) {
            common_batch_clear(batch);
            size_t e = std::min(keep.size(), i + (size_t) n_batch);
            for (size_t k = i; k < e; ++k) {
                int j = keep[k];
                bool want_logits = (k == keep.size() - 1);
                llama_pos pos = g_native_pos ? (llama_pos) j : (llama_pos) k;
                common_batch_add(batch, prompt[j], pos, {0}, want_logits);
            }
            if (llama_decode(tctx, batch)) {
                LOG_ERR("target pruned prefill decode failed (r=%.2f) [native_pos=%d]\n", r, g_native_pos);
                llama_batch_free(batch);
                return { -1.0, std::string("<decode-fail>") };
            }
            i = e;
        }
        double ttft_ms = ms_since(t0);   // prefill time == TTFT (logits for first new token ready)
        llama_batch_free(batch);

        // decode answer: llama_batch_get_one auto-continues from the KV's current max position,
        // so the next token lands at last_pos+1 (== P in native_pos, == K in dense). greedy.
        (void) K;
        std::string answer;
        llama_token tok = greedy_next(tctx, tvocab);
        for (int s = 0; s < N_ANSWER; ++s) {
            answer += vis_piece(tvocab, tok);
            if (llama_vocab_is_eog(tvocab, tok)) break;
            if (llama_decode(tctx, llama_batch_get_one(&tok, 1))) break;
            tok = greedy_next(tctx, tvocab);
        }
        LOG_INF("ratio %s: kept %zu / %d tokens, TTFT(prefill) = %.1f ms\n",
                full ? "FULL(1.00)" : (std::to_string(r)).c_str(), keep.size(), P, ttft_ms);
        return { ttft_ms, answer };
    };

    // ---- run full baseline, then 0.7 / 0.5 / 0.3 ----
    struct Row { std::string label; double r; size_t kept; double ttft; std::string ans; };
    std::vector<Row> rows;

    {
        auto [ttft, ans] = run_ratio(1.0, true);
        rows.push_back({ "full", 1.0, (size_t) P, ttft, ans });
    }
    for (double r : { 0.7, 0.5, 0.3 }) {
        // recompute kept count for the row label
        int keep_chunks = std::max(1, (int) std::lround(r * n_chunks));
        auto [ttft, ans] = run_ratio(r, false);
        // recount actual kept tokens by replaying selection (cheap)
        std::vector<int> order(n_chunks); for (int c=0;c<n_chunks;++c) order[c]=c;
        std::sort(order.begin(), order.end(), [&](int a,int b){ return chunk_imp[a]>chunk_imp[b]; });
        std::vector<char> sel(n_chunks,0);
        for (int i=0;i<keep_chunks && i<n_chunks;++i) sel[order[i]]=1;
        sel[0]=1; sel[n_chunks-1]=1;
        size_t kept=0; for (int j=0;j<P;++j) if (sel[j/CHUNK]) kept++;
        rows.push_back({ std::string("r=")+std::to_string(r).substr(0,3), r, kept, ttft, ans });
    }

    // =========================== REPORT ===========================
    double full_ttft = rows[0].ttft;
    LOG("\n");
    LOG("================= SpecPrefill TTFT report =================\n");
    LOG("prompt P=%d tokens, chunk=%d (%d chunks), N_lookahead=%d\n", P, CHUNK, n_chunks, N_lookahead);
    LOG("draft importance pass overhead = %.1f ms\n", t_importance_ms);
    LOG("target full-prompt prefill (baseline TTFT) = %.1f ms\n", full_ttft);
    LOG("\n");
    LOG("%-8s %6s %10s %12s %12s %10s\n", "ratio", "kept", "prefill_ms", "net_ttft_ms", "net_vs_full", "speedup");
    for (auto & row : rows) {
        if (row.label == "full") {
            LOG("%-8s %6zu %10.1f %12.1f %12s %9.2fx\n",
                row.label.c_str(), row.kept, row.ttft, row.ttft, "(baseline)", 1.0);
        } else {
            double net = row.ttft + t_importance_ms;   // pruned prefill + the importance cost
            LOG("%-8s %6zu %10.1f %12.1f %11.1f%% %9.2fx\n",
                row.label.c_str(), row.kept, row.ttft, net,
                100.0 * net / full_ttft, full_ttft / net);
        }
    }
    LOG("\n--- answers (greedy %d tok; quality / needle check) ---\n", N_ANSWER);
    for (auto & row : rows) {
        LOG("[%s] %s\n", row.label.c_str(), row.ans.c_str());
    }
    LOG("\n");

    llama_backend_free();
    return 0;
}
