// DFlash standalone harness (P3/P4).
//
// Drives a DFlash block-diffusion drafter against a target model through the PUBLIC llama API — the same
// primitives oRKLLM's JS orchestration will call. For a de-risked first milestone this MEASURES the drafter
// (generates via the target greedily, and at each block runs the DFlash draft and compares to the target's
// actual continuation) rather than doing full speculative rollback: it validates (a) the draft block-forward
// and (b) that both the encoder/context-precompute and the block forward route to the NPU (M>1 grouped),
// plus the acceptance length tau. Turning the drafts into skip-ahead speculative decoding is a follow-up.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include "../../src/llama-ext.h" // staging API: layer-inp extraction, embeddings_nextn, target_layer_ids, dflash ctx

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // ---- target model (uses the ggml-ork/NPU backend if enabled) ----
    auto llama_init_tgt = common_init_from_params(params);
    llama_model   * model_tgt = llama_init_tgt->model();
    llama_context * ctx_tgt   = llama_init_tgt->context();
    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // ---- draft (DFlash) model — borrows the target's tok_embd/output via ctx_other ----
    llama_model_ptr   model_dft;
    llama_context_ptr ctx_dft;
    {
        auto params_dft = params;
        params_dft.model = params.speculative.draft.mparams;

        auto mparams_dft = common_model_params_to_llama(params_dft);
        model_dft.reset(llama_model_load_from_file(params_dft.model.path.c_str(), mparams_dft));
        if (!model_dft) {
            LOG_ERR("%s: failed to load draft model '%s'\n", __func__, params_dft.model.path.c_str());
            return 1;
        }

        auto cparams = common_context_params_to_llama(params_dft);
        cparams.ctx_other       = ctx_tgt;                        // borrow target embeddings/lm_head
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED; // block attn uses non-flash build_attn_mha
        ctx_dft.reset(llama_init_from_model(model_dft.get(), cparams));
        if (!ctx_dft) {
            LOG_ERR("%s: failed to create draft context\n", __func__);
            return 1;
        }
        llama_set_embeddings_nextn(ctx_dft.get(), true, /*masked=*/ false); // capture fused encoder output
    }

    const int32_t   n_embd_tgt = llama_model_n_embd(model_tgt);
    const int32_t   n_embd_dec = llama_model_n_embd(model_dft.get());
    const int32_t * tlids      = llama_model_target_layer_ids  (model_dft.get());
    const uint32_t  n_tlids    = llama_model_target_layer_ids_n(model_dft.get());
    const int32_t   n_embd_enc = (int32_t) n_tlids * n_embd_tgt;
    const llama_token mask_tok = llama_vocab_mask(llama_model_get_vocab(model_dft.get()));

    int block_size = 16;
    if (const char * e = getenv("DFLASH_BLOCK")) { block_size = atoi(e); }

    LOG_INF("%s: DFlash target_layers=%u n_embd_tgt=%d n_embd_dec=%d n_embd_enc=%d mask_tok=%d block=%d\n",
            __func__, n_tlids, n_embd_tgt, n_embd_dec, n_embd_enc, mask_tok, block_size);
    if (mask_tok < 0) {
        LOG_ERR("%s: draft model has no mask token (tokenizer.ggml.mask_token_id)\n", __func__);
        return 1;
    }

    for (uint32_t k = 0; k < n_tlids; ++k) {
        llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) tlids[k], true);
    }

    auto argmax = [&](const float * lg) {
        int b = 0; float bv = lg[0];
        for (int v = 1; v < n_vocab; ++v) { if (lg[v] > bv) { bv = lg[v]; b = v; } }
        return b;
    };

    std::vector<float>   ctx_g;    // [n_embd_dec, ctx_len] flat (grows with the sequence)
    std::vector<int32_t> ctx_pos;  // [ctx_len]

    // encode a [n, n_embd_enc] feature buffer through the draft encoder and append the g_embd rows to ctx_g
    auto encode_append = [&](const float * feat, int32_t n, int32_t base_pos) {
        llama_batch enc = { n, nullptr, const_cast<float *>(feat), nullptr, nullptr, nullptr, nullptr };
        if (llama_encode(ctx_dft.get(), enc) != 0) { LOG_ERR("%s: llama_encode failed\n", __func__); return false; }
        const float * g = llama_get_embeddings_nextn(ctx_dft.get());
        if (!g) { LOG_ERR("%s: encoder produced no g_embd\n", __func__); return false; }
        const size_t off = ctx_g.size();
        ctx_g.resize(off + (size_t) n * n_embd_dec);
        std::memcpy(ctx_g.data() + off, g, (size_t) n * n_embd_dec * sizeof(float));
        for (int32_t i = 0; i < n; ++i) { ctx_pos.push_back(base_pos + i); }
        return true;
    };

    // copy the last target decode's `n` extracted layer rows (interleaved) into feat[row_off ..]
    std::vector<float> feat;
    auto grab_features = [&](int32_t n, int32_t row_off) {
        for (uint32_t k = 0; k < n_tlids; ++k) {
            const float * layer = llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) tlids[k]);
            if (!layer) { LOG_ERR("%s: target layer %d not extracted\n", __func__, tlids[k]); return false; }
            for (int32_t i = 0; i < n; ++i) {
                std::memcpy(feat.data() + (size_t) (row_off + i) * n_embd_enc + (size_t) k * n_embd_tgt,
                            layer + (size_t) i * n_embd_tgt, (size_t) n_embd_tgt * sizeof(float));
            }
        }
        return true;
    };

    // ---- prefill the prompt on the target, then build the initial context ----
    std::vector<llama_token> inp = common_tokenize(ctx_tgt, params.prompt, true, true);
    const int32_t n_prompt = (int32_t) inp.size();

    llama_batch pb = llama_batch_init(std::max(n_prompt, block_size), 0, 1);
    common_batch_clear(pb);
    for (int32_t i = 0; i < n_prompt; ++i) { common_batch_add(pb, inp[i], i, { 0 }, i == n_prompt - 1); }
    if (llama_decode(ctx_tgt, pb) != 0) { LOG_ERR("%s: prompt decode failed\n", __func__); return 1; }

    feat.assign((size_t) n_prompt * n_embd_enc, 0.0f);
    if (!grab_features(n_prompt, 0))         { return 1; }
    if (!encode_append(feat.data(), n_prompt, 0)) { return 1; }

    int32_t     n_dec   = n_prompt;
    llama_token id_next = argmax(llama_get_logits_ith(ctx_tgt, n_prompt - 1)); // target token @ n_dec

    const int n_predict = params.n_predict > 0 ? params.n_predict : 128;
    int64_t n_blocks = 0, tau_sum = 0, n_gen = 0;

    llama_batch blk = llama_batch_init(block_size + 1, 0, 1); // anchor + block_size mask tokens
    llama_batch tb  = llama_batch_init(1, 0, 1);          // single target token

    LOG("\n");
    for (auto id : inp) { LOG("%s", common_token_to_piece(ctx_tgt, id).c_str()); }

    while (n_gen < n_predict) {
        // 1) commit the anchor (id_next = target token @ n_dec) on the TARGET so its hidden enters the
        //    context. This gives the block BOTH the anchor's target hidden (context) and its token (below).
        common_batch_clear(tb);
        common_batch_add(tb, id_next, n_dec, { 0 }, true);
        if (llama_decode(ctx_tgt, tb) != 0) { LOG_ERR("%s: target decode failed\n", __func__); break; }
        feat.assign((size_t) n_embd_enc, 0.0f);
        if (!grab_features(1, 0)) { return 1; }
        if (!encode_append(feat.data(), 1, n_dec)) { break; } // context now covers [0..n_dec]
        LOG("%s", common_token_to_piece(ctx_tgt, id_next).c_str());
        n_gen++;
        const llama_token anchor = id_next;
        const bool anchor_eog = llama_vocab_is_eog(vocab, anchor);
        n_dec++;                                                // context = [0..n_dec-1]; anchor @ n_dec-1
        id_next = argmax(llama_get_logits_ith(ctx_tgt, 0));     // target token @ n_dec
        if (anchor_eog || n_gen >= n_predict) { break; }

        // 2) DFlash draft: block = [anchor token @ n_dec-1, masks @ [n_dec .. n_dec+B-1]] predicting the next
        //    B, conditioned on context [0..n_dec-1] (which now includes the anchor's target hidden @ n_dec-1).
        //    Clear the draft's (unused, out-of-band-context) KV first so the anchor position doesn't conflict.
        llama_memory_seq_rm(llama_get_memory(ctx_dft.get()), 0, -1, -1);
        llama_set_dflash_context(ctx_dft.get(), ctx_g.data(), (int32_t) ctx_pos.size(), ctx_pos.data());
        common_batch_clear(blk);
        common_batch_add(blk, anchor, n_dec - 1, { 0 }, true);
        for (int j = 0; j < block_size; ++j) { common_batch_add(blk, mask_tok, n_dec + j, { 0 }, true); }
        if (llama_decode(ctx_dft.get(), blk) != 0) { LOG_ERR("%s: draft decode failed\n", __func__); break; }

        std::vector<llama_token> draft(block_size);
        for (int j = 0; j < block_size; ++j) { draft[j] = argmax(llama_get_logits_ith(ctx_dft.get(), j + 1)); }

        // 3) advance the target greedily by up to B tokens [n_dec..], accumulating features, to compare
        feat.assign((size_t) block_size * n_embd_enc, 0.0f);
        std::vector<llama_token> tgt(block_size);
        int stop = block_size;
        for (int j = 0; j < block_size; ++j) {
            tgt[j] = id_next;
            LOG("%s", common_token_to_piece(ctx_tgt, id_next).c_str());
            n_gen++;
            common_batch_clear(tb);
            common_batch_add(tb, id_next, n_dec + j, { 0 }, true);
            if (llama_decode(ctx_tgt, tb) != 0) { LOG_ERR("%s: target decode failed\n", __func__); stop = j + 1; break; }
            if (!grab_features(1, j)) { return 1; }
            id_next = argmax(llama_get_logits_ith(ctx_tgt, 0));
            if (llama_vocab_is_eog(vocab, tgt[j]) || n_gen >= n_predict) { stop = j + 1; break; }
        }

        // 4) acceptance: draft[j] predicts token @ n_dec+j; compare to tgt[j] (longest prefix)
        int acc = 0; while (acc < stop && draft[acc] == tgt[acc]) { acc++; }
        tau_sum += acc; n_blocks++;

        // 5) grow the context with the `stop` decoded target tokens [n_dec .. n_dec+stop-1]
        if (!encode_append(feat.data(), stop, n_dec)) { break; }
        n_dec += stop;

        if (llama_vocab_is_eog(vocab, tgt[stop - 1])) { break; }
    }

    LOG("\n\n");
    const double mean_acc = n_blocks ? (double) tau_sum / n_blocks : 0.0;
    LOG_INF("%s: blocks=%lld  mean drafts accepted=%.3f / %d (%.1f%%)  |  acceptance length (tokens/cycle, "
            "incl. bonus) tau=%.3f  n_gen=%lld\n", __func__,
            (long long) n_blocks, mean_acc, block_size,
            n_blocks ? 100.0 * tau_sum / (n_blocks * block_size) : 0.0, mean_acc + 1.0, (long long) n_gen);

    llama_batch_free(pb);
    llama_batch_free(blk);
    llama_batch_free(tb);
    llama_backend_free();
    return 0;
}
