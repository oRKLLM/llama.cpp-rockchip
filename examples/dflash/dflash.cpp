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

#include <chrono>
#include <cstdint>
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

    // Skip-ahead speculative decode. State: ctx_g / target-KV cover positions [0..n_ctx_tok-1]; `anchor` is
    // the token @ n_ctx_tok-1 (its target hidden is in ctx_g); `t0` is the target's greedy token for the
    // next position n_ctx_tok (saved from the last target decode). The DFlash draft speculates the next B
    // tokens; the target verifies all B in ONE forward (M=B, grouped -> NPU); we accept the longest matching
    // prefix + the target's correction (bonus). This is the loop that ports into oRKLLM's addon run_dflash.
    int32_t     n_ctx_tok = n_prompt;
    llama_token anchor    = inp[n_prompt - 1];
    llama_token t0        = argmax(llama_get_logits_ith(ctx_tgt, n_prompt - 1)); // target token @ n_ctx_tok

    const int n_predict = params.n_predict > 0 ? params.n_predict : 128;
    int64_t n_cycles = 0, acc_sum = 0, n_gen = 0, target_forwards = 0;

    llama_batch blk = llama_batch_init(block_size + 1, 0, 1); // anchor + block_size mask tokens
    llama_batch vb  = llama_batch_init(block_size, 0, 1);     // verify batch (B draft tokens, one forward)
    std::vector<uint8_t> ckpt;                               // target-state checkpoint for spec rollback

    LOG("\n");
    for (auto id : inp) { LOG("%s", common_token_to_piece(ctx_tgt, id).c_str()); }

    const auto t_start = std::chrono::steady_clock::now();
    bool eog = false;
    while (n_gen < n_predict && !eog) {
        // 1) DFlash draft: [anchor @ n_ctx_tok-1, masks @ n_ctx_tok..n_ctx_tok+B-1] over context [0..n_ctx_tok-1]
        //    (incl. the anchor's target hidden). draft[j] speculates position n_ctx_tok+j. Clear the draft's
        //    (out-of-band-context) KV first so the reused anchor position doesn't conflict.
        llama_memory_seq_rm(llama_get_memory(ctx_dft.get()), 0, -1, -1);
        llama_set_dflash_context(ctx_dft.get(), ctx_g.data(), (int32_t) ctx_pos.size(), ctx_pos.data());
        common_batch_clear(blk);
        common_batch_add(blk, anchor, n_ctx_tok - 1, { 0 }, true);
        for (int j = 0; j < block_size; ++j) { common_batch_add(blk, mask_tok, n_ctx_tok + j, { 0 }, true); }
        if (llama_decode(ctx_dft.get(), blk) != 0) { LOG_ERR("%s: draft decode failed\n", __func__); break; }
        std::vector<llama_token> d(block_size);
        for (int j = 0; j < block_size; ++j) { d[j] = argmax(llama_get_logits_ith(ctx_dft.get(), j + 1)); }

        // 2) checkpoint the target, then VERIFY: one forward over the B drafts (M=B, grouped -> NPU).
        //    logits[j] predict position n_ctx_tok+j+1. (M-RoPE KV can't partial-seq_rm, so we roll back via a
        //    saved state checkpoint — the pattern speculative-simple uses when seq_rm isn't FULL.)
        const size_t ck_sz = llama_state_seq_get_size(ctx_tgt, 0);
        ckpt.resize(ck_sz);
        llama_state_seq_get_data(ctx_tgt, ckpt.data(), ck_sz, 0);

        common_batch_clear(vb);
        for (int j = 0; j < block_size; ++j) { common_batch_add(vb, d[j], n_ctx_tok + j, { 0 }, true); }
        if (llama_decode(ctx_tgt, vb) != 0) { LOG_ERR("%s: verify decode failed\n", __func__); break; }
        target_forwards++;

        // 3) accept longest prefix: d[j] iff == target greedy t[j] (t[0]=t0; t[j>=1]=argmax(verify logits[j-1])).
        int acc = 0;
        while (acc < block_size) {
            const llama_token tj = (acc == 0) ? t0 : argmax(llama_get_logits_ith(ctx_tgt, acc - 1));
            if (d[acc] != tj) { break; }
            acc++;
        }
        const llama_token bonus = (acc == 0) ? t0 : argmax(llama_get_logits_ith(ctx_tgt, acc - 1));

        // 4) roll back the speculative verify (restore checkpoint) and commit [accepted..., bonus] cleanly
        //    (M=acc+1) to set the KV + extract their target hiddens for the context.
        llama_state_seq_set_data(ctx_tgt, ckpt.data(), ck_sz, 0);
        common_batch_clear(blk); // reuse (cap block_size+1 >= acc+1)
        for (int j = 0; j < acc; ++j) { common_batch_add(blk, d[j], n_ctx_tok + j, { 0 }, true); }
        common_batch_add(blk, bonus, n_ctx_tok + acc, { 0 }, true);
        if (llama_decode(ctx_tgt, blk) != 0) { LOG_ERR("%s: commit decode failed\n", __func__); break; }
        target_forwards++;
        feat.assign((size_t) (acc + 1) * n_embd_enc, 0.0f);
        if (!grab_features(acc + 1, 0)) { return 1; }       // hiddens for [n_ctx_tok..n_ctx_tok+acc]
        t0 = argmax(llama_get_logits_ith(ctx_tgt, acc));    // bonus @ index acc predicts n_ctx_tok+acc+1

        // 5) emit accepted drafts + bonus; grow the context with their target hiddens.
        for (int j = 0; j < acc; ++j) { LOG("%s", common_token_to_piece(ctx_tgt, d[j]).c_str()); }
        LOG("%s", common_token_to_piece(ctx_tgt, bonus).c_str());
        if (!encode_append(feat.data(), acc + 1, n_ctx_tok)) { break; }

        n_gen += acc + 1; acc_sum += acc; n_cycles++;
        for (int j = 0; j < acc; ++j) { if (llama_vocab_is_eog(vocab, d[j])) { eog = true; } }
        if (llama_vocab_is_eog(vocab, bonus)) { eog = true; }
        anchor = bonus; n_ctx_tok += acc + 1;
    }

    const double dt_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    LOG("\n\n");
    const double mean_acc = n_cycles ? (double) acc_sum / n_cycles : 0.0;
    LOG_INF("%s: cycles=%lld  mean drafts accepted=%.3f / %d  |  acceptance length tau=%.3f tokens/cycle  |  "
            "decode %lld tok in %.2fs = %.2f tok/s  (target forwards=%lld -> %.2f tokens/forward)\n", __func__,
            (long long) n_cycles, mean_acc, block_size, mean_acc + 1.0,
            (long long) n_gen, dt_s, dt_s > 0 ? n_gen / dt_s : 0.0,
            (long long) target_forwards, target_forwards ? (double) n_gen / target_forwards : 0.0);

    llama_batch_free(pb);
    llama_batch_free(blk);
    llama_batch_free(vb);
    llama_backend_free();
    return 0;
}
