#include "models.h"

#include "ggml-backend.h"

// Custom decoder graph input: feeds the fused target context [n_embd, ctx_len] and its RoPE positions,
// filled out-of-band from cparams (set by the speculative driver via llama_set_dflash_context). Every
// draft layer projects this same context through its own attn_k/attn_v to form the injected context K/V.
class dflash_input_ctx : public llm_graph_input_i {
public:
    dflash_input_ctx(const llama_cparams & cparams) : cparams(cparams) {}

    void set_input(const llama_ubatch * /*ubatch*/) override {
        if (ctx_embd && cparams.dflash_ctx) {
            ggml_backend_tensor_set(ctx_embd, cparams.dflash_ctx, 0, ggml_nbytes(ctx_embd));
        }
        if (ctx_pos && cparams.dflash_ctx_pos) {
            ggml_backend_tensor_set(ctx_pos, cparams.dflash_ctx_pos, 0, ggml_nbytes(ctx_pos));
        }
    }

    ggml_tensor * ctx_embd = nullptr; // F32 [n_embd, ctx_len]
    ggml_tensor * ctx_pos  = nullptr; // I32 [ctx_len]
    const llama_cparams & cparams;
};

// DFlash — block-diffusion speculative-decode drafter.
// EAGLE-3 relative: target-hidden-conditioned draft head, but a multi-layer Qwen3 body, N target
// layers fused via `fc`->`enc_output_norm`, per-layer target-context K/V injection (P2b), non-causal
// block attention, Qwen3 q/k-norm, borrowed tok_embd/output via cparams.ctx_other.
//
// Two graphs (templated on params.gtype):
//   graph<true>  ENCODER: target features [N*n_embd_tgt, n_tokens] -> fc -> enc_output_norm -> t_h_nextn
//                (read back by the driver via llama_get_embeddings_nextn -> the fused context g_embd).
//   graph<false> DECODER: block tokens (+ P2b: injected context) -> Qwen3 body -> borrowed lm_head -> logits.
//
// P2a (this commit): encoder + decoder BODY (Qwen3 layers, non-causal, borrowed embd/head). The per-layer
// context K/V injection + custom block/context mask is P2b (see TODO in graph<false>).

void llama_model_dflash::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    if (!ml.get_arr(LLM_KV_TARGET_LAYERS, target_layer_ids, false)) {
        throw std::runtime_error("DFlash model requires 'target_layers' in GGUF metadata");
    }
    const int64_t n_tgt_layers = (int64_t) target_layer_ids.size();
    if (n_tgt_layers < 1) {
        throw std::runtime_error("DFlash requires at least one target layer id");
    }

    // DFlash GGUFs carry no target_hidden_size KV, so derive it from the fc fusion tensor:
    // fc is {n_embd_inp, n_embd} with n_embd_inp = n_tgt_layers * n_embd_tgt.
    const struct ggml_tensor * fc_meta = ml.get_tensor_meta("fc.weight");
    if (!fc_meta) {
        throw std::runtime_error("DFlash model requires the 'fc' feature-fusion tensor");
    }
    hparams.n_embd_inp_enc_impl = (uint32_t) fc_meta->ne[0];

    ml.get_key(LLM_KV_BLOCK_SIZE, block_size, false); // block-parallel draft width (optional)

    // block-diffusion drafting is non-causal within a block
    hparams.causal_attn = false;

    LLAMA_LOG_INFO("%s: DFlash n_target_layers = %lld, n_embd_inp_enc = %u, block_size = %u\n",
            __func__, (long long) n_tgt_layers, hparams.n_embd_inp_enc_impl, block_size);

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_dflash::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_embd_inp = hparams.n_embd_inp_enc();

    // optional d2t (draft->target vocab remap); default: share the target vocab
    int64_t n_draft_vocab = n_vocab;
    const struct ggml_tensor * d2t_meta = ml->get_tensor_meta("d2t");
    if (d2t_meta) {
        n_draft_vocab = d2t_meta->ne[0];
        d2t = create_tensor(tn(LLM_TENSOR_D2T), {n_draft_vocab}, 0);
        LLAMA_LOG_INFO("%s: DFlash using d2t mapping (draft_vocab = %lld)\n", __func__, (long long) n_draft_vocab);
    } else {
        d2t = nullptr;
    }

    // feature fusion (N target layers concatenated -> draft hidden) + its shared hidden_norm
    fc              = create_tensor(tn(LLM_TENSOR_FC,              "weight"), {n_embd_inp, n_embd}, 0);
    enc_output_norm = create_tensor(tn(LLM_TENSOR_ENC_OUTPUT_NORM, "weight"), {n_embd}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_draft_vocab}, TENSOR_NOT_REQUIRED);

    // optional own token embeddings; else borrow the target's via ctx_other at graph time
    const struct ggml_tensor * tok_embd_meta =
        ml->get_tensor_meta(tn(LLM_TENSOR_TOKEN_EMBD, "weight").str().c_str());
    if (tok_embd_meta) {
        const int64_t n_tok_vocab = tok_embd_meta->ne[1];
        tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_tok_vocab}, 0);
        LLAMA_LOG_INFO("%s: DFlash using its own token_embd (vocab = %lld)\n", __func__, (long long) n_tok_vocab);
    }

    // Qwen3-style draft body (standard attention dims + per-head q/k RMS norm)
    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_dflash::build_arch_graph(const llm_graph_params & params) const {
    switch (params.gtype) {
        case LLM_GRAPH_TYPE_ENCODER:
            return std::make_unique<graph<true>>(*this, params);
        case LLM_GRAPH_TYPE_DEFAULT:
        case LLM_GRAPH_TYPE_DECODER:
            return std::make_unique<graph<false>>(*this, params);
        default:
            GGML_ABORT("invalid graph type");
    };
}

template <>
ggml_tensor * llama_model_dflash::graph<true>::build_inp_embd_enc() const {
    auto inp_target = std::make_unique<llm_graph_input_embd>(hparams.n_embd_inp_enc());
    inp_target->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp_enc(), n_tokens);
    ggml_set_input(inp_target->embd);
    ggml_tensor * cur = inp_target->embd;
    cb(cur, "inp_embd", -1);
    res->add_input(std::move(inp_target));
    return cur;
}

// ENCODER: fuse the N concatenated target layers into the shared context g_embd.
// Input:  target_features [N*n_embd_tgt, n_tokens] (fed via ubatch->embd by the driver)
// Output: g_embd [n_embd, n_tokens] stored in res->t_h_nextn (read via llama_get_embeddings_nextn).
template <>
llama_model_dflash::graph<true>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur = build_inp_embd_enc();

    cur = build_lora_mm(model.fc, cur);
    cb(cur, "fc_out", -1);

    // hidden_norm on the fused context (applied once; reused by every decoder layer's K/V injection)
    cur = build_norm(cur, model.enc_output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "enc_output_norm", -1);

    ggml_set_output(cur);
    res->t_h_nextn = cur;

    ggml_build_forward_expand(gf, cur);
}

// DECODER: run the Qwen3 draft body over the block tokens and project to draft logits.
// P2a: body only (non-causal self-attention over the block). P2b will inject the target context K/V
// (from res->t_h_nextn, projected per-layer by attn_k/attn_v) ahead of each layer's block K/V, with a
// custom [n_ctx+block, block] non-causal mask.
template <>
llama_model_dflash::graph<false>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // token embeddings — borrow the target's if the draft has none
    auto * tok_embd = model.tok_embd;
    if (tok_embd == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr && "DFlash decoder needs token embeddings (own or target's)");
        const auto * model_other = llama_get_model(cparams.ctx_other);
        GGML_ASSERT(model_other->tok_embd != nullptr);
        tok_embd = model_other->tok_embd;
    }

    inpL = build_inp_embd(tok_embd);

    ggml_tensor * inp_pos = build_inp_pos();

    // DFlash target context (P2b): [n_embd, ctx_len], injected into EVERY layer's K/V. ctx_len is set
    // out-of-band per decode; 0 at reserve (falls back to plain block self-attention).
    const int32_t dflash_ctx_len = cparams.dflash_ctx_len;

    // non-causal (block-diffusion) attention, no KV cache — only for the ctx_len==0 fallback. When context
    // is injected we build attention manually via build_attn_mha, so registering an unused no_cache input
    // here would leave its mask tensor unallocated (dead) and trip the backend-buffer assert.
    llm_graph_input_attn_no_cache * inp_attn = (dflash_ctx_len > 0) ? nullptr : build_attn_inp_no_cache();
    ggml_tensor * ctx_hidden = nullptr;
    ggml_tensor * ctx_pos    = nullptr;
    if (dflash_ctx_len > 0) {
        auto inp_ctx = std::make_unique<dflash_input_ctx>(cparams);
        inp_ctx->ctx_embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, dflash_ctx_len);
        ggml_set_input(inp_ctx->ctx_embd);
        inp_ctx->ctx_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, dflash_ctx_len);
        ggml_set_input(inp_ctx->ctx_pos);
        ctx_hidden = inp_ctx->ctx_embd;
        ctx_pos    = inp_ctx->ctx_pos;
        res->add_input(std::move(inp_ctx));
    }

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention (Qwen3 q/k-norm), with the target context prepended to each layer's K/V.
        {
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur, n_embd_head, n_head, n_head_kv, il);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            if (ctx_hidden) {
                // context K/V from the shared fused context, projected by THIS layer's k/v (+ k-norm, RoPE)
                ggml_tensor * ctxK = build_lora_mm(model.layers[il].wk, ctx_hidden);
                ctxK = ggml_reshape_3d(ctx0, ctxK, n_embd_head, n_head_kv, dflash_ctx_len);
                ctxK = build_norm(ctxK, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
                ctxK = ggml_rope_ext(ctx0, ctxK, ctx_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                        freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

                ggml_tensor * ctxV = build_lora_mm(model.layers[il].wv, ctx_hidden);
                ctxV = ggml_reshape_3d(ctx0, ctxV, n_embd_head, n_head_kv, dflash_ctx_len);

                // K/V = [context | block]; single block -> full (non-causal) attention (mask = nullptr)
                ggml_tensor * Kfull = ggml_concat(ctx0, ctxK, Kcur, 2);
                ggml_tensor * Vfull = ggml_concat(ctx0, ctxV, Vcur, 2);

                cur = build_attn_mha(Qcur, Kfull, Vfull, nullptr, nullptr, nullptr, nullptr, kq_scale, il);
                cur = build_lora_mm(model.layers[il].wo, cur);
            } else {
                cur = build_attn(inp_attn, model.layers[il].wo, NULL, NULL,
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            }
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur  = ggml_add(ctx0, cur, ffn_inp);
        inpL = cur;
    }

    cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head — borrow the target's if the draft has none
    auto * output = model.output;
    if (output == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr && "DFlash decoder needs an output projection (own or target's)");
        const auto * model_other = llama_get_model(cparams.ctx_other);
        GGML_ASSERT(model_other->output != nullptr);
        output = model_other->output;
    }
    cur = build_lora_mm(output, cur);

    // draft->target vocab remap (scatter draft logits into the full target vocab)
    if (model.d2t) {
        const int64_t n_draft_vocab = cur->ne[0];
        const int64_t n_outputs     = cur->ne[1];
        const int64_t n_vocab_full  = (int64_t) model.vocab.n_tokens();

        GGML_ASSERT(model.d2t->type == GGML_TYPE_I64);
        GGML_ASSERT(model.d2t->ne[0] == n_draft_vocab);

        ggml_tensor * logits = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_vocab_full, n_outputs), -INFINITY);
        cur = ggml_set_rows(ctx0, logits,
                ggml_reshape_3d(ctx0, cur,       1,             n_draft_vocab, n_outputs),
                ggml_reshape_3d(ctx0, model.d2t, n_draft_vocab, 1,             1));
        cur = ggml_reshape_2d(ctx0, cur, n_vocab_full, n_outputs);
    }

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
