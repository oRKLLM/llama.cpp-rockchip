#include "llama.h"
#include "ggml.h"
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) {
        printf("Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }
    
    llama_backend_init();
    
    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 1024;
    llama_context * ctx = llama_init_from_model(model, cparams);
    
    auto tokenize = [&](const std::string & text) {
        std::vector<llama_token> toks(1024);
        int n = llama_tokenize(llama_model_get_vocab(model), text.c_str(), text.length(), toks.data(), toks.size(), true, true);
        toks.resize(n);
        return toks;
    };
    
    auto toks = tokenize("Hello world. The moment has come to test state sequence saving.");
    
    auto run_test = [&](bool do_save, const char* name) {
        printf("\n--- RUNNING TEST: %s (save=%d) ---\n", name, do_save);
        llama_memory_clear(llama_get_memory(ctx), true);
        
        int n_past = 0;
        
        // Prefill
        int64_t t0 = ggml_time_us();
        llama_batch batch = llama_batch_get_one(toks.data(), toks.size());
        llama_decode(ctx, batch);
        n_past += toks.size();
        int64_t t1 = ggml_time_us();
        printf("Prefill %d tokens: %.2f ms\n", (int)toks.size(), (t1 - t0) / 1000.0);
        
        if (do_save) {
            int64_t ts0 = ggml_time_us();
            llama_state_seq_save_file(ctx, "test_state.bin", 0, toks.data(), toks.size());
            int64_t ts1 = ggml_time_us();
            printf("Save state: %.2f ms\n", (ts1 - ts0) / 1000.0);
        }
        
        // Decode
        llama_batch b = llama_batch_init(4, 0, 1);
        for (int i = 0; i < 4; i++) {
            b.n_tokens = 1;
            b.token[0] = 12;
            b.pos[0] = n_past;
            b.n_seq_id[0] = 1;
            b.seq_id[0][0] = 0;
            b.logits[0] = true;
            
            int64_t td0 = ggml_time_us();
            llama_decode(ctx, b);
            int64_t td1 = ggml_time_us();
            n_past++;
            
            printf("Decode step %d: %.2f ms\n", i, (td1 - td0) / 1000.0);
        }
        llama_batch_free(b);
    };
    
    run_test(false, "A (No save)");
    run_test(true, "B (With save)");
    
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    
    return 0;
}
