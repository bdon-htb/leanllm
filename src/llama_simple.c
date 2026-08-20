/*
 * simple.c
 *
 * Minimal llama.cpp inference example written in C.
 *
 * Based on llama.cpp's C++ simple example:
 * https://github.com/ggml-org/llama.cpp/blob/master/examples/simple/simple.cpp
 *
 * Adapted to use C-style memory management and the owning llama_batch API.
 * Demonstrates the basic inference pipeline:
 * model loading -> tokenization -> context creation -> sampling ->
 * batch decoding -> token generation -> cleanup.
 *
 * Intended as a reference implementation for development of the leanllm
 * wrapper.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <llama.h>


int main(void) {
    bool error = false;
    const char *path = "./models/SmolLM2-135M-Instruct-Q4_K_M.gguf";
    const char *prompt = "Hello my name is";

    int prompt_len = (int)strlen(prompt);

    // Number of tokens to predict.
    int n_predict_tokens = 32;

    // Initialize pointers ahead of time for easy cleanup.
    struct llama_model *model = NULL;
    llama_token *prompt_tokens = NULL;
    struct llama_context *ctx = NULL;
    struct llama_sampler *smpl = NULL;
    llama_batch batch;
    bool batch_initialized = false;

    // Load dynamic backends
    ggml_backend_load_all();

    // Initialize the model.
    struct llama_model_params model_params = llama_model_default_params();
    model = llama_model_load_from_file(path, model_params);

    if (model == NULL) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        error = true;
        goto cleanup;
    }

    // Tokenize the prompt and get the number of tokens in it.
    const struct llama_vocab *vocab = llama_model_get_vocab(model);

    // llama_tokenize with no token buffer returns a negative value indicating how many tokens were required.
    int token_count = llama_tokenize(vocab, prompt, prompt_len, NULL, 0, true, true);

    if (token_count >= 0) {
        fprintf(stderr, "%s: error: unexpected tokenize result\n", __func__);
        error = true;
        goto cleanup;
    }

    const int n_prompt_tokens = -token_count;

    // Allocate space for the tokens.
    prompt_tokens = malloc(n_prompt_tokens * sizeof(*prompt_tokens));

    if (prompt_tokens == NULL) {
        fprintf(stderr, "%s: error: failed to allocate prompt tokens\n", __func__);
        error = true;
        goto cleanup;
    }

    // Tokenize prompt.
    int actual = llama_tokenize(vocab, prompt, prompt_len, prompt_tokens, n_prompt_tokens, true, true); 
    
    if (actual != n_prompt_tokens) {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        error = true;
        goto cleanup;
    }

    // Initialize the context.
    struct llama_context_params ctx_params = llama_context_default_params();
    // n_ctx is the context size.
    ctx_params.n_ctx = n_prompt_tokens + n_predict_tokens - 1;
    // n_batch is the maximum number of tokens that can be processed in a single call to llama_decode.
    ctx_params.n_batch = n_prompt_tokens;
    // Enable performance counters.
    ctx_params.no_perf = false;

    ctx = llama_init_from_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr, "%s: error: failed to create the llama_context\n", __func__);
        error = true;
        goto cleanup;
    }

    // Initialize the sampler.
    struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;

    // Create sampler chain.
    smpl = llama_sampler_chain_init(sparams);

    if (smpl == NULL) {
        fprintf(stderr, "%s: error: failed to initialize sampler\n", __func__);
        error = true;
        goto cleanup;
    }

    // Add greedy sampler rule.
    struct llama_sampler *greedy = llama_sampler_init_greedy();

    if (greedy == NULL) {
        fprintf(stderr, "%s: error: failed to initialize greedy sampler\n", __func__);
        error = true;
        goto cleanup;
    }

    llama_sampler_chain_add(smpl, greedy);

    // Print the prompt token-by-token.

    for (int i = 0; i < n_prompt_tokens; i++) {
        char buf[128];

        llama_token id = prompt_tokens[i];

        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);

        if (n < 0) {
            fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
            error = true;
            goto cleanup;
        }

        fwrite(buf, 1, n, stdout);
    }

    // Prepare a batch for the prompt.
    // We will skip the encoding step since our model doesn't have one.

    batch = llama_batch_init(n_prompt_tokens, 0, 1);
    batch_initialized = true;

    // Fill batch with the prompt.
    batch.n_tokens = n_prompt_tokens;

    for (int i = 0; i < n_prompt_tokens; i++) {
        batch.token[i] = prompt_tokens[i];
        batch.pos[i] = i;

        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;

        batch.logits[i] = (i == n_prompt_tokens - 1);
    }

    // Main loop.

    const int64_t t_main_start = ggml_time_us();
    int n_decode = 0;

    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt_tokens + n_predict_tokens; ) {
        
        // Decode whatever is currently in the batch.
        int decode_result = llama_decode(ctx, batch);
        
        if (decode_result != 0) {
            fprintf(stderr, "%s : failed to decode, return code %d\n", __func__, decode_result);
            error = true;
            goto cleanup;
        }

        n_pos += batch.n_tokens;

        // Sample from the logits produced by that decode.
        llama_token new_token_id  = llama_sampler_sample(smpl, ctx, -1);

        // Is it an end of generation?
        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        char buf[128];

        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);

        if (n < 0) {
            fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
            error = true;
            goto cleanup;
        }

        fwrite(buf, 1, n, stdout);
        fflush(stdout);

        // The next decode only needs the newly generated token.
        batch.n_tokens = 1;
        batch.token[0] = new_token_id;
        batch.pos[0] = n_pos;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;

        n_decode++;
    }

    printf("\n");

    const int64_t t_main_end = ggml_time_us();

    fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));
    
    fprintf(stderr, "\n");
    llama_perf_sampler_print(smpl);
    llama_perf_context_print(ctx);
    fprintf(stderr, "\n");

cleanup:
    if (batch_initialized) {
        llama_batch_free(batch);
    }

    if (smpl != NULL) {
        llama_sampler_free(smpl);
    }

    if (ctx != NULL) {
        llama_free(ctx);
    }

    free(prompt_tokens);

    if (model != NULL) {
        llama_model_free(model);
    }

    return error ? 1 : 0;
}