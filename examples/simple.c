/*
 * simple.c
 *
 * Basic LeanLLM example.
 *
 * Loads a model, creates a chat context, generates a response
 * from a short conversation, and streams the generated text to stdout.
 */

#include "leanllm.h"

#include <stdio.h>

static bool print_piece(const char *text, size_t length, void *userdata)
{
    (void)userdata;

    fprintf(stderr, "[callback: %zu bytes]\n", length);

    fwrite(text, 1, length, stdout);
    fflush(stdout);

    return true;
}

int main(void)
{
    fprintf(stderr, "[init]\n");
    leanllm_init();

    fprintf(stderr, "[load model]\n");
    leanllm_model *model =
        leanllm_model_load("./models/gemma-3-270m-it-Q8_0.gguf", NULL);

    if (model == NULL)
    {
        fprintf(stderr, "[MODEL LOAD FAILED]\n");
        leanllm_shutdown();
        return 1;
    }

    fprintf(stderr, "[create chat]\n");

    leanllm_chat_options opts = leanllm_chat_default_options();
    opts.temperature = 0.8f;
    opts.top_k = 40;
    opts.top_p = 0.95f;

    leanllm_chat *chat = leanllm_chat_create(model, &opts);

    if (chat == NULL)
    {
        fprintf(stderr, "[CHAT CREATE FAILED]\n");
        leanllm_model_free(model);
        leanllm_shutdown();
        return 1;
    }

    leanllm_message messages[] = {
        {.role = LEANLLM_ROLE_USER,
         .content = "My name is Bob and I like cats."},
        {.role = LEANLLM_ROLE_ASSISTANT,
         .content = "Nice to meet you, Bob. Cats are great."},
        {.role = LEANLLM_ROLE_USER,
         .content = "What is my name?"},
    };

    fprintf(stderr, "[generate]\n");

    leanllm_error err =
        leanllm_generate(chat, messages, 3, print_piece, NULL);

    if (err != LEANLLM_OK) {
        fprintf(stderr, "LeanLLM error: %s\n", leanllm_error_string(err));
    }

    fprintf(stderr, "[generate returned %d]\n", (int)err);

    leanllm_chat_free(chat);
    leanllm_model_free(model);
    leanllm_shutdown();

    fprintf(stderr, "[done]\n");

    return err == LEANLLM_OK ? 0 : 1;
}