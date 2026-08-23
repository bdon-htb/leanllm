#ifndef LEANLLM_H
#define LEANLLM_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>


#define LEANLLM_RANDOM_SEED UINT32_MAX

typedef enum leanllm_error {
    LEANLLM_OK = 0,
    LEANLLM_ERROR_INVALID_ARGUMENT,
    LEANLLM_ERROR_OUT_OF_MEMORY,
    LEANLLM_ERROR_MODEL_LOAD,
    LEANLLM_ERROR_CONTEXT,
    LEANLLM_ERROR_TEMPLATE,
    LEANLLM_ERROR_TOKENIZE,
    LEANLLM_ERROR_DECODE,
    LEANLLM_ERROR_TOKEN_TO_PIECE
} leanllm_error;

typedef enum leanllm_role {
    LEANLLM_ROLE_SYSTEM,
    LEANLLM_ROLE_USER,
    LEANLLM_ROLE_ASSISTANT
} leanllm_role;

typedef struct leanllm_model leanllm_model;
typedef struct leanllm_chat leanllm_chat;

typedef struct leanllm_model_options {
    int32_t gpu_layers; // Number of model layers to offload to the GPU. 0 = no GPU offload, negative = all layers.
} leanllm_model_options;

typedef struct leanllm_chat_options {
    uint32_t context_size;
    uint32_t batch_size;
    uint32_t max_tokens;

    float temperature;
    int32_t top_k;
    float top_p;

    uint32_t seed;
} leanllm_chat_options;

typedef struct leanllm_message {
    enum leanllm_role role;
    const char *content;
} leanllm_message;

typedef bool (*leanllm_stream_callback)(
    const char *text,
    size_t length,
    void *userdata
);

void leanllm_init(void);

void leanllm_shutdown(void);

leanllm_model_options leanllm_model_default_options(void);

leanllm_chat_options leanllm_chat_default_options(void);

leanllm_model *leanllm_model_load(const char *path, const leanllm_model_options *options);

void leanllm_model_free(leanllm_model *model);

leanllm_chat *leanllm_chat_create(leanllm_model *model, const leanllm_chat_options *options);

void leanllm_chat_free(leanllm_chat *chat);

leanllm_error leanllm_generate(leanllm_chat *chat, const leanllm_message *messages, size_t message_count, leanllm_stream_callback callback, void *userdata);

const char *leanllm_error_string(leanllm_error error);

#endif