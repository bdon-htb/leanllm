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

    float temperature; // Sampling temperature. 0.0 = greedy sampling.
    int32_t top_k;     // Top-k sampling limit. <= 0 disables top-k filtering.
    float top_p;       // Top-p (nucleus) sampling threshold. Must be in (0, 1].

    uint32_t seed;     // RNG seed. LEANLLM_RANDOM_SEED uses a random seed.
} leanllm_chat_options;

typedef struct leanllm_message {
    enum leanllm_role role;
    const char *content;
} leanllm_message;

/**
 * @brief Callback invoked when generated text becomes available.
 *
 * The text buffer contains exactly @p length bytes and is not guaranteed
 * to be null-terminated. The buffer is owned by LeanLLM and is only valid
 * for the duration of the callback.
 *
 * @param text Generated text bytes.
 * @param length Number of bytes in the text buffer.
 * @param userdata User-provided pointer passed to leanllm_generate().
 * @return true to continue generation, or false to stop generation.
 */
typedef bool (*leanllm_stream_callback)(
    const char *text,
    size_t length,
    void *userdata
);

/**
 * @brief Initializes LeanLLM and its underlying inference backends.
 *
 * Must be called before loading models or creating chats.
 */
void leanllm_init(void);

/**
 * @brief Shuts down LeanLLM and its underlying inference backends.
 *
 * Should be called after all models and chats have been freed.
 */
void leanllm_shutdown(void);

/**
 * @brief Returns the default model loading options.
 *
 * @return The default model options.
 */
leanllm_model_options leanllm_model_default_options(void);

/**
 * @brief Returns the default chat and generation options.
 *
 * @return The default chat options.
 */
leanllm_chat_options leanllm_chat_default_options(void);

/**
 * @brief Loads a GGUF model from a file.
 *
 * @param path Path to the GGUF model file.
 * @param options Model loading options, or NULL to use the defaults.
 * @return A newly allocated model, or NULL if the model could not be loaded.
 */
leanllm_model *leanllm_model_load(const char *path, const leanllm_model_options *options);

/**
 * @brief Frees a loaded model.
 *
 * All chats using the model should be freed before calling this function.
 * Passing NULL is safe.
 *
 * @param model Model to free.
 */
void leanllm_model_free(leanllm_model *model);

/**
 * @brief Creates a chat context for a loaded model.
 *
 * The model must remain valid for the lifetime of the chat.
 *
 * @param model Model to use for inference.
 * @param options Chat and generation options, or NULL to use the defaults.
 * @return A newly allocated chat, or NULL if the chat could not be created.
 */
leanllm_chat *leanllm_chat_create(leanllm_model *model, const leanllm_chat_options *options);

/**
 * @brief Frees a chat context and its associated resources.
 *
 * Passing NULL is safe. This does not free the model used by the chat.
 *
 * @param chat Chat to free.
 */
void leanllm_chat_free(leanllm_chat *chat);

/**
 * @brief Generates a response from a sequence of chat messages.
 *
 * Generated text is streamed to the callback as it becomes available.
 * Generation stops when the model produces an end-of-generation token,
 * the maximum token count is reached, the context is exhausted, or the
 * callback returns false.
 *
 * The message contents only need to remain valid for the duration of this
 * call.
 *
 * @param chat Chat context to use for generation.
 * @param messages Array of input chat messages.
 * @param message_count Number of messages in the array.
 * @param callback Function called for each generated piece of text.
 * @param userdata User-provided pointer passed unchanged to the callback.
 * @return LEANLLM_OK on success, or an error code on failure.
 */
leanllm_error leanllm_generate(leanllm_chat *chat, const leanllm_message *messages, size_t message_count, leanllm_stream_callback callback, void *userdata);

/**
 * @brief Returns a human-readable description of a LeanLLM error code.
 *
 * The returned string is owned by LeanLLM and must not be modified or freed.
 *
 * @param error Error code to describe.
 * @return A null-terminated string describing the error.
 */
const char *leanllm_error_string(leanllm_error error);

#endif