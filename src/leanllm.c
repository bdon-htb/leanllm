#include "leanllm.h"

#include <stdio.h>
#include <stdlib.h>

#include <llama.h>

#define LEANLLM_INITIAL_MESSAGE_CAPACITY 16
#define LEANLLM_INITIAL_PROMPT_CAPACITY 1024
#define LEANLLM_INITIAL_TOKEN_CAPACITY 256
#define LEANLLM_INITIAL_TOKEN_PIECE_CAPACITY 128

struct leanllm_model {
    struct llama_model *model;
    const struct llama_vocab *vocab;
};

struct leanllm_chat {
    leanllm_model *model;
    const char *template;

    uint32_t context_size;
    struct llama_context *ctx;
    struct llama_sampler *sampler;

    uint32_t batch_size;

    uint32_t max_tokens;

    llama_chat_message *message_buffer;
    size_t message_capacity;

    char *prompt_buffer;
    size_t prompt_capacity;

    llama_token *token_buffer;
    size_t token_capacity;

    char *piece_buffer;
    size_t piece_capacity;
};

static const char *get_role_string(leanllm_role role) {
    switch (role) {
    case LEANLLM_ROLE_SYSTEM:
        return "system";
    case LEANLLM_ROLE_USER:
        return "user";
    case LEANLLM_ROLE_ASSISTANT:
        return "assistant";
    default:
        return NULL;
    }
}

void leanllm_init(void) {
    ggml_backend_load_all();
    llama_backend_init();
}

void leanllm_shutdown(void) {
    llama_backend_free();
}

leanllm_model_options leanllm_model_default_options(void) {
    return (leanllm_model_options) {
        .gpu_layers = 0
    };
}

leanllm_chat_options leanllm_chat_default_options(void) {
    return (leanllm_chat_options){
        .context_size = 2048,
        .batch_size = 512,
        .max_tokens = 128,
        .temperature = 0.0f,
        .top_k = 40,
        .top_p = 0.95f,
        .seed = LEANLLM_RANDOM_SEED
    };
}

leanllm_model *leanllm_model_load(const char *path, const leanllm_model_options *options) {
    if (path == NULL) {
        return NULL;
    }

    leanllm_model_options opts = (options != NULL) ? *options : leanllm_model_default_options();

    struct llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = opts.gpu_layers;

    struct llama_model *llama_model = llama_model_load_from_file(path, model_params);

    if (llama_model == NULL) {
        return NULL;
    }

    leanllm_model *model = malloc(sizeof(*model));

    if (model == NULL) {
        llama_model_free(llama_model);
        return NULL;
    }

    model->model = llama_model;
    model->vocab = llama_model_get_vocab(llama_model);

    return model;
}

void leanllm_model_free(leanllm_model *model) {
    if (model == NULL) {
        return;
    }

    llama_model_free(model->model);
    free(model);
}

leanllm_chat *leanllm_chat_create(leanllm_model *model, const leanllm_chat_options *options) {
    if (model == NULL) {
        return NULL;
    }

    leanllm_chat_options opts = (options != NULL) ? *options : leanllm_chat_default_options();

    if (opts.context_size == 0 ||
        opts.batch_size == 0 ||
        opts.max_tokens == 0 ||
        opts.temperature < 0.0f ||
        opts.top_p <= 0.0f ||
        opts.top_p > 1.0f) {
        return NULL;
    }

    leanllm_chat *chat = malloc(sizeof(*chat));

    if (chat == NULL) {
        return NULL;
    }

    chat->model = model;
    chat->context_size = opts.context_size;
    chat->ctx = NULL;
    chat->sampler = NULL;
    chat->batch_size = opts.batch_size;
    chat->max_tokens = opts.max_tokens;
    chat->message_buffer = NULL;
    chat->message_capacity = 0;
    chat->prompt_buffer = NULL;
    chat->prompt_capacity = 0;
    chat->token_buffer = NULL;
    chat->token_capacity = 0;
    chat->piece_buffer = NULL;
    chat->piece_capacity = 0;

    struct llama_context_params ctx_params = llama_context_default_params();

    ctx_params.n_ctx = opts.context_size;
    ctx_params.n_batch = opts.batch_size;

    chat->ctx = llama_init_from_model(model->model, ctx_params);

    if (chat->ctx == NULL) {
        free(chat);
        return NULL;
    }

    const char *tmpl = llama_model_chat_template(model->model, NULL);

    if (tmpl == NULL)
    {
        llama_free(chat->ctx);
        free(chat);
        return NULL;
    }

    chat->template = tmpl;

    struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();

    chat->sampler = llama_sampler_chain_init(sparams);

    if (chat->sampler == NULL) {
        llama_free(chat->ctx);
        free(chat);
        return NULL;
    }

    if (opts.temperature == 0.0f) {
        struct llama_sampler *greedy = llama_sampler_init_greedy();

        if (greedy == NULL) {
            llama_sampler_free(chat->sampler);
            llama_free(chat->ctx);
            free(chat);
            return NULL;
        }

        llama_sampler_chain_add(chat->sampler, greedy);
    } 
    else {
        struct llama_sampler *top_k = llama_sampler_init_top_k(opts.top_k);
        struct llama_sampler *top_p = llama_sampler_init_top_p(opts.top_p, 1);
        struct llama_sampler *temp = llama_sampler_init_temp(opts.temperature);
        struct llama_sampler *dist = llama_sampler_init_dist(opts.seed);

        if (top_k == NULL || top_p == NULL || temp == NULL || dist == NULL) {
            if (top_k != NULL) {
                llama_sampler_free(top_k);
            }

            if (top_p != NULL) {
                llama_sampler_free(top_p);
            }

            if (temp != NULL) {
                llama_sampler_free(temp);
            }

            if (dist != NULL) {
                llama_sampler_free(dist);
            }

            llama_sampler_free(chat->sampler);
            llama_free(chat->ctx);
            free(chat);
            return NULL;
        }

        llama_sampler_chain_add(chat->sampler, top_k);
        llama_sampler_chain_add(chat->sampler, top_p);
        llama_sampler_chain_add(chat->sampler, temp);
        llama_sampler_chain_add(chat->sampler, dist);
    }

    return chat;
}

void leanllm_chat_free(leanllm_chat *chat) {
    if (chat == NULL) {
        return;
    }

    if (chat->sampler != NULL) {
        llama_sampler_free(chat->sampler);
    }

    if (chat->ctx != NULL) {
        llama_free(chat->ctx);
    }

    if (chat->message_buffer != NULL) {
        free(chat->message_buffer);
    }

    if (chat->prompt_buffer != NULL) {
        free(chat->prompt_buffer);
    }

    if (chat->token_buffer != NULL) {
        free(chat->token_buffer);
    }

    if (chat->piece_buffer != NULL){
        free(chat->piece_buffer);
    }

    free(chat);
}

leanllm_error leanllm_generate(leanllm_chat *chat, const leanllm_message *messages, size_t message_count, leanllm_stream_callback callback, void *userdata) {
    leanllm_error error;
    void *tmp = NULL; // Temporary memory address for realloc calls.
    llama_batch batch;
    bool batch_initialized = false;

    // Step 1: Convert leanllm_message array to a llama_chat_message array.

    // We keep track of the following buffers internally so we don't malloc too often:
    // message_buffer: a buffer for converted llama_chat_message items.
    // prompt_buffer: a buffer for the llama_chat_message with the templates applied.
    // token_buffer: a buffer for tokenized prompt tokens.

    if (chat == NULL ||
        messages == NULL ||
        message_count == 0 ||
        callback == NULL ||
        chat->ctx == NULL ||
        chat->sampler == NULL) {
        error = LEANLLM_ERROR_INVALID_ARGUMENT;
        goto leanllm_generate_error;
    }

    // Reset context memory and sampler.
    llama_memory_t mem = llama_get_memory(chat->ctx);
    llama_memory_clear(mem, false);
    llama_sampler_reset(chat->sampler);

    // Allocate / reallocate the llama_chat_message buffer if needed.

    size_t new_message_capacity = (chat->message_buffer == NULL) ? LEANLLM_INITIAL_MESSAGE_CAPACITY : chat->message_capacity;

    while (message_count > new_message_capacity) {
        new_message_capacity *= 2;
    }

    if (chat->message_buffer == NULL) {
        chat->message_buffer = malloc(new_message_capacity * sizeof(*chat->message_buffer));
        if (chat->message_buffer == NULL) {
            error = LEANLLM_ERROR_OUT_OF_MEMORY;
            goto leanllm_generate_error;
        }
        chat->message_capacity = new_message_capacity;
    }
    else if (new_message_capacity > chat->message_capacity) {
        tmp = realloc(chat->message_buffer, new_message_capacity * sizeof(*chat->message_buffer));
        if (tmp == NULL) {
            error = LEANLLM_ERROR_OUT_OF_MEMORY;
            goto leanllm_generate_error;
        }
        chat->message_buffer = tmp;
        chat->message_capacity = new_message_capacity;
    }

    // Convert and copy leanllm_message messages to llama_chat_message buffer.
    for (size_t i = 0; i < message_count; i++) {

        const char *role = get_role_string(messages[i].role);
        const char *content = messages[i].content;

        if (role == NULL || content == NULL) {
            error = LEANLLM_ERROR_INVALID_ARGUMENT;
            goto leanllm_generate_error;
        }

        chat->message_buffer[i] = (llama_chat_message) {
            .role = role,
            .content = content,
        };
    }

    // Step 2: Apply model template to llama_chat_message array.

    if (chat->prompt_buffer == NULL) {
        chat->prompt_buffer = malloc(LEANLLM_INITIAL_PROMPT_CAPACITY * sizeof(*chat->prompt_buffer));
        if (chat->prompt_buffer == NULL) {
            error = LEANLLM_ERROR_OUT_OF_MEMORY;
            goto leanllm_generate_error;
        }
        chat->prompt_capacity = LEANLLM_INITIAL_PROMPT_CAPACITY;
    }

    // Try to apply the template.

    // The size of the prompt in characters / bytes.
    int32_t template_result = llama_chat_apply_template(chat->template, chat->message_buffer, message_count, true, chat->prompt_buffer, (int32_t)chat->prompt_capacity);

    if (template_result <= 0) {
        error = LEANLLM_ERROR_TEMPLATE;
        goto leanllm_generate_error;
    }

    size_t prompt_size = (size_t)template_result;

    // If our prompt buffer isn't big enough, resize it.
    if (prompt_size > chat->prompt_capacity) {
        size_t new_prompt_capacity = chat->prompt_capacity;
        
        while (prompt_size > new_prompt_capacity) {
            new_prompt_capacity *= 2;
        }

        tmp = realloc(chat->prompt_buffer, new_prompt_capacity * sizeof(*chat->prompt_buffer));
        if (tmp == NULL) {
            error = LEANLLM_ERROR_OUT_OF_MEMORY;
            goto leanllm_generate_error;
        }
        chat->prompt_buffer = tmp;
        chat->prompt_capacity = new_prompt_capacity;

        // Try to apply the template a second time.
        template_result = llama_chat_apply_template(chat->template, chat->message_buffer, message_count, true, chat->prompt_buffer, (int32_t)chat->prompt_capacity);

        if (template_result <= 0) {
            error = LEANLLM_ERROR_TEMPLATE;
            goto leanllm_generate_error;
        }

        prompt_size = (size_t)template_result;
    }

    // We skip the encoding step. lleanllm only supports text-only gguf models.

    // Step 3. Tokenize prompt.

    if (chat->token_buffer == NULL) {
        chat->token_buffer = malloc(LEANLLM_INITIAL_TOKEN_CAPACITY * sizeof(*chat->token_buffer));
        if (chat->token_buffer == NULL) {
            error = LEANLLM_ERROR_OUT_OF_MEMORY;
            goto leanllm_generate_error;
        }
        chat->token_capacity = LEANLLM_INITIAL_TOKEN_CAPACITY;
    }

    // Try to tokenize our prompt.

    int32_t token_result = llama_tokenize(
        chat->model->vocab, 
        chat->prompt_buffer, 
        (int32_t)prompt_size,
        chat->token_buffer,
        (int32_t)chat->token_capacity,
        true,
        true
    );

    size_t prompt_token_count = (token_result < 0) ? (size_t)-token_result : (size_t)token_result;

    // If our token buffer isn't big enough, resize it.
    if (prompt_token_count > chat->token_capacity) {
        size_t new_token_capacity = chat->token_capacity;

        while (prompt_token_count > new_token_capacity) {
            new_token_capacity *= 2;
        }

        tmp = realloc(chat->token_buffer, new_token_capacity * sizeof(*chat->token_buffer));
        if (tmp == NULL) {
            error = LEANLLM_ERROR_OUT_OF_MEMORY;
            goto leanllm_generate_error;
        }
        chat->token_buffer = tmp;
        chat->token_capacity = new_token_capacity;

        // Try to tokenize the prompt a second time.
        token_result = llama_tokenize(
            chat->model->vocab,
            chat->prompt_buffer,
            (int32_t)prompt_size,
            chat->token_buffer,
            (int32_t)chat->token_capacity,
            true,
            true
        );

        if (token_result < 0) {
            error = LEANLLM_ERROR_TOKENIZE;
            goto leanllm_generate_error;
        }

        prompt_token_count = (size_t)token_result;
    }

    if (prompt_token_count >= chat->context_size) {
        error = LEANLLM_ERROR_CONTEXT;
        goto leanllm_generate_error;
    }

    // Process prompt.

    // Number of prompt tokens consumed so far during the batch processing.
    size_t prompt_tokens_consumed = 0;

    // Initialize batch.
    size_t batch_size = (size_t)chat->batch_size;
    batch = llama_batch_init(chat->batch_size, 0, 1);
    batch_initialized = true;

    // Fill batch with the prompt. Might require multiple batch decodes to fit it.
    while (prompt_tokens_consumed < prompt_token_count) {
        size_t i = 0;
        while (i < batch_size && prompt_tokens_consumed + i < prompt_token_count) {
            batch.token[i] = chat->token_buffer[prompt_tokens_consumed + i];
            batch.pos[i] = (llama_pos)(prompt_tokens_consumed + i);

            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;

            batch.logits[i] = (prompt_tokens_consumed + i == prompt_token_count - 1);

            i++;
        }

        batch.n_tokens = (int32_t)i;
        prompt_tokens_consumed += i;

        int32_t decode_result = llama_decode(chat->ctx, batch);

        if (decode_result != 0) {
            error = LEANLLM_ERROR_DECODE;
            goto leanllm_generate_error;
        }
    }

    // Begin text generation.
    
    for (uint32_t i = 0; i < chat->max_tokens; i++) {

        if (prompt_token_count + (size_t)i >= (size_t)chat->context_size) {
            break;
        }

        // We skip decoding on the first iteration because we can
        // carry over the decode from the prompt processing.
        if (i > 0) {
            int32_t decode_result = llama_decode(chat->ctx, batch);
        
            if (decode_result != 0) {
                error = LEANLLM_ERROR_DECODE;
                goto leanllm_generate_error;
            }
        }

        // Sample from the logits produced by the decode
        llama_token new_token_id = llama_sampler_sample(chat->sampler, chat->ctx, -1);

        if (llama_vocab_is_eog(chat->model->vocab, new_token_id)) {
            break;
        }

        if (chat->piece_buffer == NULL)
        {
            chat->piece_buffer = malloc(LEANLLM_INITIAL_TOKEN_PIECE_CAPACITY * sizeof(*chat->piece_buffer));
            if (chat->piece_buffer == NULL) {
                error = LEANLLM_ERROR_OUT_OF_MEMORY;
                goto leanllm_generate_error;
            }
            chat->piece_capacity = LEANLLM_INITIAL_TOKEN_PIECE_CAPACITY * sizeof(*chat->piece_buffer);
        }

        int32_t n = llama_token_to_piece(chat->model->vocab, new_token_id, chat->piece_buffer, (int32_t)chat->piece_capacity, 0, true);

        if (n < 0) {
            size_t required_piece_buffer_size = -n;
            size_t new_piece_capacity = chat->piece_capacity;

            while (required_piece_buffer_size > new_piece_capacity) {
                new_piece_capacity *= 2;
            }
            
            tmp = realloc(chat->piece_buffer, new_piece_capacity * sizeof(*chat->piece_buffer));
            if (tmp == NULL) {
                error = LEANLLM_ERROR_OUT_OF_MEMORY;
                goto leanllm_generate_error;
            }

            chat->piece_buffer = tmp;
            chat->piece_capacity = new_piece_capacity;

            n = llama_token_to_piece(chat->model->vocab, new_token_id, chat->piece_buffer, (int32_t)chat->piece_capacity, 0, true);
            if (n < 0) {
                error = LEANLLM_ERROR_TOKEN_TO_PIECE;
                goto leanllm_generate_error;
            }
        }

        if (!callback(chat->piece_buffer, (size_t)n, userdata)) {
            break;
        }

        // Next decode only needs the newly generated token.

        batch.n_tokens = 1;
        batch.token[0] = new_token_id;
        batch.pos[0] = (llama_pos)(prompt_token_count + i);
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;
    }

    llama_batch_free(batch);

    return LEANLLM_OK;

leanllm_generate_error:
    if (batch_initialized) {
        llama_batch_free(batch);
    }
    return error;
}

const char *leanllm_error_string(leanllm_error error) {
    switch (error) {
    case LEANLLM_OK:
        return "success";
    case LEANLLM_ERROR_INVALID_ARGUMENT:
        return "invalid argument";
    case LEANLLM_ERROR_OUT_OF_MEMORY:
        return "out of memory";
    case LEANLLM_ERROR_MODEL_LOAD:
        return "failed to load model";
    case LEANLLM_ERROR_CONTEXT:
        return "context error";
    case LEANLLM_ERROR_TEMPLATE:
        return "chat template error";
    case LEANLLM_ERROR_TOKENIZE:
        return "tokenization error";
    case LEANLLM_ERROR_DECODE:
        return "decode error";
    case LEANLLM_ERROR_TOKEN_TO_PIECE:
        return "token-to-piece conversion error";
    default:
        return "unknown error";
    }
}