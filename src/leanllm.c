#include "leanllm.h"

#include <stdlib.h>

#include <llama.h>

#define LEANLLM_INITIAL_PROMPT_CAPACITY 1024

struct leanllm_model {
    struct llama_model *model;
    const struct llama_vocab *vocab;
};

struct leanllm_chat {
    leanllm_model *model;
    const char *template;

    struct llama_context *ctx;
    struct llama_sampler *sampler;

    uint32_t batch_size;
    llama_batch batch;
    bool batch_initialized;

    int32_t max_tokens;

    llama_chat_message *message_buffer;
    size_t message_capacity;

    char *prompt_buffer;
    size_t prompt_capacity;

    llama_token *token_buffer;
    size_t token_capacity;
};


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
    chat->ctx = NULL;
    chat->sampler = NULL;
    chat->batch_size = opts.batch_size;
    chat->batch_initialized = false;
    chat->max_tokens = opts.max_tokens;
    chat->message_buffer = NULL;
    chat->message_capacity = 0;
    chat->prompt_buffer = NULL;
    chat->prompt_capacity = 0;
    chat->token_buffer = NULL;
    chat->token_capacity = 0;

    struct llama_context_params ctx_params = llama_context_default_params();

    ctx_params.n_ctx = opts.context_size;
    ctx_params.n_batch = opts.batch_size;

    chat->ctx = llama_init_from_model(model->model, ctx_params);

    if (chat->ctx == NULL) {
        free(chat);
        return NULL;
    }

    const char *tmpl = llama_model_chat_template(model, NULL);

    if (tmpl == NULL)
    {
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

    if (chat->batch_initialized) {
        llama_batch_free(chat->batch);
    }

    if (chat->sampler != NULL) {
        llama_sampler_free(chat->sampler);
    }

    if (chat->ctx != NULL) {
        llama_free(chat->ctx);
    }

    if (chat->prompt_buffer != NULL) {
        free(chat->prompt_buffer);
    }

    if (chat->token_buffer != NULL) {
        free(chat->token_buffer);
    }

    free(chat);
}

int leanllm_generate(leanllm_chat *chat, const leanllm_message *messages, size_t message_count, leanllm_stream_callback callback, void *userdata) {
    if (!chat->batch_initialized) {
        // Prepare the first batch.
        chat->batch = llama_batch_init(chat->batch_size, 0, 1);
        chat->batch_initialized = true;
    }

    // Step 1: Convert leanllm_message array to a llama_chat_message array.

    // We keep track of the following buffers internally so we don't malloc too often:
    // message_buffer: a buffer for converted llama_chat_message items.
    // prompt_buffer: a buffer for the llama_chat_message with the templates applied.
    // token_buffer: idk what this is for actually yet. chatgpt suggested it lol.

    // Step 2: Apply template to llama_chat_message array.

    // We might need to call this possibly twice, once to compute the size, second to push to the buffer.
    // Though, if it honours the passed capacity size then shouldn't it be safe to pass the buffer both times and avoid a second call?
    // size_t prompt_size = llama_chat_apply_template(chat->template, chat->message_buffer, message_count, false, chat->prompt_buffer, chat->prompt_capacity);

    // Step 3: Fill batch with the prompt.
    // ...

    // Step 4: Decode?

    // Step 5: ...

    // Step 6: Profit?
}