#include "leanllm.h"

#include <stdlib.h>

#include <llama.h>


struct leanllm_model {
    struct llama_model *model;
    const struct llama_vocab *vocab;
};

struct leanllm_chat {
    leanllm_model *model;

    struct llama_context *ctx;
    struct llama_sampler *sampler;

    llama_batch batch;
    bool batch_initialized;
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

leanllm_model *leanllm_model_load(const char *path, const leanllm_model_options *options) {
    if (path == NULL) {
        return NULL;
    }

    leanllm_model_options opts = (options != NULL) ? *options : leanllm_model_default_options();

    struct llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = (int32_t)opts.gpu_layers;

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