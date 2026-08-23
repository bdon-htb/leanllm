# LeanLLM

A tiny C wrapper around llama.cpp for quick and simple local LLM inference.

LeanLLM provides a small API for loading GGUF models and generating text without needing to work directly with the larger llama.cpp API.

LeanLLM currently focuses on text-only GGUF models and streaming text generation. It does not support multimodal inference.

## Building

### Requirements

- CMake 3.16+
- A C11 compiler
- A C++17 compiler
- Git

Clone the repository:

```bash
git clone <repository-url>
cd leanllm
```

Configure the project:

```bash
cmake -S . -B build
```

Build:

```bash
cmake --build build
```

CMake will automatically download and build the supported version of llama.cpp.

### Windows with MinGW

If CMake does not automatically select MinGW:

```bash
cmake -S . -B build -G "MinGW Makefiles"
cmake --build build
```

## Example

A basic example is available in `examples/simple.c`.

```c
#include <leanllm.h>

#include <stdio.h>

static bool print_piece(const char *text, size_t length, void *userdata)
{
    (void)userdata;

    fwrite(text, 1, length, stdout);
    fflush(stdout);

    return true;
}

int main(void)
{
    leanllm_init();

    leanllm_model *model = leanllm_model_load("model.gguf", NULL);
    if (model == NULL) {
        leanllm_shutdown();
        return 1;
    }

    leanllm_chat *chat = leanllm_chat_create(model, NULL);
    if (chat == NULL) {
        leanllm_model_free(model);
        leanllm_shutdown();
        return 1;
    }

    leanllm_message messages[] = {
        {
            .role = LEANLLM_ROLE_USER,
            .content = "Hello!"
        }
    };

    leanllm_error error = leanllm_generate(
        chat,
        messages,
        1,
        print_piece,
        NULL
    );

    if (error != LEANLLM_OK) {
        fprintf(stderr, "%s\n", leanllm_error_string(error));
    }

    leanllm_chat_free(chat);
    leanllm_model_free(model);
    leanllm_shutdown();

    return error == LEANLLM_OK ? 0 : 1;
}
```

## Status

LeanLLM is currently an early work in progress. The API may change.