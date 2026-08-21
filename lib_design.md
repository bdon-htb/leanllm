## LeanLLM Design Doc

- Two major container structs, leanllm_model, and leanllm_chat.
- leanllm_model tracks model parameters like model weights and vocabulary.
- leanllm_chat tracks a given inference context. Things like KV cache, batch state and conversation histoy.
- Should be self contained, not requiring any external dependencies other than llamacpp.
- Library functions are synchronous operations.
