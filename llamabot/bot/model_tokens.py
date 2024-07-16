"""Constants for model tokens and token budgets."""

DEFAULT_TOKEN_BUDGET = 4_096

# Reference: https://platform.openai.com/docs/models
# For the Ollama models, refer to individual model documentation page: https://ollama.ai/library
# Please update the numbers here as necessary.
model_context_window_sizes = {
    "gpt-3.5-turbo-0301": 4_097,
    "gpt-3.5-turbo-0613": 4_097,
    "gpt-3.5-turbo-16k-0613": 16_385,
    "gpt-3.5-turbo-16k": 16_385,
    "gpt-3.5-turbo-instruct-0914": 4_097,
    "gpt-3.5-turbo-instruct": 4_097,
    "gpt-3.5-turbo": 4_097,
    "gpt-4-0314": 8_192,
    "gpt-4-0613": 8_192,
    "gpt-4-32k": 32_768,
    "gpt-4-32k-0613": 32_768,
    "gpt-4": 8_192,
    "llama2": 4_096,
    "vicuna:7b-16k": 16_385,
    "mistral/mistral-tiny": 4_096,
    "mistral/mistral-small": 8_192,
    "mistral/mistral-medium": 32_768,
}
