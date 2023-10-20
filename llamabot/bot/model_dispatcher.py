"""Dispatch to the appropriate model based on string name."""

from langchain.chat_models import ChatOpenAI, ChatOllama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackManager

# get this list from: https://ollama.ai/library
ollama_model_keywords = [
    "mistral",
    "llama2",
    "codellama",
    "vicuna",
    "orca-mini",
    "llama2-uncensored",
    "wizard-vicuna-uncensored",
    "nous-hermes",
    "phind-codellama",
    "wizardcoder",
    "mistral-openorca",
    "wizard-math",
    "stable-beluga",
    "llama2-chinese",
    "codeup",
    "everythinglm",
    "wizardlm-uncesored",
    "medllama2",
    "wizard-vicuna",
    "falcon",
    "open-orca-platypus2",
    "zephyr",
    "wizardlm",
    "starcoder",
    "samantha-mistral",
    "sqlcoder",
    "openhermes2-mistral",
    "nexusraven",
]


def create_model(
    model_name,
    temperature=0.0,
    streaming=True,
    verbose=True,
):
    """Dispatch and create the right model.

    This is necessary to validate b/c LangChain doesn't do the validation for us.

    :param model_name: The name of the model to use.
    :param temperature: The model temperature to use.
    :param streaming: (LangChain config) Whether to stream the output to stdout.
    :param verbose: (LangChain config) Whether to print debug messages.
    :return: The model.
    """
    ModelClass = ChatOpenAI
    if model_name.split(":")[0] in ollama_model_keywords:
        ModelClass = ChatOllama

    return ModelClass(
        model_name=model_name,
        temperature=temperature,
        streaming=streaming,
        verbose=verbose,
        callback_manager=BaseCallbackManager(
            handlers=[StreamingStdOutCallbackHandler()] if streaming else []
        ),
    )
