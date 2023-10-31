"""Dispatch to the appropriate model based on string name."""

import subprocess
import signal
import sys
from functools import wraps
import os


from langchain.chat_models import ChatOpenAI, ChatOllama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackManager
from time import sleep
from loguru import logger
from functools import partial
from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=128)
def ollama_model_keywords() -> list:
    """Return ollama model keywords.

    This is stored within a the `ollama_model_names.txt` file
    that is distributed with this package.

    :returns: The list of model names.
    """
    with open(Path(__file__).parent / "ollama_model_names.txt") as f:
        return [line.strip() for line in f.readlines()]


def create_model(
    model_name,
    temperature=0.0,
    streaming=True,
    verbose=True,
):
    """Dispatch and create the right model.

    This is necessary to validate b/c LangChain doesn't do the validation for us.

    Example usage:

    ```python
    # use the vicuna model
    model = create_model(model_name="vicuna")

    # use the llama2 model
    model = create_model("llama2")

    # use codellama with a temperature of 0.5
    model = create_model("codellama:13b", temperature=0.5)
    ```

    :param model_name: The name of the model to use.
    :param temperature: The model temperature to use.
    :param streaming: (LangChain config) Whether to stream the output to stdout.
    :param verbose: (LangChain config) Whether to print debug messages.
    :return: The model.
    """
    # We use a `partial` here to ensure that we have the correct way of specifying
    # a model name between ChatOpenAI and ChatOllama.
    ModelClass = partial(ChatOpenAI, model_name=model_name)
    if model_name.split(":")[0] in ollama_model_keywords():
        ModelClass = partial(ChatOllama, model=model_name)

    return ModelClass(
        temperature=temperature,
        streaming=streaming,
        verbose=verbose,
        callback_manager=BaseCallbackManager(
            handlers=[StreamingStdOutCallbackHandler()] if streaming else []
        ),
    )


def kill_process_if_python_dies(command_template):
    """Decorator to kill a subprocess if the Python process dies.

    NOTE: This function is challenging to test.
    I am not sure how to test it at the moment.

    This decorator is used within llamabot
    to launch the Ollama API locally in an automagical way.
    However, it definitely is not specific to llamabot
    and can probably be used in other settings.

    :param command_template: The command template to use.
    """
    child_process = None

    def signal_handler(signal_number, frame):
        """Signal handler for SIGINT and SIGTERM.

        This function terminates the child process and exits the program
        if the child process is running.

        :param signal_number: The signal number.
            In plain English, this is the signal
            that was received from the operating system,
            which is usually a string like 'SIGINT' or 'SIGTERM'.
        :param frame: The frame.
            In plain English, this is the execution frame
            that was interrupted by the signal,
            which is usually the main thread.
        """
        nonlocal child_process
        if child_process:
            logger.info("Terminating child process")
            child_process.terminate()
        sys.exit(0)

    def decorator(func):
        """Decorator to manage the child process.

        :param func: The function to decorate.
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            """Wrapper function to manage the child process.

            :param args: The positional arguments.
            :param kwargs: The keyword arguments.
            """
            nonlocal child_process
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            model_name = args[0] if args else kwargs.get("model_name", "")
            command = [part.format(model_name=model_name) for part in command_template]

            verbose = kwargs.get("verbose", True)
            if verbose:
                child_process = subprocess.Popen(command)
            else:
                with open(os.devnull, "w") as nullfile:
                    child_process = subprocess.Popen(
                        command, stdout=nullfile, stderr=nullfile
                    )
            logger.info(f"Launched child process with PID {child_process.pid}")

            return func(*args, **kwargs)

        return wrapper

    return decorator


# Use a template for the command, with {model_name} as a placeholder for the model name
@kill_process_if_python_dies(["ollama", "serve", "{model_name}"])
def launch_ollama(model_name, verbose=True):
    """Launch the Ollama API.

    :param model_name: The name of the model to use.
    :param verbose: Whether to print debug messages.
    """
    logger.info("Launching Ollama API...")
    sleep(2)
