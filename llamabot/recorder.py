"""Prompt recorder class definition."""
import contextvars
from typing import Optional

import pandas as pd

prompt_recorder_var = contextvars.ContextVar("prompt_recorder")


class PromptRecorder:
    """Prompt recorder to support recording of prompts and responses."""

    def __init__(self):
        """Initialize prompt recorder."""
        self.prompts_and_responses = []

    def __enter__(self):
        """Enter the context manager."""
        print("Recording prompt and response...")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager.

        :param exc_type: The exception type.
        :param exc_val: The exception value.
        :param exc_tb: The exception traceback.
        """
        print("Recording complete!🎉")

    def log(self, prompt: str, response: str):
        """Log the prompt and response in chat history.

        :param prompt: The human prompt.
        :param response: A the response from the bot.
        """
        self.prompts_and_responses.append({"prompt": prompt, "response": response})

    def __repr__(self):
        """Return a string representation of the prompt recorder.

        :return: A string form of the prompts and responses as a dataframe.
        """
        return pd.DataFrame(self.prompts_and_responses).__str__()

    def _repr_html_(self):
        """Return an HTML representation of the prompt recorder.

        :return: We delegate to the _repr_html_ method of the pandas DataFrame class.
        """
        return pd.DataFrame(self.prompts_and_responses)._repr_html_()


def autorecord(prompt, response):
    """Record a prompt and response.

    :param prompt: The human prompt.
    :param response: A the response from the bot.
    """
    # Log the response.
    prompt_recorder: Optional[PromptRecorder] = prompt_recorder_var.get(None)
    if prompt_recorder:
        prompt_recorder.log(prompt, response.content)
