"""Prompt recorder class definition."""
from typing import Union

import pandas as pd

from llamabot import ChatBot, QueryBot, SimpleBot


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

    def log(self, bot: Union[SimpleBot, ChatBot, QueryBot], prompt: str):
        """Log the prompt and response in chat history.

        :param bot: An instance of a llamabot bot.
        :param prompt: A prompt to record along with the bot's response.
        """
        response = bot(prompt)

        self.prompts_and_responses.append(
            {"prompt": prompt, "response": response.content}
        )

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
