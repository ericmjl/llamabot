"""Prompt recorder class definition."""
import contextvars
from pathlib import Path
from typing import Optional

import pandas as pd
import panel as pn

prompt_recorder_var = contextvars.ContextVar("prompt_recorder")


class PromptRecorder:
    """Prompt recorder to support recording of prompts and responses."""

    def __init__(self):
        """Initialize prompt recorder."""
        self.prompts_and_responses = []

    def __enter__(self):
        """Enter the context manager.

        :returns: The prompt recorder.
        """
        prompt_recorder_var.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager.

        :param exc_type: The exception type.
        :param exc_val: The exception value.
        :param exc_tb: The exception traceback.
        """
        prompt_recorder_var.set(None)

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
        return self.dataframe()._repr_html_()

    def dataframe(self):
        """Return a pandas DataFrame representation of the prompt recorder.

        :return: A pandas DataFrame representation of the prompt recorder.
        """
        return pd.DataFrame(self.prompts_and_responses)

    def save(self, path: Path):
        """Save the prompt recorder to a path.

        :param path: The path to save the prompt recorder to.
        """
        path = Path(path)  # coerce to pathlib.Path
        with path.open("w+") as f:
            for prompt_and_response in self.prompts_and_responses:
                f.write(
                    f"**{prompt_and_response['prompt']}**\n\n{prompt_and_response['response']}\n\n"
                )

    def panel(self):
        """Return a panel representation of the prompt recorder.

        :return: A panel representation of the prompt recorder.
        """
        global index
        index = 0
        pn.extension()

        next_button = pn.widgets.Button(name=">")
        next_button
        prev_button = pn.widgets.Button(name="<")
        prev_button

        buttons = pn.Row(prev_button, next_button)

        prompt_header = pn.pane.Markdown("# Prompt")
        prompt_display = pn.pane.Markdown(self.prompts_and_responses[index]["prompt"])

        prompt = pn.Column(prompt_header, prompt_display)

        response_header = pn.pane.Markdown("# Response")
        response_display = pn.pane.Markdown(
            self.prompts_and_responses[index]["response"]
        )
        response = pn.Column(response_header, response_display)

        display = pn.Row(prompt, response)

        def update_objects(index):
            """Update the prompt and response Markdown panes.

            :param index: The index of the prompt and response to update.
            """
            prompt_display.object = self.prompts_and_responses[index]["prompt"]
            response_display.object = self.prompts_and_responses[index]["response"]

        def next_button_callback(event):
            """Callback function for the next button.

            :param event: The click event.
            """
            global index
            index += 1
            if index > len(self.prompts_and_responses) - 1:
                index = len(self.prompts_and_responses) - 1

            update_objects(index)

        def prev_button_callback(event):
            """Callback function for the previous button.

            :param event: The click event.
            """
            global index
            index -= 1
            if index < 0:
                index = 0
            update_objects(index)

        next_button.on_click(next_button_callback)
        prev_button.on_click(prev_button_callback)

        return pn.Column(buttons, display)


def autorecord(prompt: str, response: str):
    """Record a prompt and response.

    This is intended to be called within every bot.
    If we are within a prompt recorder context,
    then the prompt recorder will record the prompt and response
    as specified in the function.

    :param prompt: The human prompt.
    :param response: A the response from the bot.
    """
    # Log the response.
    prompt_recorder: Optional[PromptRecorder] = prompt_recorder_var.get(None)
    if prompt_recorder:
        prompt_recorder.log(prompt, response)
