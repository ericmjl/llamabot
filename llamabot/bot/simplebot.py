"""Class definition for SimpleBot."""
import contextvars
from typing import Optional


# from langchain.schema import AIMessage, HumanMessage, SystemMessage, BaseMessage

from llamabot.components.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)
from llamabot.recorder import autorecord
from llamabot.config import default_language_model
from litellm import completion
from loguru import logger

prompt_recorder_var = contextvars.ContextVar("prompt_recorder")


class SimpleBot:
    """Simple Bot that is primed with a system prompt, accepts a human message,
    and sends back a single response.

    This bot does not retain chat history.

    :param system_prompt: The system prompt to use.
    :param temperature: The model temperature to use.
        See https://platform.openai.com/docs/api-reference/completions/create#completions/create-temperature
        for more information.
    :param model_name: The name of the model to use.
    :param stream: Whether to stream the output to stdout.
    :param json_mode: Whether to print debug messages.
    """

    def __init__(
        self,
        system_prompt: str,
        temperature=0.0,
        model_name=default_language_model(),
        stream=True,
        json_mode: bool = False,
        api_key: Optional[str] = None,
    ):
        self.system_prompt: SystemMessage = SystemMessage(content=system_prompt)
        self.model_name = model_name
        self.temperature = temperature
        self.stream = stream
        self.json_mode = json_mode
        self.api_key = api_key

    def __call__(self, human_message: str) -> AIMessage:
        """Call the SimpleBot.

        :param human_message: The human message to use.
        :return: The response to the human message, primed by the system prompt.
        """

        messages: list[BaseMessage] = [
            self.system_prompt,
            HumanMessage(content=human_message),
        ]
        response = self.generate_response(messages)
        autorecord(human_message, response.content)
        return response

    def generate_response(self, messages: list[BaseMessage]) -> AIMessage:
        """Generate a response from the given messages."""

        messages_dumped: list[dict] = [m.model_dump() for m in messages]
        completion_kwargs = dict(
            model=self.model_name,
            messages=messages_dumped,
            temperature=self.temperature,
            stream=self.stream,
        )
        if self.json_mode:
            completion_kwargs["response_format"] = {"type": "json_object"}
        if self.api_key:
            completion_kwargs["api_key"] = self.api_key
        logger.info(f"Using SimpleBot API key: {self.api_key}...")
        response = completion(**completion_kwargs)

        if self.stream:
            ai_message = ""
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta is not None:
                    print(delta, end="")
                    ai_message += delta
            return AIMessage(content=ai_message)

        return AIMessage(content=response.choices[0].message.content)

    # Commented out until later.
    # def panel(
    #     self,
    #     input_text_label="Input",
    #     output_text_label="Output",
    #     submit_button_label="Submit",
    #     site_name="SimpleBot",
    #     title="SimpleBot",
    #     show=False,
    # ):
    #     """Create a Panel app that wraps a LlamaBot.

    #     :param input_text_label: The label for the input text.
    #     :param output_text_label: The label for the output text.
    #     :param submit_button_label: The label for the submit button.
    #     :param site_name: The name of the site.
    #     :param title: The title of the site.
    #     :param show: Whether to show the app.
    #         If False, we return the Panel app directly.
    #         If True, we call `.show()` on the app.
    #     :return: The Panel app, either showed or directly.
    #     """
    #     input_text = pn.widgets.TextAreaInput(
    #         name=input_text_label, value="", height=200, width=500
    #     )
    #     output_text = pn.pane.Markdown("")
    #     submit = pn.widgets.Button(name=submit_button_label, button_type="success")

    #     def b(event):
    #         """Button click handler.

    #         :param event: The button click event.
    #         """
    #         logger.info(input_text.value)
    #         output_text.object = ""
    #         markdown_handler = PanelMarkdownCallbackHandler(output_text)
    #         self.model.callback_manager.set_handler(markdown_handler)
    #         response = self(input_text.value)
    #         logger.info(response)

    #     submit.on_click(b)

    #     app = pn.template.FastListTemplate(
    #         site=site_name,
    #         title=title,
    #         main=[
    #             pn.Column(
    #                 *[
    #                     input_text,
    #                     submit,
    #                     pn.pane.Markdown(output_text_label),
    #                     output_text,
    #                 ]
    #             )
    #         ],
    #         main_max_width="768px",
    #     )
    #     app = pn.panel(app)
    #     if show:
    #         return app.show()
    #     return app
