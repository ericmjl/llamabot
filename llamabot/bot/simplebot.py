"""Class definition for SimpleBot."""
import contextvars

import panel as pn
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from loguru import logger

from llamabot.panel_utils import PanelMarkdownCallbackHandler
from llamabot.recorder import autorecord
from llamabot.bot.model_dispatcher import create_model
from llamabot.config import default_language_model

prompt_recorder_var = contextvars.ContextVar("prompt_recorder")


class SimpleBot:
    """Simple Bot that is primed with a system prompt, accepts a human message,
    and sends back a single response.

    This bot does not retain chat history.

    :param system_prompt: The system prompt to use.
    :param temperature: The model temperature to use.
        See https://platform.openai.com/docs/api-reference/completions/create#completions/create-temperature
        for more information.
    :param model_name: The name of the OpenAI model to use.
    :param streaming: (LangChain config) Whether to stream the output to stdout.
    :param verbose: (LangChain config) Whether to print debug messages.
    """

    def __init__(
        self,
        system_prompt,
        temperature=0.0,
        model_name=default_language_model(),
        streaming=True,
        verbose=True,
    ):
        self.system_prompt = system_prompt
        self.model = create_model(
            model_name=model_name,
            temperature=temperature,
            streaming=streaming,
            verbose=verbose,
        )
        self.chat_history = []
        self.model_name = model_name

    def __call__(self, human_message: str) -> AIMessage:
        """Call the SimpleBot.

        :param human_message: The human message to use.
        :return: The response to the human message, primed by the system prompt.
        """

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=human_message),
        ]
        response = self.model(messages)
        self.chat_history.append(HumanMessage(content=human_message))
        self.chat_history.append(response)
        autorecord(human_message, response.content)
        return response

    def panel(
        self,
        input_text_label="Input",
        output_text_label="Output",
        submit_button_label="Submit",
        site_name="SimpleBot",
        title="SimpleBot",
        show=False,
    ):
        """Create a Panel app that wraps a LlamaBot.

        :param input_text_label: The label for the input text.
        :param output_text_label: The label for the output text.
        :param submit_button_label: The label for the submit button.
        :param site_name: The name of the site.
        :param title: The title of the site.
        :param show: Whether to show the app.
            If False, we return the Panel app directly.
            If True, we call `.show()` on the app.
        :return: The Panel app, either showed or directly.
        """
        input_text = pn.widgets.TextAreaInput(
            name=input_text_label, value="", height=200, width=500
        )
        output_text = pn.pane.Markdown("")
        submit = pn.widgets.Button(name=submit_button_label, button_type="success")

        def b(event):
            """Button click handler.

            :param event: The button click event.
            """
            logger.info(input_text.value)
            output_text.object = ""
            markdown_handler = PanelMarkdownCallbackHandler(output_text)
            self.model.callback_manager.set_handler(markdown_handler)
            response = self(input_text.value)
            logger.info(response)

        submit.on_click(b)

        app = pn.template.FastListTemplate(
            site=site_name,
            title=title,
            main=[
                pn.Column(
                    *[
                        input_text,
                        submit,
                        pn.pane.Markdown(output_text_label),
                        output_text,
                    ]
                )
            ],
            main_max_width="768px",
        )
        app = pn.panel(app)
        if show:
            return app.show()
        return app
