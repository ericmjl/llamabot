"""Class definition for ChatBot."""

import contextvars

import panel as pn
import tiktoken
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from loguru import logger

from llamabot.config import default_language_model
from llamabot.panel_utils import PanelMarkdownCallbackHandler
from llamabot.recorder import autorecord

prompt_recorder_var = contextvars.ContextVar("prompt_recorder")

model_chat_token_budgets = {
    "gpt-4-32k": 32_000,
    "gpt-4": 8_000,
}


class ChatBot:
    """Chat Bot that is primed with a system prompt, accepts a human message.

    Automatic chat memory management happens.

    h/t Andrew Giessel/GPT4 for the idea.
    """

    def __init__(
        self,
        system_prompt,
        temperature=0.0,
        model_name=default_language_model(),
        logging=False,
        response_budget=2_000,
    ):
        """Initialize the ChatBot.

        :param system_prompt: The system prompt to use.
        :param temperature: The model temperature to use.
            See https://platform.openai.com/docs/api-reference/completions/create#completions/create-temperature
            for more information.
        :param model_name: The name of the OpenAI model to use.
        :param logging: Whether to log the chat history.
        :param response_budget: The number of tokens to budget for the response.
        """
        self.model = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            streaming=True,
            verbose=True,
            callback_manager=BaseCallbackManager(
                handlers=[StreamingStdOutCallbackHandler()]
            ),
        )
        self.chat_history = [
            SystemMessage(content="Always return Markdown-compatible text."),
            SystemMessage(content=system_prompt),
        ]
        self.logging = logging
        self.model_name = model_name
        self.response_budget = response_budget

    def __call__(self, human_message) -> AIMessage:
        """Call the ChatBot.

        :param human_message: The human message to use.
        :return: The response to the human message, primed by the system prompt.
        """
        self.chat_history.append(HumanMessage(content=human_message))

        # Get out the last 6000 tokens of chat history.
        faux_chat_history = []
        enc = tiktoken.encoding_for_model("gpt-4")
        token_budget = model_chat_token_budgets[self.model_name] - self.response_budget

        # Put the system message into the faux chat history.
        system_messages = []
        for message in self.chat_history:
            if isinstance(message, SystemMessage):
                system_messages.append(message)
                tokens = enc.encode(message.content)
                token_budget -= len(tokens)

        faux_chat_history = []
        for message in self.chat_history[::-1]:
            if token_budget > 0:
                if self.logging:
                    logger.info(f"Token budget: {token_budget}")
                faux_chat_history.append(message)
                tokens = enc.encode(message.content)
                token_budget -= len(tokens)

        messages = system_messages + faux_chat_history[::-1]
        if self.logging:
            logger.info(messages)

        response = self.model(messages)
        self.chat_history.append(response)
        autorecord(human_message, response.content)

        return response

    def __repr__(self):
        """Return a string representation of the ChatBot.

        :return: A string representation of the ChatBot.
        """
        representation = ""

        for message in self.chat_history:
            if isinstance(message, SystemMessage):
                prefix = "[System]\n"
            elif isinstance(message, HumanMessage):
                prefix = "[Human]\n"
            elif isinstance(message, AIMessage):
                prefix = "[AI]\n"

            representation += f"{prefix}{message.content}" + "\n\n"
        return representation

    def panel(
        self,
        show: bool = True,
        site="ChatBot",
        title="ChatBot",
        width=768,
    ):
        """Create a Panel app that wraps a LlamaBot.

        :param show: Whether to show the app.
            If False, we return the Panel app directly.
            If True, we call `.show()` on the app.
        :param site: The name of the site.
        :param title: The title of the app.
        :param width: The width of the app in pixels.
        :return: The Panel app, either showed or directly.
        """

        text_input = pn.widgets.TextAreaInput(placeholder="Start chatting...")
        chat_history = pn.Column(*[])
        send_button = pn.widgets.Button(name="Send", button_type="primary")

        widget_width = width - 150

        def b(event):
            """Button click handler.

            :param event: The button click event.
            """
            chat_messages = []
            for message in self.chat_history:
                if isinstance(message, SystemMessage):
                    pass
                elif isinstance(message, HumanMessage):
                    chat_markdown = pn.pane.Markdown(
                        f"Human: {message.content}", width=widget_width
                    )
                    chat_messages.append(chat_markdown)
                elif isinstance(message, AIMessage):
                    chat_markdown = pn.pane.Markdown(
                        f"Bot: {message.content}", width=widget_width
                    )
                    chat_messages.append(chat_markdown)

            chat_messages.append(
                pn.pane.Markdown(f"Human: {text_input.value}", width=widget_width)
            )
            bot_reply = pn.pane.Markdown("Bot: ", width=widget_width)
            chat_messages.append(bot_reply)
            chat_history.objects = chat_messages
            markdown_handler = PanelMarkdownCallbackHandler(bot_reply)
            self.model.callback_manager.set_handler(markdown_handler)
            self(text_input.value)
            text_input.value = ""

        send_button.on_click(b)
        input_pane = pn.Row(text_input, send_button)
        output_pane = pn.Column(chat_history, scroll=False, height=500)

        main = pn.Row(input_pane, output_pane)
        app = pn.template.FastListTemplate(
            site=site,
            title=title,
            main=main,
            main_max_width=f"{width}px",
        )
        if show:
            return app.show()
        return app
