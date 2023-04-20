"""Class definition for ChatBot."""

import panel as pn
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from llama_index.response.schema import Response

from llamabot.panel_utils import PanelMarkdownCallbackHandler


class ChatBot:
    """Chat Bot that is primed with a system prompt, accepts a human message.

    Automatic chat memory management happens.

    h/t Andrew Giessel/GPT4 for the idea.
    """

    def __init__(self, system_prompt, temperature=0.0, model_name="gpt-4"):
        """Initialize the ChatBot.

        :param system_prompt: The system prompt to use.
        :param temperature: The model temperature to use.
            See https://platform.openai.com/docs/api-reference/completions/create#completions/create-temperature
            for more information.
        :param model_name: The name of the OpenAI model to use.
        """
        self.model = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            streaming=True,
            verbose=True,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        )
        self.chat_history = [
            SystemMessage(content="Always return Markdown-compatible text."),
            SystemMessage(content=system_prompt),
        ]

    def __call__(self, human_message, test_mode: bool = False) -> Response:
        """Call the ChatBot.

        :param human_message: The human message to use.
        :param test_mode: Whether to use testing mode,
            in which case no API calls are made,
            and we just with respond a "test" message back.
        :return: The response to the human message, primed by the system prompt.
        """
        self.chat_history.append(HumanMessage(content=human_message))
        if test_mode:
            response = self.model(self.chat_history)
        else:
            response = AIMessage(content="test")
        self.chat_history.append(response)
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

    def panel(self, show: bool = True):
        """Create a Panel app that wraps a LlamaBot.

        :param show: Whether to show the app.
            If False, we return the Panel app directly.
            If True, we call `.show()` on the app.
        :return: The Panel app, either showed or directly.
        """

        text_input = pn.widgets.TextAreaInput(placeholder="Start chatting...")
        chat_history = pn.Column(*[])
        send_button = pn.widgets.Button(name="Send", button_type="primary")

        def b(event):
            """Button click handler.

            :param event: The button click event.
            """
            chat_messages = []
            for message in self.chat_history:
                if isinstance(message, SystemMessage):
                    pass
                elif isinstance(message, HumanMessage):
                    chat_markdown = pn.pane.Markdown(f"Human: {message.content}")
                    chat_messages.append(chat_markdown)
                elif isinstance(message, AIMessage):
                    chat_markdown = pn.pane.Markdown(f"Bot: {message.content}")
                    chat_messages.append(chat_markdown)

            chat_messages.append(pn.pane.Markdown(f"Human: {text_input.value}"))
            bot_reply = pn.pane.Markdown("Bot: ")
            chat_messages.append(bot_reply)
            chat_history.objects = chat_messages
            markdown_handler = PanelMarkdownCallbackHandler(bot_reply)
            self.model.callback_manager.set_handler(markdown_handler)
            self(text_input.value)
            text_input.value = ""

        send_button.on_click(b)
        input_pane = pn.Row(text_input, send_button)
        output_pane = pn.Column(chat_history, scroll=True, height=500)

        main = pn.Row(input_pane, output_pane)
        app = pn.template.FastListTemplate(
            site="ChatBot",
            title="ChatBot",
            main=main,
            main_max_width="768px",
        )
        if show:
            return app.show()
        return app
