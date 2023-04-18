"""Class definition for ChatBot."""
from typing import Any, Dict, List, Union

import panel as pn
from langchain.callbacks.base import BaseCallbackHandler, CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AgentAction,
    AgentFinish,
    AIMessage,
    HumanMessage,
    LLMResult,
    SystemMessage,
)
from llama_index.response.schema import Response


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

    def __call__(self, human_message) -> Response:
        """Call the ChatBot.

        :param human_message: The human message to use.
        :return: The response to the human message, primed by the system prompt.
        """
        self.chat_history.append(HumanMessage(content=human_message))
        response = self.model(self.chat_history)
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


class PanelMarkdownCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming. Only works with LLMs that support streaming."""

    def __init__(self, markdown_object: pn.Pane):
        self.md = markdown_object

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running.

        # noqa: DAR101
        """

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled.

        # noqa: DAR101
        """
        self.md.object += f"{token}"

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running.

        # noqa: DAR101
        """

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when LLM errors.

        # noqa: DAR101
        #"""

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when chain starts running.

        # noqa: DAR101
        """

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running.

        # noqa: DAR101
        #"""

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when chain errors.

        # noqa: DAR101
        """

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Run when tool starts running.

        # noqa: DAR101
        """

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action.

        # noqa: DAR101
        """
        pass

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running.

        # noqa: DAR101
        """

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when tool errors.

        # noqa: DAR101
        """

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on arbitrary text.

        # noqa: DAR101
        """

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run on agent end.

        # noqa: DAR101
        """
