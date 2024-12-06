"""Class definition for SimpleBot."""

import contextvars
from typing import Generator, Optional, Union


from llamabot.components.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
    process_messages,
)
from llamabot.recorder import autorecord, sqlite_log
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
    :param model_name: The name of the model to use.
    :param stream_target: The target to stream the response to.
        Should be one of ("stdout", "panel", "api").
    :param json_mode: Whether to print debug messages.
    :param api_key: The OpenAI API key to use.
    :param mock_response: A mock response to use, for testing purposes only.
    :param completion_kwargs: Additional keyword arguments to pass to the completion function.
    """

    def __init__(
        self,
        system_prompt: str,
        temperature=0.0,
        model_name=default_language_model(),
        stream_target: str = "stdout",
        json_mode: bool = False,
        api_key: Optional[str] = None,
        mock_response: Optional[str] = None,
        **completion_kwargs,
    ):
        # Validate `stream_target.
        if stream_target not in ("stdout", "panel", "api", "none"):
            raise ValueError(
                f"stream_target must be one of ('stdout', 'panel', 'api', 'none'), got {stream_target}."
            )

        self.system_prompt = system_prompt
        if isinstance(system_prompt, str):
            self.system_prompt = SystemMessage(content=system_prompt)

        self.model_name = model_name
        self.stream_target = stream_target
        self.temperature = temperature
        self.json_mode = json_mode
        self.api_key = api_key
        self.mock_response = mock_response
        self.completion_kwargs = completion_kwargs

        # Set special cases for for o1 models.
        if model_name in ["o1-preview", "o1-mini"]:
            self.system_prompt = HumanMessage(
                content=system_prompt.content, prompt_hash=system_prompt.prompt_hash
            )
            self.temperature = 1.0
            self.stream_target = "none"

    def __call__(
        self, *human_messages: Union[str, BaseMessage, list[Union[str, BaseMessage]]]
    ) -> Union[AIMessage, Generator]:
        """Call the SimpleBot.

        :param human_messages: One or more human messages to use, or lists of messages.
        :return: The response to the human messages, primed by the system prompt.
        """
        processed_messages = process_messages(human_messages)
        messages = [self.system_prompt] + processed_messages
        match self.stream_target:
            case "stdout":
                return self.stream_stdout(messages)
            case "panel":
                return self.stream_panel(messages)
            case "api":
                return self.stream_api(messages)
            case "none":
                return self.stream_none(messages)
        return AIMessage(content="")

    # @cache.memoize(ignore={0})
    def stream_stdout(self, messages: list[BaseMessage]) -> AIMessage:
        """Stream the response to stdout.

        :param messages: A list of messages.
        """
        response = _make_response(self, messages)
        message = ""
        for chunk in response:
            delta = chunk.choices[0].delta["content"]
            if delta is not None:
                print(delta, end="")
                message += delta
        response_message = AIMessage(content=message)
        autorecord(messages[-1].content, response_message.content)
        sqlite_log(self, messages + [response_message])
        return response_message

    def stream_panel(self, messages: list[BaseMessage]) -> Generator:
        """Stream the response to a Panel app.

        :param messages: A list of messages.
        """
        response = _make_response(self, messages)
        response_message = ""
        for chunk in response:
            delta = chunk.choices[0].delta["content"]
            if delta is not None:
                response_message += delta
                yield response_message
        autorecord(messages[-1].content, response_message)
        sqlite_log(self, messages + [AIMessage(content=response_message)])

    # @cache.memoize(ignore={0})
    def stream_none(self, messages: list[BaseMessage]) -> AIMessage:
        """Stream the response to None.

        :param messages: A list of messages.
        """
        response = _make_response(self, messages, stream=False)
        response_message = AIMessage(
            content=response.choices[0].message.content.strip()
        )
        autorecord(messages[-1].content, response_message.content)
        sqlite_log(self, messages + [response_message])
        return response_message

    def stream_api(self, messages: list[BaseMessage]) -> Generator:
        """Stream the response to an API.

        :param messages: A list of messages.
        """
        response = _make_response(self, messages)
        response_message = ""
        for chunk in response:
            delta = chunk.choices[0].delta["content"]
            if delta is not None:
                response_message += delta
                yield delta
        autorecord(messages[-1].content, response_message)
        sqlite_log(self, messages + [AIMessage(content=response_message)])

    # @cache.memoize(ignore={0})
    def generate_response(self, messages: list[BaseMessage]) -> AIMessage:
        """Generate a response from the given messages.

        :param messages: A list of messages.
        :return: The response to the messages.
        """
        response = _make_response(self, messages)
        response_message = AIMessage(
            content=response.choices[0].message.content.strip()
        )
        autorecord(messages[-1].content, response_message.content)
        sqlite_log(self, messages + [response_message])
        return response_message

    def stream_response(self, messages: list[BaseMessage]):
        """Stream the response from the given messages.

        This is intended to be used with Panel's ChatInterface as part of the callback.

        :param messages: A list of messages.
        :return: A generator that yields the response.
        """
        response = _make_response(self, messages)
        message = ""
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta is not None:
                message += delta
                print(delta, end="")
                yield message
        print()


def _make_response(bot: SimpleBot, messages: list[BaseMessage], stream: bool = True):
    """Make a response from the given messages.

    :param bot: A SimpleBot
    :param messages: A list of Messages.
    :return: A response object.
    """
    from litellm import completion

    messages_dumped: list[dict] = [m.model_dump() for m in messages]
    completion_kwargs = dict(
        model=bot.model_name,
        messages=messages_dumped,
        temperature=bot.temperature,
        stream=stream,
    )
    completion_kwargs.update(bot.completion_kwargs)

    if bot.mock_response:
        completion_kwargs["mock_response"] = bot.mock_response
    if bot.json_mode:
        completion_kwargs["response_format"] = {"type": "json_object"}
    if bot.api_key:
        completion_kwargs["api_key"] = bot.api_key
    return completion(**completion_kwargs)

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
