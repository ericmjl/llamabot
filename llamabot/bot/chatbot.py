"""Class definition for ChatBot."""

import contextvars


from llamabot.bot.simplebot import SimpleBot
from llamabot.config import default_language_model
from llamabot.recorder import autorecord
from llamabot.components.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from llamabot.components.history import History
from functools import partial

prompt_recorder_var = contextvars.ContextVar("prompt_recorder")


class ChatBot:
    """ChatBot that is primed with a system prompt, accepts a human message,
    and sends back a single response.

    It is essentially SimpleBot composed with History.

    :param system_prompt: The system prompt to use.
    :param session_name: The name of the session to use.
        This is used in the chat history class.
    :param temperature: The temperature to use.
    :param model_name: The model name to use.
    :param stream: Whether to stream the output to stdout.
    :param response_budget: The response budget to use, in terms of number of characters.
    :param chat_history_class: The chat history class to use.
        Should be a callable that returns an object
        that follows the History API.
        Defaults to History from `llamabot.components.history`.
    """

    def __init__(
        self,
        system_prompt: str,
        session_name: str,
        temperature=0.0,
        model_name=default_language_model(),
        stream=True,
        response_budget=2_000,
        chat_history_class: callable = partial(History),
    ):
        chat_history_class_kwargs = {
            "session_name": session_name,
        }
        self.bot = SimpleBot(
            system_prompt=system_prompt,
            temperature=temperature,
            model_name=model_name,
            stream=stream,
        )
        self.model_name = model_name
        self.chat_history = chat_history_class(**chat_history_class_kwargs)
        self.response_budget = response_budget
        self.session_name = session_name

    def __call__(self, human_message: str) -> AIMessage:
        """Call the ChatBot.

        :param human_message: The human message to use.
        :return: The response to the human message, primed by the system prompt.
        """
        human_message = HumanMessage(content=human_message)
        history = self.chat_history.retrieve(
            query=human_message, character_budget=self.response_budget
        )
        messages = [self.bot.system_prompt] + history + [human_message]
        response_string: str = self.bot.generate_response(messages)
        autorecord(human_message, response_string)

        response: AIMessage = AIMessage(content=response_string)
        self.chat_history.append(human_message)
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
