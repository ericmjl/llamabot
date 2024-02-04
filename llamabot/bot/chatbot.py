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

prompt_recorder_var = contextvars.ContextVar("prompt_recorder")


class ChatBot(SimpleBot, History):
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
    ):
        SimpleBot.__init__(
            self,
            system_prompt=system_prompt,
            temperature=temperature,
            model_name=model_name,
            stream=stream,
        )
        History.__init__(self, session_name=session_name)
        self.response_budget = response_budget

    def __call__(self, message: str) -> AIMessage:
        """Call the ChatBot.

        :param human_message: The human message to use.
        :return: The response to the human message, primed by the system prompt.
        """
        human_message = HumanMessage(content=message)
        history = self.retrieve(
            query=human_message, character_budget=self.response_budget
        )
        messages = [self.system_prompt] + history + [human_message]
        if self.stream:
            return self.stream_response(messages)
        response = self.generate_response(messages)
        autorecord(message, response.content)

        self.append(human_message)
        self.append(response)
        return response

    def __repr__(self):
        """Return a string representation of the ChatBot.

        :return: A string representation of the ChatBot.
        """
        representation = ""
        prefix = ""

        for message in self.messages:
            if isinstance(message, SystemMessage):
                prefix = "[System]\n"
            elif isinstance(message, HumanMessage):
                prefix = "[Human]\n"
            elif isinstance(message, AIMessage):
                prefix = "[AI]\n"

            representation += f"{prefix}{message.content}" + "\n\n"
        return representation
