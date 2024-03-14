"""Class definition for ChatBot."""

import contextvars
from typing import Generator, Union


from llamabot.bot.simplebot import SimpleBot
from llamabot.config import default_language_model
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
    :param stream_target: The stream target to use.
        Should be one of ("stdout", "panel", "api").
    :param response_budget: The response budget to use, in terms of number of characters.
    :param completion_kwargs: Additional keyword arguments to pass to the completion function.
    """

    def __init__(
        self,
        system_prompt: str,
        session_name: str,
        stream_target: str = "stdout",
        temperature=0.0,
        model_name=default_language_model(),
        response_budget=2_000,
        **completion_kwargs,
    ):
        if stream_target == "api":
            raise ValueError("ChatBot does not support API streaming.")
        SimpleBot.__init__(
            self,
            system_prompt=system_prompt,
            temperature=temperature,
            model_name=model_name,
            stream_target=stream_target,
            **completion_kwargs,
        )
        History.__init__(self, session_name=session_name)
        self.response_budget = response_budget

    def __call__(self, message: str) -> Union[AIMessage, Generator]:
        """Call the ChatBot.

        :param human_message: The human message to use.
        :return: The response to the human message, primed by the system prompt.
        """
        human_message = HumanMessage(content=message)
        history = self.retrieve(
            query=human_message, character_budget=self.response_budget
        )
        messages = [self.system_prompt] + history + [human_message]
        match self.stream_target:
            case "stdout":
                response = self.stream_stdout(messages)
                self.append(human_message)
                self.append(response)
                return response
            case "panel":
                return self.stream_panel(messages)

        return AIMessage(content="")

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
