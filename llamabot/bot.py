"""Bot abstractions that let me quickly build new GPT-based applications."""
import os

import openai
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


class SimpleBot:
    """Simple Bot that is primed with a system prompt, accepts a human message, and sends back a single response."""

    def __init__(self, system_prompt, temperature=0.0):
        """Initialize the SimpleBot.

        :param system_prompt: The system prompt to use.
        :param temperature: The temperature to use.
        """
        self.system_prompt = system_prompt
        self.model = ChatOpenAI(model_name="gpt-4", temperature=temperature)

    def __call__(self, human_message):
        """Call the SimpleBot.

        :param human_message: The human message to use.
        :return: The response to the human message, primed by the system prompt.
        """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=human_message),
        ]
        return self.model(messages)


class ChatBot:
    """Chat Bot that is primed with a system prompt, accepts a human message.

    Automatic chat memory management happens.

    h/t Andrew Giessel/GPT4 for the idea.
    """

    def __init__(self, system_prompt, temperature=0.0):
        """Initialize the ChatBot.

        :param system_prompt: The system prompt to use.
        :param temperature: The temperature to use.
        """
        self.model = ChatOpenAI(model_name="gpt-4", temperature=temperature)
        self.chat_history = [SystemMessage(content=system_prompt)]

    def __call__(self, human_message) -> str:
        """Call the ChatBot.

        :param human_message: The human message to use.
        :return: The response to the human message, primed by the system prompt.
        """
        self.chat_history.append(HumanMessage(content=human_message))
        response = self.model(self.chat_history)
        self.chat_history.append(response)
        return response.content

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
