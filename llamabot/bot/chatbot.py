"""Class definition for ChatBot."""
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
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
        self.model = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.chat_history = [SystemMessage(content=system_prompt)]

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
