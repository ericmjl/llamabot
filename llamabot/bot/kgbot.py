"""Knowledge Graph bot."""

import json
from llamabot.components.messages import HumanMessage
from llamabot.config import default_language_model

from llamabot.prompt_manager import prompt
from .simplebot import SimpleBot


@prompt(role="system")
def kgbot_sysprompt() -> str:
    """You are an expert ontologist. You are tasked with taking in a chunk of text
    and extrapolating as many relationships as possible from that text.

    For each relationship, return a JSON according to the following schema:

    {
        "subject": "string",
        "predicate": "string",
        "object": "string"
    }
    """


class KGBot(SimpleBot):
    """KGBot is the Knowledge Graph bot.

    Tested with mistral-medium.

    It takes in a chunk of text and returns a JSON of triplets.
    """

    def __init__(self):
        SimpleBot.__init__(
            self, system_prompt=kgbot_sysprompt(), model_name=default_language_model()
        )

    def __call__(self, query: str) -> dict:
        """Call the bot with a query and return a JSON of triplets.

        :param query: The query to use.
        """
        response = self.generate_response([HumanMessage(content=query)])
        return json.loads(response.content)
