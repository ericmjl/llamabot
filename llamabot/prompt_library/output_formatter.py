"""Prompts and bots for validating outputs."""
import json

from outlines import text

from llamabot import SimpleBot


@text.prompt
def output_formatter_sysprompt():
    """You are an output formatter.

    You will be presented with a text that should be turned into a JSON.
    Please convert it to a JSON.
    Ensure that the keys are always in snake case.
    Use as few words as possible for the keys.
    """


def output_formatter() -> SimpleBot:
    """Return the output formatter bot.

    :return: The output formatter bot.
    """
    return SimpleBot(
        output_formatter_sysprompt(),
        temperature=0.2,
    )


def coerce_dict(bot_response: str) -> dict:
    """Coerce the bot_response to be a dictionary.

    :param bot_response: The bot response to be coerced.
    :return: The bot response as a dictionary.
    :raises ValueError: If the bot response cannot be coerced to a dictionary.
    """
    formatter: SimpleBot = output_formatter()

    max_retries = 5
    num_retries = 0
    is_dict = False
    while not is_dict and num_retries < max_retries:
        try:
            parsed_dictionary = json.loads(bot_response)
            is_dict = True
            return parsed_dictionary
        except Exception:
            pass
        bot_response = formatter(bot_response).content
        num_retries += 1
    raise ValueError(f"Could not coerce to dictionary after {max_retries} tries.")
