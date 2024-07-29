"""Tests for StructuredBot."""

from llamabot.bot.structuredbot import StructuredBot
from llamabot.prompt_manager import prompt
from pydantic import BaseModel


@prompt
def phibot_sysprompt() -> str:
    """You are a bot that returns JSON data according to the schema provided to you.

    You will be provided with a schema. Return JSON that follows the schema.
    Do not return the schema itself.
    """


class PhiBotOutput(BaseModel):
    """A dummy BaseModel for StructuredBot."""

    num: int
    txt: str


def test_structuredbot():
    """Test that StructuredBot returns a pydantic model."""
    bot = StructuredBot(phibot_sysprompt(), PhiBotOutput, model_name="ollama/phi3")
    response = bot("I need a number and a text.", num_attempts=50)
    assert isinstance(response, BaseModel)
