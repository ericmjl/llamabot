"""Tests for StructuredBot."""

from llamabot.bot.structuredbot import StructuredBot
from llamabot.prompt_manager import prompt
from pydantic import BaseModel


@prompt
def phibot_sysprompt() -> str:
    """You are a bot that returns JSON data."""


class PhiBotOutput(BaseModel):
    """A dummy BaseModel for StructuredBot."""

    num: int
    txt: str


def test_structuredbot():
    """Test that StructuredBot returns a pydantic model."""
    bot = StructuredBot(phibot_sysprompt(), PhiBotOutput, model_name="ollama/phi3")
    response = bot("I need a number and a text.")
    assert isinstance(response, BaseModel)
