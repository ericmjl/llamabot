"""Tests for StructuredBot."""

import pytest

from llamabot.components.messages import SystemMessage
from llamabot.bot.structuredbot import StructuredBot
from llamabot.prompt_manager import prompt
from pydantic import BaseModel, Field
from typing import Optional


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


@pytest.mark.xfail(reason="Can be flaky b/c it depends on a small model.")
def test_structuredbot():
    """Test that StructuredBot returns a pydantic model."""
    bot = StructuredBot(phibot_sysprompt(), PhiBotOutput, model_name="ollama/phi3")
    response = bot("I need a number and a text.", num_attempts=3)
    assert isinstance(response, BaseModel)


class TestModel(BaseModel):
    """Test model for StructuredBot tests."""

    required_field: str
    optional_field: Optional[str] = None
    number_field: int = Field(gt=0)  # must be positive


def test_structuredbot_valid_data(mocker):
    """Test that StructuredBot correctly processes valid data."""
    # Mock the API response with valid data
    mock_response = mocker.MagicMock()
    mock_response.content = '{"required_field": "test", "number_field": 1}'
    mocker.patch(
        "llamabot.bot.structuredbot.SimpleBot.stream_none", return_value=mock_response
    )

    bot = StructuredBot(
        system_prompt=SystemMessage(content="Test prompt"),
        pydantic_model=TestModel,
        stream_target="none",
        model_name="gpt-4o",
    )

    result = bot("test message")
    assert isinstance(result, TestModel)
    assert result.required_field == "test"
    assert result.number_field == 1


def test_structuredbot_unsupported_model():
    """Test that StructuredBot raises ValueError for models without structured response support."""
    with pytest.raises(ValueError) as exc_info:
        StructuredBot(
            system_prompt="Test prompt",
            pydantic_model=TestModel,
            model_name="mistral/mistral-medium",
        )

    assert "does not support structured responses" in str(exc_info.value)
    assert "mistral-medium" in str(exc_info.value)
