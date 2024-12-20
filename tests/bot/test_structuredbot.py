"""Tests for StructuredBot."""

import pytest
from typing import Optional

from llamabot.bot.structuredbot import StructuredBot
from llamabot.prompt_manager import prompt
from pydantic import BaseModel, Field
from llamabot.components.messages import SystemMessage


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


def test_structuredbot_allow_failed_validation(mocker):
    """Test that StructuredBot returns partial data when allow_failed_validation=True."""
    # Mock the API response with invalid data (negative number)
    mock_response = mocker.MagicMock()
    mock_response.content = {"required_field": "test", "number_field": -1}
    mocker.patch(
        "llamabot.bot.structuredbot.SimpleBot.stream_none", return_value=mock_response
    )

    bot = StructuredBot(
        system_prompt=SystemMessage(content="Test prompt"),
        pydantic_model=TestModel,
        allow_failed_validation=True,
        stream_target="none",
    )

    # Should return object even with invalid data
    result = bot("test message")
    assert isinstance(result, TestModel)
    assert result.required_field == "test"
    assert result.number_field == -1  # Invalid value is retained


def test_structuredbot_disallow_failed_validation(mocker):
    """Test that StructuredBot raises validation error when allow_failed_validation=False."""
    # Mock the API response with invalid data (negative number)
    mock_response = mocker.MagicMock()
    mock_response.content = '{"required_field": "test", "number_field": -1}'
    mocker.patch(
        "llamabot.bot.structuredbot.SimpleBot.stream_none", return_value=mock_response
    )

    bot = StructuredBot(
        system_prompt=SystemMessage(content="Test prompt"),
        pydantic_model=TestModel,
        allow_failed_validation=False,
        stream_target="none",
    )

    # Should raise ValidationError
    with pytest.raises(Exception):  # Using general Exception as it might be wrapped
        bot("test message")


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
    )

    result = bot("test message")
    assert isinstance(result, TestModel)
    assert result.required_field == "test"
    assert result.number_field == 1
