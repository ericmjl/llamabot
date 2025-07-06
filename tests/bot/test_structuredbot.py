"""Tests for StructuredBot."""

import json
import pytest
from typing import Optional, List, Any, Dict

from pydantic import BaseModel, Field, ValidationError

from llamabot.bot.structuredbot import StructuredBot
from llamabot.components.messages import SystemMessage


class TestModel(BaseModel):
    """Test model for StructuredBot tests."""

    required_field: str
    optional_field: Optional[str] = None
    number_field: int = Field(gt=0)  # must be positive


class ComplexTestModel(BaseModel):
    """A more complex test model with nested fields."""

    title: str
    count: int
    tags: List[str]
    metadata: Dict[str, Any] = {}


def test_structuredbot_initialization():
    """Test that StructuredBot can be properly initialized."""
    system_prompt = "Return data according to the schema provided."

    # Should not raise an error for ollama_chat models
    bot = StructuredBot(
        system_prompt=system_prompt,
        pydantic_model=TestModel(required_field="default", number_field=1),
        model_name="ollama_chat/gemma2:2b",
    )
    assert bot.model_name == "ollama_chat/gemma2:2b"
    assert isinstance(bot.pydantic_model, TestModel)
    assert not bot.allow_failed_validation


def test_structuredbot_valid_data(mocker):
    """Test that StructuredBot correctly processes valid data."""
    # Mock the API response with valid data
    mock_response = mocker.MagicMock()
    mock_response.content = '{"required_field": "test", "number_field": 1}'
    mocker.patch("llamabot.bot.structuredbot.make_response", return_value=mock_response)
    mocker.patch("llamabot.bot.structuredbot.stream_chunks", return_value=mock_response)
    mocker.patch(
        "llamabot.bot.structuredbot.extract_content",
        return_value='{"required_field": "test", "number_field": 1}',
    )
    mocker.patch("llamabot.bot.structuredbot.sqlite_log")

    bot = StructuredBot(
        system_prompt=SystemMessage(content="Test prompt"),
        pydantic_model=TestModel(required_field="default", number_field=1),
        model_name="ollama_chat/gemma2:2b",
        stream_target="none",
    )

    result = bot("test message")
    assert isinstance(result, TestModel)
    assert result.required_field == "test"
    assert result.number_field == 1
    assert result.optional_field is None


def test_structuredbot_invalid_data_retry(mocker):
    """Test that StructuredBot tries to fix validation errors."""
    # Setup mocks for the first call (invalid response)
    first_invalid_response = mocker.MagicMock()
    first_invalid_response.content = '{"required_field": "test", "number_field": 0}'  # Invalid: number_field must be > 0

    # Setup mocks for the second call (valid response)
    second_valid_response = mocker.MagicMock()
    second_valid_response.content = '{"required_field": "fixed", "number_field": 5}'

    # Mock make_response to return our responses in sequence
    make_response_mock = mocker.patch("llamabot.bot.structuredbot.make_response")
    make_response_mock.side_effect = [first_invalid_response, second_valid_response]

    # Mock stream_chunks to return the same
    stream_chunks_mock = mocker.patch("llamabot.bot.structuredbot.stream_chunks")
    stream_chunks_mock.side_effect = [first_invalid_response, second_valid_response]

    # Mock extract_content to return our content
    extract_content_mock = mocker.patch("llamabot.bot.structuredbot.extract_content")
    extract_content_mock.side_effect = [
        '{"required_field": "test", "number_field": 0}',
        '{"required_field": "fixed", "number_field": 5}',
    ]

    # Mock sqlite_log
    mocker.patch("llamabot.bot.structuredbot.sqlite_log")

    bot = StructuredBot(
        system_prompt=SystemMessage(content="Test prompt"),
        pydantic_model=TestModel(required_field="default", number_field=1),
        model_name="ollama_chat/gemma2:2b",
        stream_target="none",
    )

    # Test that it retries and succeeds
    result = bot("test message")

    # Verify that it made two calls
    assert make_response_mock.call_count == 2
    assert stream_chunks_mock.call_count == 2
    assert extract_content_mock.call_count == 2

    # Verify the final result
    assert isinstance(result, TestModel)
    assert result.required_field == "fixed"
    assert result.number_field == 5


def test_structuredbot_allow_failed_validation(mocker):
    """Test that StructuredBot returns partial results when allow_failed_validation is True."""
    # Mock an invalid response
    mock_response = mocker.MagicMock()
    mock_response.content = '{"required_field": "test", "number_field": 0}'  # Invalid: number_field must be > 0

    # Setup mocks
    mocker.patch("llamabot.bot.structuredbot.make_response", return_value=mock_response)
    mocker.patch("llamabot.bot.structuredbot.stream_chunks", return_value=mock_response)
    mocker.patch(
        "llamabot.bot.structuredbot.extract_content",
        return_value='{"required_field": "test", "number_field": 0}',
    )
    mocker.patch("llamabot.bot.structuredbot.sqlite_log")

    bot = StructuredBot(
        system_prompt=SystemMessage(content="Test prompt"),
        pydantic_model=TestModel(required_field="default", number_field=1),
        model_name="ollama_chat/gemma2:2b",
        stream_target="none",
        allow_failed_validation=True,  # Allow failed validation
    )

    # Should return a result with validation errors since allow_failed_validation is True
    result = bot("test message", num_attempts=1)

    # Verify the result is not None and has the expected values
    assert result is not None
    assert isinstance(result, TestModel)
    # After confirming it's a TestModel, we can safely access its attributes
    assert result.required_field == "test"
    assert result.number_field == 0  # Invalid value but returned anyway


def test_structuredbot_complex_model(mocker):
    """Test that StructuredBot correctly processes complex model data."""
    # Valid complex model data
    complex_data = {
        "title": "Test Title",
        "count": 42,
        "tags": ["test", "example", "complex"],
        "metadata": {"source": "unit test", "priority": "high"},
    }

    # Mock the API response
    mock_response = mocker.MagicMock()
    mock_response.content = json.dumps(complex_data)

    # Setup mocks
    mocker.patch("llamabot.bot.structuredbot.make_response", return_value=mock_response)
    mocker.patch("llamabot.bot.structuredbot.stream_chunks", return_value=mock_response)
    mocker.patch(
        "llamabot.bot.structuredbot.extract_content",
        return_value=json.dumps(complex_data),
    )
    mocker.patch("llamabot.bot.structuredbot.sqlite_log")

    bot = StructuredBot(
        system_prompt=SystemMessage(content="Test prompt"),
        pydantic_model=ComplexTestModel(title="Default", count=1, tags=["default"]),
        model_name="ollama_chat/gemma2:2b",
        stream_target="none",
    )

    result = bot("Give me a complex test object")

    assert isinstance(result, ComplexTestModel)
    assert result.title == "Test Title"
    assert result.count == 42
    assert len(result.tags) == 3
    assert "test" in result.tags
    assert result.metadata["source"] == "unit test"
    assert result.metadata["priority"] == "high"


def test_structuredbot_max_attempts_reached(mocker):
    """Test that StructuredBot raises ValidationError when max attempts are reached."""
    # Mock a consistently invalid response
    mock_response = mocker.MagicMock()
    mock_response.content = '{"required_field": "test", "number_field": 0}'  # Invalid

    # Setup mocks to always return invalid data
    mocker.patch("llamabot.bot.structuredbot.make_response", return_value=mock_response)
    mocker.patch("llamabot.bot.structuredbot.stream_chunks", return_value=mock_response)
    mocker.patch(
        "llamabot.bot.structuredbot.extract_content",
        return_value='{"required_field": "test", "number_field": 0}',
    )
    mocker.patch("llamabot.bot.structuredbot.sqlite_log")

    bot = StructuredBot(
        system_prompt=SystemMessage(content="Test prompt"),
        pydantic_model=TestModel(required_field="default", number_field=1),
        model_name="ollama_chat/gemma2:2b",
        stream_target="none",
        allow_failed_validation=False,  # Don't allow failed validation
    )

    # Should raise ValidationError after max attempts
    with pytest.raises(ValidationError) as exc_info:
        bot("test message", num_attempts=3)

    assert "number_field" in str(exc_info.value)
    assert "greater than" in str(exc_info.value)
