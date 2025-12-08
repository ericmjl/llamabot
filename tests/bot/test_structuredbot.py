"""Tests for StructuredBot."""

import json
from typing import Any, Dict, List, Optional

import pytest
from pydantic import BaseModel, Field, ValidationError

from llamabot.bot.structuredbot import StructuredBot
from llamabot.components.messages import SystemMessage
from llamabot.recorder import SpanList, enable_span_recording


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


def test_structuredbot_spans_property_returns_spanlist(tmp_path, monkeypatch):
    """Test that StructuredBot.spans property returns a SpanList."""
    enable_span_recording()
    db_path = tmp_path / "test_spans.db"

    # Patch the database path to use our test database
    monkeypatch.setattr("llamabot.recorder.find_or_set_db_path", lambda x: db_path)

    bot = StructuredBot(
        system_prompt="Return data according to the schema.",
        pydantic_model=TestModel,
        mock_response='{"required_field": "test", "number_field": 1}',
    )

    # Before any calls, spans should be empty SpanList
    spans = bot.spans
    assert isinstance(spans, SpanList)
    assert len(spans) == 0

    # Make a bot call
    bot("Test query")

    # After call, spans should contain spans
    spans = bot.spans
    assert isinstance(spans, SpanList)
    assert len(spans) > 0


def test_structuredbot_spans_property_inherited_from_simplebot():
    """Test that StructuredBot inherits .spans property from SimpleBot."""
    bot = StructuredBot(
        system_prompt="Test prompt",
        pydantic_model=TestModel,
    )

    # Verify .spans property exists and returns SpanList
    assert hasattr(bot, "spans")
    spans = bot.spans
    assert isinstance(spans, SpanList)


# Tests for new StructuredBot display methods
def test_format_field_type_simple_types():
    """Test format_field_type() with simple types."""
    from pydantic import BaseModel

    class SimpleModel(BaseModel):
        """Test model with simple field types."""

        string_field: str
        int_field: int

    bot = StructuredBot(
        system_prompt="Test prompt",
        pydantic_model=SimpleModel,
        model_name="ollama_chat/gemma2:2b",
    )

    schema = SimpleModel.model_json_schema()

    # Test string type
    field_type = bot.format_field_type(schema["properties"]["string_field"], schema)
    assert field_type == "string"

    # Test integer type
    field_type = bot.format_field_type(schema["properties"]["int_field"], schema)
    assert field_type == "integer"


def test_format_field_type_array():
    """Test format_field_type() with array types."""
    from pydantic import BaseModel
    from typing import List

    class ArrayModel(BaseModel):
        """Test model with array field type."""

        items: List[str]

    bot = StructuredBot(
        system_prompt="Test prompt",
        pydantic_model=ArrayModel,
        model_name="ollama_chat/gemma2:2b",
    )

    schema = ArrayModel.model_json_schema()
    field_type = bot.format_field_type(schema["properties"]["items"], schema)

    assert "array" in field_type.lower()


def test_format_field_type_optional():
    """Test format_field_type() with optional fields."""
    from pydantic import BaseModel
    from typing import Optional

    class OptionalModel(BaseModel):
        """Test model with optional field."""

        optional_field: Optional[str] = None

    bot = StructuredBot(
        system_prompt="Test prompt",
        pydantic_model=OptionalModel,
        model_name="ollama_chat/gemma2:2b",
    )

    schema = OptionalModel.model_json_schema()
    field_type = bot.format_field_type(schema["properties"]["optional_field"], schema)

    # Should show as union with null
    assert "|" in field_type or "null" in field_type


def test_render_field_required():
    """Test render_field() with required field."""
    from pydantic import BaseModel

    class RequiredModel(BaseModel):
        """Test model with required field."""

        required_field: str

    bot = StructuredBot(
        system_prompt="Test prompt",
        pydantic_model=RequiredModel,
        model_name="ollama_chat/gemma2:2b",
    )

    schema = RequiredModel.model_json_schema()
    field_html = bot.render_field(
        "required_field",
        schema["properties"]["required_field"],
        required=True,
        schema=schema,
    )

    assert "required_field" in field_html
    assert "required" in field_html


def test_render_field_optional():
    """Test render_field() with optional field."""
    from pydantic import BaseModel
    from typing import Optional

    class OptionalModel(BaseModel):
        """Test model with optional field."""

        optional_field: Optional[str] = None

    bot = StructuredBot(
        system_prompt="Test prompt",
        pydantic_model=OptionalModel,
        model_name="ollama_chat/gemma2:2b",
    )

    schema = OptionalModel.model_json_schema()
    field_html = bot.render_field(
        "optional_field",
        schema["properties"]["optional_field"],
        required=False,
        schema=schema,
    )

    assert "optional_field" in field_html
    assert "optional" in field_html


def test_render_field_with_default():
    """Test render_field() with default value."""
    from pydantic import BaseModel

    class DefaultModel(BaseModel):
        """Test model with default value."""

        field_with_default: str = "default_value"

    bot = StructuredBot(
        system_prompt="Test prompt",
        pydantic_model=DefaultModel,
        model_name="ollama_chat/gemma2:2b",
    )

    schema = DefaultModel.model_json_schema()
    field_html = bot.render_field(
        "field_with_default",
        schema["properties"]["field_with_default"],
        required=False,
        schema=schema,
    )

    assert "field_with_default" in field_html
    assert "default_value" in field_html or "Default" in field_html


def test_render_field_with_description():
    """Test render_field() with field description."""
    from pydantic import BaseModel, Field

    class DescribedModel(BaseModel):
        """Test model with field description."""

        described_field: str = Field(description="This is a test field")

    bot = StructuredBot(
        system_prompt="Test prompt",
        pydantic_model=DescribedModel,
        model_name="ollama_chat/gemma2:2b",
    )

    schema = DescribedModel.model_json_schema()
    field_html = bot.render_field(
        "described_field",
        schema["properties"]["described_field"],
        required=True,
        schema=schema,
    )

    assert "described_field" in field_html
    assert "This is a test field" in field_html


def test_generate_schema_html_basic():
    """Test generate_schema_html() with basic model."""
    from pydantic import BaseModel

    class BasicModel(BaseModel):
        """Test model with basic fields."""

        field1: str
        field2: int

    bot = StructuredBot(
        system_prompt="Test prompt",
        pydantic_model=BasicModel,
        model_name="ollama_chat/gemma2:2b",
    )

    html = bot.generate_schema_html()

    assert "Pydantic Model Schema" in html
    assert "BasicModel" in html
    assert "field1" in html
    assert "field2" in html


def test_generate_schema_html_with_descriptions():
    """Test generate_schema_html() includes field descriptions."""
    from pydantic import BaseModel, Field

    class DescribedModel(BaseModel):
        """Test model with multiple described fields."""

        name: str = Field(description="The name field")
        age: int = Field(description="The age field")

    bot = StructuredBot(
        system_prompt="Test prompt",
        pydantic_model=DescribedModel,
        model_name="ollama_chat/gemma2:2b",
    )

    html = bot.generate_schema_html()

    assert "The name field" in html
    assert "The age field" in html


def test_repr_html_shows_config_and_schema():
    """Test that _repr_html_() shows both configuration and schema."""
    from pydantic import BaseModel

    class LocalTestModel(BaseModel):
        """Test model for config and schema display."""

        test_field: str

    bot = StructuredBot(
        system_prompt="Test prompt",
        pydantic_model=LocalTestModel,
        model_name="ollama_chat/gemma2:2b",
    )

    html = bot._repr_html_()

    # Should contain configuration
    assert "StructuredBot Configuration" in html
    assert "ollama_chat/gemma2:2b" in html
    assert "Test prompt" in html

    # Should contain schema
    assert "Pydantic Model Schema" in html
    assert "LocalTestModel" in html
    assert "test_field" in html


def test_repr_html_uses_generate_methods():
    """Test that _repr_html_() uses generate_config_html() and generate_schema_html()."""
    from pydantic import BaseModel

    class LocalTestModel2(BaseModel):
        """Test model for generate methods."""

        test_field: str

    bot = StructuredBot(
        system_prompt="Test prompt",
        pydantic_model=LocalTestModel2,
        model_name="ollama_chat/gemma2:2b",
    )

    # Get combined HTML
    repr_html = bot._repr_html_()

    # Schema should be part of repr_html
    # Note: We can't do exact string match because repr_html modifies the HTML
    assert "Pydantic Model Schema" in repr_html
    assert "test_field" in repr_html
