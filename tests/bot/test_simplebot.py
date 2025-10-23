"""
This module contains tests for the SimpleBot class and its component functions.

It includes tests for the following:
- SimpleBot initialization
- SimpleBot call method
- make_response function
- stream_chunks function
- extract_tool_calls function
- extract_content function
"""

import hashlib
from unittest.mock import patch, MagicMock, call
import pytest
from typing import Optional, List, Dict, Any

from hypothesis import given, settings, strategies as st
from pydantic import BaseModel

from llamabot.bot.simplebot import (
    SimpleBot,
    make_response,
    stream_chunks,
    extract_tool_calls,
    extract_content,
)
from llamabot.components.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from litellm import ModelResponse


# Helper strategies for hypothesis testing
valid_stream_targets = st.one_of(
    st.just("stdout"), st.just("panel"), st.just("api"), st.just("none")
)
valid_o1_models = st.one_of(st.just("o1-preview"), st.just("o1-mini"))
valid_non_o1_models = st.just("ollama_chat/gemma2:2b")


# Helper functions for testing
def create_mock_model_response(
    content: Optional[str] = None, tool_calls: Optional[List[Dict[str, Any]]] = None
) -> MagicMock:
    """Create a mock ModelResponse object for testing.

    :param content: Content to include in the response
    :param tool_calls: Tool calls to include in the response
    :return: A mock ModelResponse object
    """
    mock_response = MagicMock()
    mock_message = MagicMock()
    mock_message.content = content
    mock_message.tool_calls = tool_calls
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    return mock_response


def generate_mock_response(system_prompt: str, human_message: str) -> str:
    """Generate a unique mock response based on system_prompt and human_message.

    This function creates a deterministic mock response by hashing the combination
    of the system prompt and human message.

    :param system_prompt: The system prompt used in the bot.
    :param human_message: The message input by the human.
    :return: A unique mock response string.
    """
    combined = f"{system_prompt}:{human_message}"
    hash_object = hashlib.md5(combined.encode())
    return f"Mock response for {hash_object.hexdigest()}"


# Tests for SimpleBot initialization
@given(
    system_prompt=st.text(min_size=1),
    temperature=st.floats(min_value=0, max_value=2),
    model_name=st.one_of(valid_o1_models, valid_non_o1_models),
    stream_target=valid_stream_targets,
    json_mode=st.booleans(),
    api_key=st.one_of(st.none(), st.text(min_size=5)),
    mock_response=st.one_of(st.none(), st.text()),
)
@settings(deadline=None)
def test_simple_bot_init(
    system_prompt,
    temperature,
    model_name,
    stream_target,
    json_mode,
    api_key,
    mock_response,
):
    """Test that the SimpleBot is initialized correctly.

    :param system_prompt: The system prompt to use.
    :param temperature: The model temperature to use.
    :param model_name: The name of the model to use.
    :param stream_target: The target to stream the response to.
    :param json_mode: Whether to enable JSON mode.
    :param api_key: The API key to use.
    :param mock_response: A mock response to use.
    """
    bot = SimpleBot(
        system_prompt=system_prompt,
        temperature=temperature,
        model_name=model_name,
        stream_target=stream_target,
        json_mode=json_mode,
        api_key=api_key,
        mock_response=mock_response,
    )

    # First check the model name to determine expected behavior
    if model_name in ["o1-preview", "o1-mini"]:
        # For o1 models, system_prompt should be converted to HumanMessage
        assert isinstance(bot.system_prompt, HumanMessage)
        assert bot.system_prompt.content == system_prompt
        assert bot.temperature == 1.0
        assert bot.stream_target == "none"
    else:
        # For non-o1 models, check that system_prompt is converted to SystemMessage if it's a string
        if isinstance(system_prompt, str):
            assert isinstance(bot.system_prompt, SystemMessage)
            assert bot.system_prompt.content == system_prompt
        # And check that temperature and stream_target are preserved
        assert bot.temperature == temperature
        assert bot.stream_target == stream_target

    # Common assertions for all model types
    assert bot.model_name == model_name
    assert bot.json_mode == json_mode
    assert bot.api_key == api_key
    assert bot.mock_response == mock_response
    assert bot.completion_kwargs == {}


def test_simple_bot_init_invalid_stream_target():
    """Test that SimpleBot initialization raises ValueError for invalid stream_target."""
    with pytest.raises(ValueError, match="stream_target must be one of"):
        SimpleBot(system_prompt="Test prompt", stream_target="invalid")


# Tests for SimpleBot call method
@patch("llamabot.bot.simplebot.make_response")
@patch("llamabot.bot.simplebot.stream_chunks")
@patch("llamabot.bot.simplebot.extract_tool_calls")
@patch("llamabot.bot.simplebot.extract_content")
@patch("llamabot.bot.simplebot.sqlite_log")
@given(
    system_prompt=st.text(min_size=1),
    human_message=st.text(min_size=1),
)
@settings(deadline=None, max_examples=10)
def test_simple_bot_call(
    mock_sqlite_log,
    mock_extract_content,
    mock_extract_tool_calls,
    mock_stream_chunks,
    mock_make_response,
    system_prompt,
    human_message,
):
    """Test the SimpleBot call method.

    :param mock_sqlite_log: Mock for sqlite_log function
    :param mock_extract_content: Mock for extract_content function
    :param mock_extract_tool_calls: Mock for extract_tool_calls function
    :param mock_stream_chunks: Mock for stream_chunks function
    :param mock_make_response: Mock for make_response function
    :param system_prompt: The system prompt to use
    :param human_message: The human message to use
    """
    # Reset mocks before each example
    mock_make_response.reset_mock()
    mock_stream_chunks.reset_mock()
    mock_extract_tool_calls.reset_mock()
    mock_extract_content.reset_mock()
    mock_sqlite_log.reset_mock()

    # Set up mocks
    mock_response = MagicMock()
    mock_make_response.return_value = mock_response
    mock_stream_chunks.return_value = mock_response
    mock_extract_tool_calls.return_value = []
    mock_extract_content.return_value = "AI response"

    # Create bot and call it
    bot = SimpleBot(system_prompt=system_prompt)
    result = bot(human_message)

    # Check that the correct functions were called
    mock_make_response.assert_called_once()
    mock_stream_chunks.assert_called_once_with(mock_response, target="stdout")
    mock_extract_tool_calls.assert_called_once_with(mock_response)
    mock_extract_content.assert_called_once_with(mock_response)
    mock_sqlite_log.assert_called_once()

    # Check the result
    assert isinstance(result, AIMessage)
    assert result.content == "AI response"
    assert result.tool_calls == []


@patch("llamabot.bot.simplebot.make_response")
@patch("llamabot.bot.simplebot.stream_chunks")
@patch("llamabot.bot.simplebot.extract_tool_calls")
@patch("llamabot.bot.simplebot.extract_content")
@patch("llamabot.bot.simplebot.sqlite_log")
def test_simple_bot_call_with_chat_memory(
    mock_sqlite_log,
    mock_extract_content,
    mock_extract_tool_calls,
    mock_stream_chunks,
    mock_make_response,
):
    """Test the SimpleBot call method with chat memory."""
    # Set up mocks
    mock_response = MagicMock()
    mock_make_response.return_value = mock_response
    mock_stream_chunks.return_value = mock_response
    mock_extract_tool_calls.return_value = []
    mock_extract_content.return_value = "AI response"

    # Create mock chat memory
    mock_chat_memory = MagicMock()
    mock_chat_memory.retrieve.return_value = [
        "Previous message 1",
        "Previous message 2",
    ]

    # Create bot and call it
    bot = SimpleBot(system_prompt="Test prompt", memory=mock_chat_memory)
    _ = bot("Human message")

    # Check that chat memory was used
    mock_chat_memory.retrieve.assert_called_once()
    assert mock_chat_memory.append.call_count == 2  # User message + assistant response

    # Check that make_response was called with the correct messages
    args, _ = mock_make_response.call_args
    assert len(args[1]) > 1  # Should include system prompt and memory messages


# Tests for make_response function
@patch("litellm.completion")
def test_make_response_basic(mock_completion):
    """Test the make_response function with basic parameters."""
    # Set up mock
    mock_completion.return_value = "Mock response"

    # Create test data
    bot = SimpleBot(system_prompt="Test prompt")
    messages = [
        SystemMessage(content="Test prompt"),
        HumanMessage(content="Test message"),
    ]

    # Call function
    result = make_response(bot, messages)

    # Check result
    assert result == "Mock response"

    # Check that completion was called with the correct parameters
    mock_completion.assert_called_once()
    _, kwargs = mock_completion.call_args
    assert kwargs["model"] == bot.model_name
    assert kwargs["temperature"] == bot.temperature
    assert kwargs["stream"] is True
    assert len(kwargs["messages"]) == 2


@patch("litellm.completion")
def test_make_response_with_mock_response(mock_completion):
    """Test the make_response function with a mock response."""
    # Set up mock
    mock_completion.return_value = "Mock response"

    # Create test data
    bot = SimpleBot(system_prompt="Test prompt", mock_response="Test mock response")
    messages = [
        SystemMessage(content="Test prompt"),
        HumanMessage(content="Test message"),
    ]

    # Call function
    _ = make_response(bot, messages)

    # Check that completion was called with the mock_response parameter
    _, kwargs = mock_completion.call_args
    assert kwargs["mock_response"] == "Test mock response"


@patch("litellm.completion")
def test_make_response_with_json_mode(mock_completion):
    """Test the make_response function with JSON mode enabled."""
    # Set up mock
    mock_completion.return_value = "Mock response"

    # Create test Pydantic model
    class TestModel(BaseModel):
        """A test Pydantic model."""

        field1: str
        field2: int

    # Create test data
    bot = SimpleBot(system_prompt="Test prompt", json_mode=True)
    bot.pydantic_model = TestModel
    messages = [
        SystemMessage(content="Test prompt"),
        HumanMessage(content="Test message"),
    ]

    # Call function
    _ = make_response(bot, messages)

    # Check that completion was called with the response_format parameter
    _, kwargs = mock_completion.call_args
    assert kwargs["response_format"] == TestModel


@patch("litellm.completion")
def test_make_response_json_mode_without_pydantic_model(mock_completion):
    """Test that make_response raises ValueError when json_mode is True but no pydantic_model is set."""
    # Create test data
    bot = SimpleBot(system_prompt="Test prompt", json_mode=True)
    messages = [
        SystemMessage(content="Test prompt"),
        HumanMessage(content="Test message"),
    ]

    # Call function and check that it raises ValueError
    with pytest.raises(ValueError, match="Please set a pydantic_model"):
        make_response(bot, messages)


@patch("litellm.completion")
def test_make_response_json_mode_with_invalid_pydantic_model(mock_completion):
    """Test that make_response raises ValueError when json_mode is True but pydantic_model is invalid."""
    # Create test data
    bot = SimpleBot(system_prompt="Test prompt", json_mode=True)
    bot.pydantic_model = str  # Not a BaseModel
    messages = [
        SystemMessage(content="Test prompt"),
        HumanMessage(content="Test message"),
    ]

    # Call function and check that it raises ValueError
    with pytest.raises(
        ValueError, match="pydantic_model must be a Pydantic BaseModel class"
    ):
        make_response(bot, messages)


@patch("litellm.completion")
def test_make_response_with_tools(mock_completion):
    """Test the make_response function with tools."""
    # Set up mock
    mock_completion.return_value = "Mock response"

    # Create test data
    bot = SimpleBot(system_prompt="Test prompt")
    bot.tools = [{"type": "function", "function": {"name": "test_function"}}]
    messages = [
        SystemMessage(content="Test prompt"),
        HumanMessage(content="Test message"),
    ]

    # Call function
    _ = make_response(bot, messages)

    # Check that completion was called with the tools parameter
    _, kwargs = mock_completion.call_args
    assert kwargs["tools"] == bot.tools
    assert kwargs["tool_choice"] == "auto"


@patch("litellm.completion")
def test_make_response_with_custom_tool_choice(mock_completion):
    """Test that bots can specify their own tool_choice."""
    # Set up mock
    mock_completion.return_value = "Mock response"

    # Create test data
    bot = SimpleBot(system_prompt="Test prompt")
    bot.tools = [{"type": "function", "function": {"name": "test_function"}}]
    bot.tool_choice = "none"
    messages = [
        SystemMessage(content="Test prompt"),
        HumanMessage(content="Test message"),
    ]

    # Call function
    _ = make_response(bot, messages)

    # Check that completion was called with the custom tool_choice
    _, kwargs = mock_completion.call_args
    assert kwargs["tools"] == bot.tools
    assert kwargs["tool_choice"] == "none"


# Tests for stream_chunks function
def test_stream_chunks_with_model_response():
    """Test that stream_chunks returns the input unchanged if it's already a ModelResponse."""
    # Create a minimal real ModelResponse instance
    model_response = ModelResponse(
        id="test-id",
        choices=[
            {
                "message": {
                    "content": "Test content",
                    "role": "assistant",
                },
                "index": 0,
                "finish_reason": "stop",
            }
        ],
        model="test-model",
        object="chat.completion",
        created=1234567890,
    )

    # Call function
    result = stream_chunks(model_response)

    # Check that the result is the same as the input
    assert result == model_response


@patch("builtins.print")
# Patch stream_chunk_builder to prevent sorting errors with mock objects
@patch("llamabot.bot.simplebot.stream_chunk_builder")
def test_stream_chunks_stdout(mock_stream_chunk_builder, mock_print):
    """Test the stream_chunks function with stdout target."""
    # Create mock generator
    mock_chunk1 = MagicMock()
    mock_chunk1.choices = [MagicMock()]
    mock_chunk1.choices[0].delta = {"content": "Hello"}

    mock_chunk2 = MagicMock()
    mock_chunk2.choices = [MagicMock()]
    mock_chunk2.choices[0].delta = {"content": " world"}

    mock_generator = MagicMock()
    mock_generator.__iter__.return_value = [mock_chunk1, mock_chunk2]

    # Call function
    _ = stream_chunks(mock_generator, target="stdout")

    # Check that print was called with the correct parameters
    mock_print.assert_has_calls([call("Hello", end=""), call(" world", end="")])


@patch("llamabot.bot.simplebot.stream_chunk_builder")
def test_stream_chunks_panel(mock_stream_chunk_builder):
    """Test the stream_chunks function with panel target."""
    # Create mock generator
    mock_chunk1 = MagicMock()
    mock_chunk1.choices = [MagicMock()]
    mock_chunk1.choices[0].delta = {"content": "Hello"}

    mock_chunk2 = MagicMock()
    mock_chunk2.choices = [MagicMock()]
    mock_chunk2.choices[0].delta = {"content": " world"}

    mock_generator = MagicMock()
    mock_generator.__iter__.return_value = [mock_chunk1, mock_chunk2]

    # Call function
    result = stream_chunks(mock_generator, target="panel")

    # Check that the result is a generator
    assert hasattr(result, "__next__")

    # Consume the generator
    result_list = list(result)

    # Check the results
    assert result_list == ["Hello", "Hello world"]


@patch("llamabot.bot.simplebot.stream_chunk_builder")
def test_stream_chunks_api(mock_stream_chunk_builder):
    """Test the stream_chunks function with api target."""
    # Create mock generator
    mock_chunk1 = MagicMock()
    mock_chunk1.choices = [MagicMock()]
    mock_chunk1.choices[0].delta = {"content": "Hello"}

    mock_chunk2 = MagicMock()
    mock_chunk2.choices = [MagicMock()]
    mock_chunk2.choices[0].delta = {"content": " world"}

    mock_generator = MagicMock()
    mock_generator.__iter__.return_value = [mock_chunk1, mock_chunk2]

    # Call function
    result = stream_chunks(mock_generator, target="api")

    # Check that the result is a generator
    assert hasattr(result, "__next__")

    # Consume the generator
    result_list = list(result)

    # Check the results
    assert result_list == ["Hello", " world"]


# Tests for extract_tool_calls function
@given(
    tool_calls=st.one_of(
        st.none(),
        st.lists(
            st.dictionaries(
                keys=st.text(min_size=1),
                values=st.one_of(st.text(), st.integers(), st.floats(), st.booleans()),
                min_size=1,
            ),
            min_size=0,
            max_size=3,
        ),
    )
)
def test_extract_tool_calls(tool_calls):
    """Test the extract_tool_calls function.

    :param tool_calls: Tool calls to include in the response
    """
    # Create mock ModelResponse
    mock_response = create_mock_model_response(tool_calls=tool_calls)

    # Call function
    result = extract_tool_calls(mock_response)

    # Check the result
    if tool_calls is None:
        assert result == []
    else:
        assert result == tool_calls


# Tests for extract_content function
@given(content=st.one_of(st.none(), st.text()))
def test_extract_content(content):
    """Test the extract_content function.

    :param content: Content to include in the response
    """
    # Create mock ModelResponse
    mock_response = create_mock_model_response(content=content)

    # Call function
    result = extract_content(mock_response)

    # Check the result
    if content is None:
        assert result == ""
    else:
        assert result == content
