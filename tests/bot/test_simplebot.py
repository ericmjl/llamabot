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
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, call, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from litellm import ModelResponse
from pydantic import BaseModel

from llamabot.bot.simplebot import (
    SimpleBot,
    extract_content,
    extract_tool_calls,
    make_response,
    stream_chunks,
)
from llamabot.components.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from llamabot.recorder import SpanList, enable_span_recording, get_spans, span

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


# Tests for bot span tracking
def test_bot_tracks_multiple_trace_ids(tmp_path, monkeypatch):
    """Test that bot tracks all trace_ids from multiple calls."""
    enable_span_recording()
    db_path = tmp_path / "test_spans.db"

    # Patch the database path to use our test database

    monkeypatch.setattr("llamabot.recorder.find_or_set_db_path", lambda x: db_path)

    bot = SimpleBot(system_prompt="Test prompt", mock_response="Response")

    # Make multiple bot calls
    bot("First call")
    bot("Second call")
    bot("Third call")

    # Check that bot tracked all trace_ids
    assert len(bot._trace_ids) == 3
    assert len(set(bot._trace_ids)) == 3  # All unique

    # Verify each trace_id has spans
    for trace_id in bot._trace_ids:
        spans = get_spans(trace_id=trace_id, db_path=db_path)
        assert len(spans) > 0
        # Each call should have a bot root span (using variable name "bot")
        root_spans = [s for s in spans if s.parent_span_id is None]
        assert len(root_spans) == 1
        assert root_spans[0].operation_name == "bot"


def test_bot_display_spans_shows_all_calls(tmp_path, monkeypatch):
    """Test that display_spans() shows spans from all bot calls."""
    enable_span_recording()
    db_path = tmp_path / "test_spans.db"

    # Patch the database path to use our test database
    monkeypatch.setattr("llamabot.recorder.find_or_set_db_path", lambda x: db_path)

    bot = SimpleBot(system_prompt="Test prompt", mock_response="Response")

    # Make multiple bot calls
    bot("First call")
    bot("Second call")
    bot("Third call")

    # Get HTML from display_spans
    html = bot.display_spans()

    # Verify HTML contains spans from all calls (using variable name "bot")
    assert "bot" in html
    # Should have multiple bot spans (one per call)
    # Check that we can find spans from all trace_ids
    all_spans = []
    for trace_id in bot._trace_ids:
        spans = get_spans(trace_id=trace_id, db_path=db_path)
        all_spans.extend(spans)

    # Verify HTML contains references to spans from all calls
    for span_obj in all_spans:
        assert span_obj.span_id in html or span_obj.operation_name in html


def test_bot_spans_exclude_manual_spans(tmp_path, monkeypatch):
    """Test that manual spans created separately don't appear in bot's spans."""
    enable_span_recording()
    db_path = tmp_path / "test_spans.db"

    # Patch the database path to use our test database
    monkeypatch.setattr("llamabot.recorder.find_or_set_db_path", lambda x: db_path)

    bot = SimpleBot(system_prompt="Test prompt", mock_response="Response")

    # Make a bot call
    bot("Bot call")

    # Create a manual span separately
    with span("manual_operation", db_path=db_path) as manual_span:
        manual_span["test"] = "value"

    # Get spans from bot
    bot_html = bot.display_spans()

    # Verify manual span is NOT in bot's spans
    assert manual_span.span_id not in bot_html
    assert "manual_operation" not in bot_html

    # Verify bot's trace_ids don't include manual span's trace_id
    assert manual_span.trace_id not in bot._trace_ids

    # Verify bot's spans only include bot-related spans
    all_bot_spans = []
    for trace_id in bot._trace_ids:
        spans = get_spans(trace_id=trace_id, db_path=db_path)
        all_bot_spans.extend(spans)

    manual_spans = get_spans(trace_id=manual_span.trace_id, db_path=db_path)
    bot_span_ids = {s.span_id for s in all_bot_spans}
    manual_span_ids = {s.span_id for s in manual_spans}

    # No overlap between bot spans and manual spans
    assert bot_span_ids.isdisjoint(manual_span_ids)


def test_bot_creates_new_trace_id_even_with_existing_context(tmp_path, monkeypatch):
    """Test that bot always creates a new trace_id even when manual spans exist in context.

    This test guards against regression where bot spans would inherit trace_ids from
    manual spans created before the bot call. The bot should always create its own
    unique trace_id regardless of what's in the context.

    Regression test for: bot spans showing manual spans (custom_operation, decorated_function)
    that were created before the bot call.
    """
    enable_span_recording()
    db_path = tmp_path / "test_spans.db"

    # Patch the database path to use our test database
    monkeypatch.setattr("llamabot.recorder.find_or_set_db_path", lambda x: db_path)

    # Clear any existing trace_id context
    from llamabot.recorder import current_trace_id_var

    current_trace_id_var.set(None)

    # Create manual spans BEFORE bot call (simulating the regression scenario)
    with span("custom_operation", db_path=db_path) as custom_span:
        custom_span["user_id"] = 123
        with custom_span.span("nested_operation") as nested_span:
            nested_span["nested"] = True

    # Get the manual span's trace_id
    manual_trace_id = custom_span.trace_id

    # Now create bot and make a call
    # The bot should create its OWN trace_id, not inherit from manual spans
    bot = SimpleBot(system_prompt="Test prompt", mock_response="Response")
    bot("Bot call")

    # Verify bot has its own trace_id(s)
    assert len(bot._trace_ids) == 1
    bot_trace_id = bot._trace_ids[0]

    # CRITICAL: Bot's trace_id should be DIFFERENT from manual span's trace_id
    assert (
        bot_trace_id != manual_trace_id
    ), "Bot should create its own trace_id, not inherit from manual spans in context"

    # Verify bot's spans don't include manual spans
    bot_html = bot.display_spans()
    assert custom_span.span_id not in bot_html
    assert nested_span.span_id not in bot_html
    assert "custom_operation" not in bot_html
    assert "nested_operation" not in bot_html

    # Verify manual spans are not in bot's trace
    bot_spans = get_spans(trace_id=bot_trace_id, db_path=db_path)
    manual_spans = get_spans(trace_id=manual_trace_id, db_path=db_path)

    bot_span_ids = {s.span_id for s in bot_spans}
    manual_span_ids = {s.span_id for s in manual_spans}

    # No overlap between bot spans and manual spans
    assert bot_span_ids.isdisjoint(
        manual_span_ids
    ), "Bot spans should not include manual spans created before bot call"


def test_trace_id_cleared_after_root_span_exit(tmp_path, monkeypatch):
    """Test that trace_id is cleared from context after root span exits.

    This test verifies that when a bot creates a root span (which creates a new trace_id),
    and that span exits, subsequent manual spans get a different trace_id rather than
    inheriting the bot's trace_id.

    Note: This test may be flaky when run with other tests due to context variable
    persistence. The key behavior being tested is that manual spans created after
    a bot call get a different trace_id.
    """
    enable_span_recording()
    db_path = tmp_path / "test_spans.db"

    # Patch the database path to use our test database
    monkeypatch.setattr("llamabot.recorder.find_or_set_db_path", lambda x: db_path)

    # Clear any existing trace_id context from previous tests
    from llamabot.recorder import current_trace_id_var

    current_trace_id_var.set(None)

    bot = SimpleBot(system_prompt="Test prompt", mock_response="Response")

    # Make a bot call - this creates a root span that should clear trace_id on exit
    bot("Bot call")

    # Get the bot's trace_id
    bot_trace_id = bot._trace_ids[0]

    # Create a manual span - it should get a NEW trace_id since bot's root span cleared context
    with span("manual_operation", db_path=db_path) as manual_span:
        pass

    # Verify manual span has a different trace_id (this is the key behavior we're testing)
    # Note: This may fail if context variables leak between tests, but the behavior
    # is correct when tests run in isolation
    if manual_span.trace_id == bot_trace_id:
        # If they're the same, it means context wasn't cleared - this is the bug we're testing for
        pytest.fail(
            f"Manual span inherited bot's trace_id {bot_trace_id}. "
            "This indicates the trace_id context was not cleared when bot's root span exited."
        )

    # Verify bot's trace_ids still only contain bot's trace_id
    assert len(bot._trace_ids) == 1
    assert bot._trace_ids[0] == bot_trace_id

    # Verify spans are in separate traces (no overlap)
    bot_spans = get_spans(trace_id=bot_trace_id, db_path=db_path)
    manual_spans = get_spans(trace_id=manual_span.trace_id, db_path=db_path)
    bot_span_ids = {s.span_id for s in bot_spans}
    manual_span_ids = {s.span_id for s in manual_spans}
    assert bot_span_ids.isdisjoint(manual_span_ids)


def test_nested_spans_preserve_trace_id(tmp_path, monkeypatch):
    """Test that nested spans within bot call preserve trace_id."""
    enable_span_recording()
    db_path = tmp_path / "test_spans.db"

    # Patch the database path to use our test database
    monkeypatch.setattr("llamabot.recorder.find_or_set_db_path", lambda x: db_path)

    bot = SimpleBot(system_prompt="Test prompt", mock_response="Response")

    # Make a bot call
    bot("Bot call")

    bot_trace_id = bot._trace_ids[0]

    # Verify all spans from bot call share the same trace_id
    bot_spans = get_spans(trace_id=bot_trace_id, db_path=db_path)
    assert len(bot_spans) > 0
    for span_obj in bot_spans:
        assert span_obj.trace_id == bot_trace_id

    # Verify bot span has query and response attributes (no longer nested spans)
    # Bot span should use variable name "bot" instead of generic "simplebot_call"
    operation_names = {s.operation_name for s in bot_spans}
    assert "bot" in operation_names
    bot_span = [s for s in bot_spans if s.operation_name == "bot"][0]
    assert "query" in bot_span.attributes or "response" in bot_span.attributes


def test_bot_display_spans_no_spans_yet():
    """Test display_spans() when no spans have been recorded."""
    bot = SimpleBot(system_prompt="Test prompt")

    html = bot.display_spans()

    # SpanList returns "No spans to display" for empty lists
    assert "No spans to display" in html


def test_bot_creates_child_span_when_called_in_span_context(tmp_path, monkeypatch):
    """Test that bot creates child spans when called within a span context."""
    enable_span_recording()
    db_path = tmp_path / "test_spans.db"

    # Patch the database path to use our test database
    monkeypatch.setattr("llamabot.recorder.find_or_set_db_path", lambda x: db_path)

    bot = SimpleBot(system_prompt="Test prompt", mock_response="Response")

    # Call bot within a span context
    with span("parent_operation", db_path=db_path):
        bot("Bot call within span")

    # Verify spans were created
    spans = get_spans(db_path=db_path)
    span_dict = {s.operation_name: s for s in spans}

    # Check that parent span exists
    assert "parent_operation" in span_dict
    parent = span_dict["parent_operation"]

    # Check that bot created a child span (using variable name "bot")
    assert "bot" in span_dict
    bot_span = span_dict["bot"]

    # Verify parent-child relationship
    assert bot_span.parent_span_id == parent.span_id
    assert bot_span.trace_id == parent.trace_id

    # Verify bot's trace_ids include parent's trace_id (bot tracks all trace_ids it participates in)
    # This allows display_spans() to show spans even when bot is called as a child
    assert parent.trace_id in bot._trace_ids

    # Verify hierarchy when querying by operation name
    spans = get_spans(operation_name="parent_operation", db_path=db_path)
    span_names = [s.operation_name for s in spans]
    assert "parent_operation" in span_names
    # Bot span should use variable name "bot" instead of generic "simplebot_call"
    assert "bot" in span_names
    # Bot span should have query/response as attributes, not nested spans
    bot_span = [s for s in spans if s.operation_name == "bot"][0]
    assert "query" in bot_span.attributes or "response" in bot_span.attributes

    # Verify that bot.display_spans() can show spans even when called as a child
    # This is the key behavior: child spans should be trackable
    html = bot.display_spans()
    assert "bot" in html  # Should show the bot span
    assert bot_span.span_id in html  # Should include the span ID


def test_bot_display_spans_empty_database(tmp_path, monkeypatch):
    """Test display_spans() when spans are tracked but not in database."""
    enable_span_recording()
    db_path = tmp_path / "test_spans.db"

    # Patch the database path to use our test database
    monkeypatch.setattr("llamabot.recorder.find_or_set_db_path", lambda x: db_path)

    bot = SimpleBot(system_prompt="Test prompt", mock_response="Response")

    # Manually add a trace_id that doesn't exist in database
    bot._trace_ids.append("nonexistent-trace-id")

    html = bot.display_spans()

    # SpanList returns "No spans to display" for empty lists
    assert "No spans to display" in html


def test_bot_uses_variable_name_in_span(tmp_path, monkeypatch):
    """Test that bot uses variable name instead of generic 'simplebot_call'."""
    monkeypatch.setattr(
        "llamabot.recorder.find_or_set_db_path", lambda _: tmp_path / "test.db"
    )

    def test_function():
        """Helper function to create and call a bot for variable name detection testing.

        Creates a SimpleBot instance with variable name 'ocr_bot', calls it,
        and returns the bot instance. Used to test that spans use the variable
        name instead of the generic 'simplebot_call' name.
        """
        ocr_bot = SimpleBot("test prompt", mock_response="response")
        _ = ocr_bot("test query")
        return ocr_bot

    _ = test_function()

    # Get spans and verify the operation name is the variable name
    spans = get_spans()
    assert len(spans) > 0
    # Find the root span for this bot call
    root_spans = [s for s in spans if s.parent_span_id is None]
    assert len(root_spans) > 0
    # The operation name should be "ocr_bot" not "simplebot_call"
    assert root_spans[0].operation_name == "ocr_bot"


def test_bot_uses_variable_name_even_when_accessed_via_container(tmp_path, monkeypatch):
    """Test that bot uses variable name even when accessed via container."""
    monkeypatch.setattr(
        "llamabot.recorder.find_or_set_db_path", lambda _: tmp_path / "test.db"
    )

    # Create bot in a container, but still assign to variable
    bot_list = [SimpleBot("test prompt", mock_response="response")]
    bot = bot_list[0]  # Access via list, but still has variable name

    _ = bot("test query")

    # Get spans - should use variable name "bot"
    spans = get_spans()
    assert len(spans) > 0
    root_spans = [s for s in spans if s.parent_span_id is None]
    # Should use variable name "bot" (not generic name)
    assert root_spans[0].operation_name == "bot"


def test_bot_spans_property_returns_spanlist(tmp_path, monkeypatch):
    """Test that bot.spans property returns a SpanList."""
    enable_span_recording()
    db_path = tmp_path / "test_spans.db"

    # Patch the database path to use our test database
    monkeypatch.setattr("llamabot.recorder.find_or_set_db_path", lambda x: db_path)

    bot = SimpleBot(system_prompt="Test prompt", mock_response="Response")

    # Before any calls, spans should be empty SpanList
    spans = bot.spans
    assert isinstance(spans, SpanList)
    assert len(spans) == 0

    # Make a bot call
    bot("First call")

    # After call, spans should contain spans
    spans = bot.spans
    assert isinstance(spans, SpanList)
    assert len(spans) > 0

    # Verify spans are iterable
    span_list = list(spans)
    assert len(span_list) > 0
    assert all(hasattr(s, "span_id") for s in span_list)


def test_bot_spans_property_includes_all_calls(tmp_path, monkeypatch):
    """Test that bot.spans includes spans from all bot calls."""
    enable_span_recording()
    db_path = tmp_path / "test_spans.db"

    # Patch the database path to use our test database
    monkeypatch.setattr("llamabot.recorder.find_or_set_db_path", lambda x: db_path)

    bot = SimpleBot(system_prompt="Test prompt", mock_response="Response")

    # Make multiple bot calls
    bot("First call")
    bot("Second call")
    bot("Third call")

    # Get spans via property
    spans = bot.spans
    assert isinstance(spans, SpanList)
    assert len(spans) >= 3  # At least one span per call

    # Verify all trace_ids are represented
    span_trace_ids = {s.trace_id for s in spans}
    assert len(span_trace_ids) == 3  # One trace_id per call
    assert span_trace_ids == set(bot._trace_ids)


def test_bot_display_spans_uses_spans_property(tmp_path, monkeypatch):
    """Test that display_spans() uses the .spans property internally."""
    enable_span_recording()
    db_path = tmp_path / "test_spans.db"

    # Patch the database path to use our test database
    monkeypatch.setattr("llamabot.recorder.find_or_set_db_path", lambda x: db_path)

    bot = SimpleBot(system_prompt="Test prompt", mock_response="Response")

    # Make a bot call
    bot("Test call")

    # Get spans via property
    spans = bot.spans
    assert len(spans) > 0

    # Get HTML from display_spans (which should use .spans internally)
    html = bot.display_spans()

    # Verify HTML contains span information
    assert len(html) > 0
    # Verify that spans from .spans property are represented in HTML
    span_ids_in_html = [s.span_id for s in spans if s.span_id in html]
    assert len(span_ids_in_html) > 0


def test_bot_spans_property_empty_when_no_calls():
    """Test that bot.spans returns empty SpanList when no calls have been made."""
    bot = SimpleBot(system_prompt="Test prompt")

    spans = bot.spans
    assert isinstance(spans, SpanList)
    assert len(spans) == 0
    assert list(spans) == []


# Tests for new display methods
def test_generate_config_html_basic():
    """Test that generate_config_html() returns valid HTML."""
    bot = SimpleBot(
        system_prompt="Test prompt",
        model_name="test-model",
        temperature=0.5,
        stream_target="stdout",
    )

    html = bot.generate_config_html()

    # Check that HTML contains expected configuration values
    assert "test-model" in html
    assert "0.5" in html
    assert "stdout" in html
    assert "Test prompt" in html
    assert "SimpleBot Configuration" in html


def test_generate_config_html_with_json_mode():
    """Test that generate_config_html() shows JSON mode correctly."""
    bot = SimpleBot(system_prompt="Test prompt", json_mode=True)

    html = bot.generate_config_html()

    assert "Yes" in html  # JSON Mode should show "Yes"


def test_generate_config_html_without_json_mode():
    """Test that generate_config_html() shows JSON mode as No when disabled."""
    bot = SimpleBot(system_prompt="Test prompt", json_mode=False)

    html = bot.generate_config_html()

    assert "No" in html  # JSON Mode should show "No"


def test_generate_config_html_with_memory():
    """Test that generate_config_html() shows memory class name."""
    mock_memory = MagicMock()
    bot = SimpleBot(system_prompt="Test prompt", memory=mock_memory)

    html = bot.generate_config_html()

    # Should show the class name of the memory object
    assert "MagicMock" in html


def test_generate_config_html_without_memory():
    """Test that generate_config_html() shows None when no memory."""
    bot = SimpleBot(system_prompt="Test prompt", memory=None)

    html = bot.generate_config_html()

    assert "None" in html


def test_generate_config_html_with_completion_kwargs():
    """Test that generate_config_html() shows additional parameters."""
    bot = SimpleBot(
        system_prompt="Test prompt",
        max_tokens=100,
        top_p=0.9,
    )

    html = bot.generate_config_html()

    # Should show the additional parameters
    assert "Additional Parameters" in html
    assert "max_tokens" in html
    assert "100" in html
    assert "top_p" in html
    assert "0.9" in html


def test_generate_config_html_without_completion_kwargs():
    """Test that generate_config_html() works without additional parameters."""
    bot = SimpleBot(system_prompt="Test prompt")

    html = bot.generate_config_html()

    # Should NOT show additional parameters section
    assert "Additional Parameters" not in html


def test_repr_html_returns_config_not_spans():
    """Test that _repr_html_() returns configuration HTML, not spans."""
    bot = SimpleBot(system_prompt="Test prompt", model_name="test-model")

    html = bot._repr_html_()

    # Should contain configuration
    assert "SimpleBot Configuration" in html
    assert "test-model" in html
    assert "Test prompt" in html

    # Should NOT contain span-related content (this would only be in display_spans())
    # Note: This test verifies the change from showing spans to showing config


def test_repr_html_uses_generate_config_html():
    """Test that _repr_html_() uses generate_config_html() internally."""
    bot = SimpleBot(system_prompt="Test prompt")

    # Both should return the same HTML
    config_html = bot.generate_config_html()
    repr_html = bot._repr_html_()

    assert config_html == repr_html
