"""Tests for ToolBot."""

import pytest
from unittest.mock import Mock, patch

from llamabot.bot.toolbot import ToolBot
from llamabot.components.tools import write_and_execute_code


def test_write_and_execute_code_syntax_error():
    """Test that write_and_execute_code returns error for invalid syntax."""
    # Test with invalid Python syntax
    invalid_code = "def test_function():\n    print('hello'\n    return 42"  # Missing closing parenthesis

    # Get the wrapper function
    globals_dict = {}
    wrapper = write_and_execute_code(globals_dict)

    # Should return error dictionary, not raise exception
    result = wrapper(invalid_code, {})
    assert "error" in result
    assert "Syntax error" in result["error"] or "SyntaxError" in result["error"]
    assert result["result"] is None


def test_write_and_execute_code_no_function_found():
    """Test that write_and_execute_code returns error for missing function definitions."""
    # Test with code that has no function definition
    code_without_function = "print('hello world')\nx = 42"

    # Get the wrapper function
    globals_dict = {}
    wrapper = write_and_execute_code(globals_dict)

    # Should return error dictionary, not raise exception
    result = wrapper(code_without_function, {})
    assert "error" in result
    assert "No function definition found" in result["error"]
    assert result["result"] is None


def test_write_and_execute_code_successful_execution():
    """Test that write_and_execute_code executes valid code successfully."""
    valid_code = """
def test_function():
    return "Hello, World!"
"""

    # Get the wrapper function
    globals_dict = {}
    wrapper = write_and_execute_code(globals_dict)
    result = wrapper(valid_code, {})

    assert result["result"] == "Hello, World!"
    assert result["code"] == valid_code


def test_write_and_execute_code_accesses_globals():
    """Test that write_and_execute_code can access variables from the current globals."""
    # Create a function that references a built-in global
    code_with_global = """
def test_function():
    return len([1, 2, 3])
"""

    # Get the wrapper function
    globals_dict = {}
    wrapper = write_and_execute_code(globals_dict)
    result = wrapper(code_with_global, {})

    assert result["result"] == 3
    assert result["code"] == code_with_global


def test_toolbot_initialization():
    """Test that ToolBot initializes correctly."""
    system_prompt = "You are a helpful assistant."
    bot = ToolBot(
        system_prompt=system_prompt,
        model_name="gpt-4.1",
        tools=[write_and_execute_code(globals_dict={})],
    )

    # The system prompt should be converted to a SystemMessage
    from llamabot.components.messages import SystemMessage

    assert isinstance(bot.system_prompt, SystemMessage)
    assert bot.system_prompt.content == system_prompt
    assert bot.model_name == "gpt-4.1"
    assert (
        len(bot.tools) == 4
    )  # today_date, respond_to_user, return_object_to_user, write_and_execute_code_wrapper
    assert "write_and_execute_code_wrapper" in bot.name_to_tool_map
    assert bot.chat_memory is not None


def test_toolbot_without_tools():
    """Test that ToolBot works without additional tools."""
    system_prompt = "You are a helpful assistant."
    bot = ToolBot(
        system_prompt=system_prompt,
        model_name="gpt-4.1",
    )

    assert len(bot.tools) == 3  # today_date, respond_to_user, return_object_to_user
    assert "today_date" in bot.name_to_tool_map
    assert "respond_to_user" in bot.name_to_tool_map
    assert "return_object_to_user" in bot.name_to_tool_map


def test_toolbot_with_custom_chat_memory():
    """Test that ToolBot can use custom chat memory."""
    from llamabot.components.chat_memory import ChatMemory

    system_prompt = "You are a helpful assistant."
    custom_memory = ChatMemory()
    bot = ToolBot(
        system_prompt=system_prompt,
        model_name="gpt-4.1",
        chat_memory=custom_memory,
    )

    assert bot.chat_memory is custom_memory


@patch("llamabot.bot.toolbot.make_response")
@patch("llamabot.bot.toolbot.stream_chunks")
@patch("llamabot.bot.toolbot.extract_tool_calls")
def test_toolbot_call(
    mock_extract_tool_calls,
    mock_stream_chunks,
    mock_make_response,
):
    """Test that ToolBot.__call__ works correctly."""
    # Mock the response chain
    mock_response = Mock()
    mock_make_response.return_value = mock_response
    mock_stream_chunks.return_value = mock_response
    mock_extract_tool_calls.return_value = []

    system_prompt = "You are a helpful assistant."
    bot = ToolBot(
        system_prompt=system_prompt,
        model_name="gpt-4.1",
    )

    # Test calling the bot
    result = bot("Hello")

    # Verify the call chain
    mock_make_response.assert_called_once()
    mock_stream_chunks.assert_called_once()
    mock_extract_tool_calls.assert_called_once()

    assert result == []


@patch("llamabot.bot.toolbot.make_response")
@patch("llamabot.bot.toolbot.stream_chunks")
@patch("llamabot.bot.toolbot.extract_tool_calls")
def test_toolbot_call_with_callable_function(
    mock_extract_tool_calls,
    mock_stream_chunks,
    mock_make_response,
):
    """Test that ToolBot.__call__ works correctly with a callable function."""
    # Mock the response chain
    mock_response = Mock()
    mock_make_response.return_value = mock_response
    mock_stream_chunks.return_value = mock_response
    mock_extract_tool_calls.return_value = []

    system_prompt = "You are a helpful assistant."
    bot = ToolBot(
        system_prompt=system_prompt,
        model_name="gpt-4.1",
    )

    # Create a callable function that returns a string
    def get_message():
        """Get a message from a callable function."""
        return "Hello from callable function"

    # Test calling the bot with a callable
    result = bot(get_message)

    # Verify the call chain
    mock_make_response.assert_called_once()
    mock_stream_chunks.assert_called_once()
    mock_extract_tool_calls.assert_called_once()

    assert result == []


def test_toolbot_call_with_invalid_callable():
    """Test that ToolBot.__call__ raises error for callable that doesn't return string."""
    system_prompt = "You are a helpful assistant."
    bot = ToolBot(
        system_prompt=system_prompt,
        model_name="gpt-4.1",
    )

    # Create a callable function that returns a non-string
    def get_message():
        """Get a message from a callable function."""
        return 42  # Returns int, not string

    # Test that calling the bot with invalid callable raises ValueError
    with pytest.raises(ValueError, match="Callable function must return a string"):
        bot(get_message)


def test_toolbot_call_with_lambda():
    """Test that ToolBot.__call__ works correctly with a lambda function."""
    from unittest.mock import patch

    with (
        patch("llamabot.bot.toolbot.make_response") as mock_make_response,
        patch("llamabot.bot.toolbot.stream_chunks") as mock_stream_chunks,
        patch("llamabot.bot.toolbot.extract_tool_calls") as mock_extract_tool_calls,
    ):
        # Mock the response chain
        mock_response = Mock()
        mock_make_response.return_value = mock_response
        mock_stream_chunks.return_value = mock_response
        mock_extract_tool_calls.return_value = []

        system_prompt = "You are a helpful assistant."
        bot = ToolBot(
            system_prompt=system_prompt,
            model_name="gpt-4.1",
        )

        # Test calling the bot with a lambda
        result = bot(lambda: "Hello from lambda")

        # Verify the call chain
        mock_make_response.assert_called_once()
        mock_stream_chunks.assert_called_once()
        mock_extract_tool_calls.assert_called_once()

        assert result == []
