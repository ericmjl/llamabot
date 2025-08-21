"""Tests for ToolBot."""

from unittest.mock import Mock, patch

from llamabot.bot.toolbot import ToolBot
from llamabot.components.tools import write_and_execute_code


def test_write_and_execute_code_syntax_error():
    """Test that write_and_execute_code handles syntax errors properly."""
    # Test with invalid Python syntax
    invalid_code = "def test_function():\n    print('hello'\n    return 42"  # Missing closing parenthesis

    result = write_and_execute_code(invalid_code, {})

    assert "Syntax error" in result
    assert "provided code" in result


def test_write_and_execute_code_no_function_found():
    """Test that write_and_execute_code handles missing function definitions."""
    # Test with code that has no function definition
    code_without_function = "print('hello world')\nx = 42"

    result = write_and_execute_code(code_without_function, {})

    assert "Code validation error" in result
    assert "No function definition found" in result


def test_write_and_execute_code_successful_execution():
    """Test that write_and_execute_code executes valid code successfully."""
    valid_code = """
def test_function():
    return "Hello, World!"
"""

    result = write_and_execute_code(valid_code, {})

    assert result == "Hello, World!"


def test_toolbot_initialization():
    """Test that ToolBot initializes correctly."""
    system_prompt = "You are a helpful assistant."
    bot = ToolBot(
        system_prompt=system_prompt,
        model_name="gpt-4",
        tools=[write_and_execute_code],
    )

    # The system prompt should be converted to a SystemMessage
    from llamabot.components.messages import SystemMessage

    assert isinstance(bot.system_prompt, SystemMessage)
    assert bot.system_prompt.content == system_prompt
    assert bot.model_name == "gpt-4"
    assert len(bot.tools) == 3  # today_date, respond_to_user, write_and_execute_code
    assert "write_and_execute_code" in bot.name_to_tool_map
    assert bot.chat_memory is not None


def test_toolbot_without_tools():
    """Test that ToolBot works without additional tools."""
    system_prompt = "You are a helpful assistant."
    bot = ToolBot(
        system_prompt=system_prompt,
        model_name="gpt-4",
    )

    assert len(bot.tools) == 2  # today_date, respond_to_user
    assert "today_date" in bot.name_to_tool_map
    assert "respond_to_user" in bot.name_to_tool_map


def test_toolbot_with_custom_chat_memory():
    """Test that ToolBot can use custom chat memory."""
    from llamabot.components.chat_memory import ChatMemory

    system_prompt = "You are a helpful assistant."
    custom_memory = ChatMemory()
    bot = ToolBot(
        system_prompt=system_prompt,
        model_name="gpt-4",
        chat_memory=custom_memory,
    )

    assert bot.chat_memory is custom_memory


@patch("llamabot.bot.toolbot.make_response")
@patch("llamabot.bot.toolbot.stream_chunks")
@patch("llamabot.bot.toolbot.extract_tool_calls")
@patch("llamabot.bot.toolbot.extract_content")
def test_toolbot_call(
    mock_extract_content,
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
    mock_extract_content.return_value = "Test response"

    system_prompt = "You are a helpful assistant."
    bot = ToolBot(
        system_prompt=system_prompt,
        model_name="gpt-4",
    )

    # Test calling the bot
    result = bot("Hello")

    # Verify the call chain
    mock_make_response.assert_called_once()
    mock_stream_chunks.assert_called_once()
    mock_extract_tool_calls.assert_called_once()
    mock_extract_content.assert_called_once()

    assert result == []
