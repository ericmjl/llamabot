"""Test suite for the AgentBot class.

This module contains tests that verify the functionality of the AgentBot,
particularly its caching mechanism and tool execution capabilities.
"""

import pytest

from llamabot.bot.agentbot import AgentBot, hash_result
from llamabot.components.tools import tool
from unittest.mock import patch, MagicMock
from llamabot.components.messages import AIMessage


@tool
def mock_tool(value: str) -> str:
    """A mock tool for testing purposes."""
    return f"Processed: {value}"


def test_hash_result():
    """Test the hash_result function produces consistent hashes."""
    # Test with simple types
    assert hash_result("test") == hash_result("test")
    assert len(hash_result("test")) == 8
    assert hash_result(123) == hash_result(123)
    assert len(hash_result(123)) == 8

    # Test with dict
    d1 = {"a": 1, "b": 2}
    d2 = {"b": 2, "a": 1}  # Same content, different order
    assert hash_result(d1) == hash_result(d2)
    assert len(hash_result(d1)) == 8


@pytest.mark.asyncio
async def test_agent_bot_execution():
    """Test the AgentBot's execution flow."""
    bot = AgentBot(
        system_prompt="Test prompt",
        tools=[mock_tool],
    )

    # Patch make_response, stream_chunks, extract_tool_calls, and extract_content
    with (
        patch("llamabot.bot.agentbot.make_response") as mock_make_response,
        patch("llamabot.bot.agentbot.stream_chunks") as mock_stream_chunks,
        patch("llamabot.bot.agentbot.extract_tool_calls") as mock_extract_tool_calls,
        patch("llamabot.bot.agentbot.extract_content") as mock_extract_content,
    ):
        tool_call = MagicMock()
        tool_call.function.name = "respond_to_user"
        tool_call.function.arguments = '{"response": "Done"}'  # Correct argument name
        ai_message = AIMessage(content="Done", tool_calls=[tool_call])
        mock_make_response.return_value = ai_message
        mock_stream_chunks.return_value = ai_message
        mock_extract_tool_calls.return_value = [tool_call]
        mock_extract_content.return_value = "Done"

        result = bot("Test message")
        assert result.content == "Done"


@pytest.mark.asyncio
async def test_agent_bot_max_iterations():
    """Test that AgentBot respects max_iterations."""
    bot = AgentBot(
        system_prompt="Test prompt",
        tools=[mock_tool],
        mock_response='{"tool_name": "mock_tool", "tool_arguments": {"value": "test"}, "use_cached_results": {}}',
    )

    with pytest.raises(RuntimeError) as exc_info:
        bot("Test message", max_iterations=2)
    assert "Agent exceeded maximum iterations" in str(exc_info.value)


@pytest.mark.asyncio
async def test_agent_bot_error_handling():
    """Test AgentBot's error handling."""
    bot = AgentBot(
        system_prompt="Test prompt",
        tools=[mock_tool],
        mock_response='{"tool_name": "return_error", "tool_arguments": {"message": "Test error"}, "use_cached_results": {}}',
    )

    with pytest.raises(RuntimeError) as exc_info:
        bot("Test message")
    assert "Agent exceeded maximum iterations" in str(exc_info.value)
