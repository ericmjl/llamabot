"""Test suite for the AgentBot class.

This module contains tests that verify the functionality of the AgentBot,
particularly its caching mechanism and tool execution capabilities.
"""

import pytest

from llamabot.bot.agentbot import AgentBot, hash_result
from llamabot.components.tools import tool


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
        mock_response='{"tool_name": "respond_to_user", "tool_arguments": {"message": "Done"}, "use_cached_results": {}}',
    )

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

    with pytest.raises(Exception) as exc_info:
        bot("Test message")
    assert str(exc_info.value) == "Test error"
