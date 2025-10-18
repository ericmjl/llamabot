"""Test suite for the AgentBot class.

This module contains tests that verify the functionality of the AgentBot,
particularly its ReAct pattern implementation and tool execution capabilities.
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
async def test_agent_bot_react_execution():
    """Test the AgentBot's ReAct execution flow."""
    bot = AgentBot(
        system_prompt="You are a helpful assistant. Always use the respond_to_user tool to answer questions.",
        tools=[mock_tool],
        model_name="ollama_chat/llama3.2",
    )

    # Test with a simple question that should trigger respond_to_user
    result = bot("Hello, how are you?")

    # Verify we get a response (content should not be empty)
    assert result.content is not None
    assert len(result.content) > 0

    # Verify the response is a string
    assert isinstance(result.content, str)


@pytest.mark.asyncio
async def test_agent_bot_max_iterations():
    """Test that AgentBot respects max_iterations in ReAct pattern."""
    bot = AgentBot(
        system_prompt="You are a helpful assistant. Never use the respond_to_user tool.",
        tools=[mock_tool],
        model_name="ollama_chat/llama3.2",
    )

    # Test with max_iterations=1 to ensure it hits the limit quickly
    with pytest.raises(RuntimeError) as exc_info:
        bot("Hello, how are you?", max_iterations=1)
    assert "Agent exceeded maximum ReAct cycles" in str(exc_info.value)


@pytest.mark.asyncio
async def test_agent_bot_error_handling():
    """Test AgentBot's error handling in ReAct pattern."""

    # Create a tool that will cause an error
    @tool
    def error_tool() -> str:
        """A tool that always raises an error."""
        raise ValueError("This tool always fails")

    bot = AgentBot(
        system_prompt="You are a helpful assistant. Always use the error_tool.",
        tools=[error_tool],
        model_name="ollama_chat/llama3.2",
    )

    # Test that the agent handles tool errors gracefully
    with pytest.raises(RuntimeError) as exc_info:
        bot("Hello, how are you?", max_iterations=2)
    assert "Agent exceeded maximum ReAct cycles" in str(exc_info.value)
