"""Test suite for the AgentBot class.

This module contains tests that verify the functionality of the AgentBot,
particularly its caching mechanism and tool execution capabilities.
"""

from datetime import datetime
from typing import Dict
import pytest

from llamabot.bot.agentbot import (
    AgentBot,
    CachedResult,
    hash_result,
    ToolToCall,
    ToolArgument,
    CachedArguments,
    agent_finish,
    return_error,
)
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


def test_cached_result_model():
    """Test the CachedResult pydantic model."""
    timestamp = datetime.now()
    cached = CachedResult(
        tool_name="test_tool",
        tool_arguments={"arg": "value"},
        result="test result",
        timestamp=timestamp,
        hash_key="abc12345",  # 8 characters
    )

    assert cached.tool_name == "test_tool"
    assert cached.tool_arguments == {"arg": "value"}
    assert cached.result == "test result"
    assert cached.timestamp == timestamp
    assert cached.hash_key == "abc12345"
    assert len(cached.hash_key) == 8


def test_tool_to_call_model():
    """Test the ToolToCall pydantic model."""
    tool = ToolToCall(
        tool_name="test_tool",
        tool_args=[
            ToolArgument(name="arg1", value="value1"),
            ToolArgument(name="arg2", value=None),
        ],
        use_cached_results=[CachedArguments(arg_name="arg2", hash_key="abc12345")],
    )

    assert tool.tool_name == "test_tool"
    assert len(tool.tool_args) == 2
    assert tool.tool_args[0].name == "arg1"
    assert tool.tool_args[0].value == "value1"
    assert tool.tool_args[1].name == "arg2"
    assert tool.tool_args[1].value is None
    assert len(tool.use_cached_results) == 1
    assert tool.use_cached_results[0].arg_name == "arg2"
    assert tool.use_cached_results[0].hash_key == "abc12345"
    assert len(tool.use_cached_results[0].hash_key) == 8


def test_agent_bot_initialization():
    """Test AgentBot initialization."""
    bot = AgentBot(
        model_name="gpt-4o", system_prompt="Test prompt", functions=[mock_tool]
    )

    assert isinstance(bot.memory, Dict)
    assert "mock_tool" in bot.tools
    assert "agent_finish" in bot.tools
    assert "return_error" in bot.tools


def test_store_result():
    """Test the _store_result method."""
    bot = AgentBot(model_name="gpt-4o", system_prompt="Test prompt")

    # Store a new result
    result = "test result"
    hash_key = bot._store_result("test_tool", result, {"arg": "value"})

    assert hash_key in bot.memory
    assert bot.memory[hash_key].result == result
    assert bot.memory[hash_key].tool_name == "test_tool"
    assert bot.memory[hash_key].tool_arguments == {"arg": "value"}

    # Store the same result again - should return same hash
    new_hash = bot._store_result("test_tool", result, {"arg": "value"})
    assert new_hash == hash_key


def test_agent_finish_tool():
    """Test the agent_finish tool."""
    result = agent_finish("Test complete")
    assert result == "Test complete"

    # Test with non-string input
    result = agent_finish({"status": "complete"})
    assert isinstance(result, str)


def test_return_error_tool():
    """Test the return_error tool."""
    with pytest.raises(Exception) as exc_info:
        return_error("Test error")
    assert str(exc_info.value) == "Test error"


@pytest.mark.asyncio
async def test_agent_bot_execution():
    """Test the AgentBot's execution flow."""
    bot = AgentBot(
        system_prompt="Test prompt",
        functions=[mock_tool],
        mock_response='{"tool_name": "agent_finish", "tool_arguments": {"message": "Done"}, "use_cached_results": {}}',
    )

    result = bot("Test message")
    assert result.content == "Done"


@pytest.mark.asyncio
async def test_agent_bot_max_iterations():
    """Test that AgentBot respects max_iterations."""
    bot = AgentBot(
        system_prompt="Test prompt",
        functions=[mock_tool],
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
        functions=[mock_tool],
        mock_response='{"tool_name": "return_error", "tool_arguments": {"message": "Test error"}, "use_cached_results": {}}',
    )

    with pytest.raises(Exception) as exc_info:
        bot("Test message")
    assert str(exc_info.value) == "Test error"


@pytest.mark.asyncio
async def test_agent_bot_cache_usage():
    """Test that AgentBot correctly uses cached results."""
    bot = AgentBot(system_prompt="Test prompt", functions=[mock_tool])

    # First, store a result in cache
    result = "cached value"
    hash_key = bot._store_result("mock_tool", result, {"value": "test"})

    # Set up mock response to use cached result
    bot.decision_bot.mock_response = (
        "{"
        f'"tool_name": "agent_finish", '
        f'"tool_arguments": {{"message": null}}, '
        f'"use_cached_results": {{"message": "{hash_key}"}}'
        "}"
    )

    response = bot("Test message")
    assert response.content == result
