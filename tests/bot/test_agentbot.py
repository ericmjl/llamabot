"""Test suite for the AgentBot class.

This module contains tests that verify the functionality of the AgentBot,
particularly its ReAct pattern implementation and tool execution capabilities.
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
async def test_agent_bot_react_execution():
    """Test the AgentBot's ReAct execution flow."""
    bot = AgentBot(
        system_prompt="Test prompt",
        tools=[mock_tool],
    )

    # Mock the ToolBot to return respond_to_user tool call
    with (
        patch.object(bot.toolbot, "__call__") as mock_toolbot_call,
        patch("llamabot.bot.agentbot.make_response") as mock_make_response,
        patch("llamabot.bot.agentbot.stream_chunks") as mock_stream_chunks,
        patch("llamabot.bot.agentbot.extract_content") as mock_extract_content,
    ):
        # Mock the thought phase
        thought_message = AIMessage(content="Thought: I need to respond to the user.")
        mock_make_response.return_value = thought_message
        mock_stream_chunks.return_value = thought_message
        mock_extract_content.return_value = "Thought: I need to respond to the user."

        # Mock ToolBot to return respond_to_user tool call
        tool_call = MagicMock()
        tool_call.function.name = "respond_to_user"
        tool_call.function.arguments = '{"response": "Done"}'
        mock_toolbot_call.return_value = [tool_call]

        result = bot("Test message")
        assert result.content == "Done"


@pytest.mark.asyncio
async def test_agent_bot_max_iterations():
    """Test that AgentBot respects max_iterations in ReAct pattern."""
    bot = AgentBot(
        system_prompt="Test prompt",
        tools=[mock_tool],
    )

    # Mock ToolBot to never return respond_to_user (causing max iterations)
    with (
        patch.object(bot.toolbot, "__call__") as mock_toolbot_call,
        patch("llamabot.bot.agentbot.make_response") as mock_make_response,
        patch("llamabot.bot.agentbot.stream_chunks") as mock_stream_chunks,
        patch("llamabot.bot.agentbot.extract_content") as mock_extract_content,
    ):
        # Mock thought phase
        thought_message = AIMessage(content="Thought: I need to do something.")
        mock_make_response.return_value = thought_message
        mock_stream_chunks.return_value = thought_message
        mock_extract_content.return_value = "Thought: I need to do something."

        # Mock ToolBot to return a non-respond_to_user tool call
        tool_call = MagicMock()
        tool_call.function.name = "mock_tool"
        tool_call.function.arguments = '{"value": "test"}'
        mock_toolbot_call.return_value = [tool_call]

        with pytest.raises(RuntimeError) as exc_info:
            bot("Test message", max_iterations=2)
        assert "Agent exceeded maximum ReAct cycles" in str(exc_info.value)


@pytest.mark.asyncio
async def test_agent_bot_error_handling():
    """Test AgentBot's error handling in ReAct pattern."""
    bot = AgentBot(
        system_prompt="Test prompt",
        tools=[mock_tool],
    )

    # Mock ToolBot to return a tool that will cause an error
    with (
        patch.object(bot.toolbot, "__call__") as mock_toolbot_call,
        patch("llamabot.bot.agentbot.make_response") as mock_make_response,
        patch("llamabot.bot.agentbot.stream_chunks") as mock_stream_chunks,
        patch("llamabot.bot.agentbot.extract_content") as mock_extract_content,
    ):
        # Mock thought phase
        thought_message = AIMessage(content="Thought: I need to do something.")
        mock_make_response.return_value = thought_message
        mock_stream_chunks.return_value = thought_message
        mock_extract_content.return_value = "Thought: I need to do something."

        # Mock ToolBot to return a tool call that will fail
        tool_call = MagicMock()
        tool_call.function.name = "nonexistent_tool"
        tool_call.function.arguments = '{"value": "test"}'
        mock_toolbot_call.return_value = [tool_call]

        with pytest.raises(RuntimeError) as exc_info:
            bot("Test message", max_iterations=2)
        assert "Agent exceeded maximum ReAct cycles" in str(exc_info.value)
