"""Test suite for the AgentBot class.

This module contains tests that verify the functionality of the AgentBot,
particularly its ReAct pattern implementation and tool execution capabilities.
"""

import pytest
from unittest.mock import patch, MagicMock

from llamabot.bot.agentbot import AgentBot, hash_result
from llamabot.components.tools import tool
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

    # Mock the LLM calls at the deepest level
    with (
        patch("llamabot.bot.simplebot.make_response") as mock_make_response,
        patch("llamabot.bot.simplebot.stream_chunks") as mock_stream_chunks,
        patch("llamabot.bot.simplebot.extract_content") as mock_extract_content,
    ):
        # Mock the thought phase
        thought_message = AIMessage(content="Thought: I need to respond to the user.")
        mock_make_response.return_value = thought_message
        mock_stream_chunks.return_value = thought_message
        mock_extract_content.return_value = "Thought: I need to respond to the user."

        # Mock ToolBot's internal LLM calls
        with (
            patch("llamabot.bot.toolbot.make_response") as mock_toolbot_make_response,
            patch("llamabot.bot.toolbot.stream_chunks") as mock_toolbot_stream_chunks,
        ):
            # Create a mock tool call response
            tool_call = MagicMock()
            tool_call.function.name = "respond_to_user"
            tool_call.function.arguments = '{"response": "Done"}'

            # Mock the ToolBot's LLM response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message = MagicMock()
            mock_response.choices[0].message.tool_calls = [tool_call]
            mock_toolbot_make_response.return_value = mock_response
            mock_toolbot_stream_chunks.return_value = mock_response

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
        patch("llamabot.bot.simplebot.make_response") as mock_make_response,
        patch("llamabot.bot.simplebot.stream_chunks") as mock_stream_chunks,
        patch("llamabot.bot.simplebot.extract_content") as mock_extract_content,
    ):
        # Mock thought phase
        thought_message = AIMessage(content="Thought: I need to do something.")
        mock_make_response.return_value = thought_message
        mock_stream_chunks.return_value = thought_message
        mock_extract_content.return_value = "Thought: I need to do something."

        # Mock ToolBot to return a non-respond_to_user tool call
        with (
            patch("llamabot.bot.toolbot.make_response") as mock_toolbot_make_response,
            patch("llamabot.bot.toolbot.stream_chunks") as mock_toolbot_stream_chunks,
        ):
            tool_call = MagicMock()
            tool_call.function.name = "mock_tool"
            tool_call.function.arguments = '{"value": "test"}'

            # Mock the ToolBot's LLM response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message = MagicMock()
            mock_response.choices[0].message.tool_calls = [tool_call]
            mock_toolbot_make_response.return_value = mock_response
            mock_toolbot_stream_chunks.return_value = mock_response

            with pytest.raises(RuntimeError) as exc_info:
                bot("Test message", max_iterations=2)
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
        system_prompt="Test prompt",
        tools=[error_tool],
    )

    # Mock ToolBot to return the error tool
    with (
        patch("llamabot.bot.simplebot.make_response") as mock_make_response,
        patch("llamabot.bot.simplebot.stream_chunks") as mock_stream_chunks,
        patch("llamabot.bot.simplebot.extract_content") as mock_extract_content,
    ):
        # Mock thought phase
        thought_message = AIMessage(content="Thought: I need to do something.")
        mock_make_response.return_value = thought_message
        mock_stream_chunks.return_value = thought_message
        mock_extract_content.return_value = "Thought: I need to do something."

        # Mock ToolBot to return the error tool call
        with (
            patch("llamabot.bot.toolbot.make_response") as mock_toolbot_make_response,
            patch("llamabot.bot.toolbot.stream_chunks") as mock_toolbot_stream_chunks,
        ):
            tool_call = MagicMock()
            tool_call.function.name = "error_tool"
            tool_call.function.arguments = "{}"

            # Mock the ToolBot's LLM response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message = MagicMock()
            mock_response.choices[0].message.tool_calls = [tool_call]
            mock_toolbot_make_response.return_value = mock_response
            mock_toolbot_stream_chunks.return_value = mock_response

            with pytest.raises(RuntimeError) as exc_info:
                bot("Test message", max_iterations=2)
            assert "Agent exceeded maximum ReAct cycles" in str(exc_info.value)
