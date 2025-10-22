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
        patch("llamabot.bot.agentbot.extract_content") as mock_extract_content,
        patch("litellm.completion") as mock_completion,
        patch("llamabot.bot.agentbot.stream_chunks") as mock_agentbot_stream_chunks,
    ):
        # Mock the thought phase
        thought_message = AIMessage(content="Thought: I need to respond to the user.")
        mock_make_response.return_value = thought_message
        mock_stream_chunks.return_value = thought_message
        mock_extract_content.return_value = "Thought: I need to respond to the user."

        # Mock the completion call to prevent real API calls
        mock_completion.return_value = thought_message

        # Mock stream_chunks to return the message directly
        mock_stream_chunks.return_value = thought_message
        mock_agentbot_stream_chunks.return_value = thought_message

        # Mock extract_content to return the content directly
        def mock_extract_content_func(response):
            if hasattr(response, "content"):
                return response.content
            return "Thought: I need to respond to the user."

        mock_extract_content.side_effect = mock_extract_content_func

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
        patch("llamabot.bot.agentbot.extract_content") as mock_extract_content,
        patch("litellm.completion") as mock_completion,
        patch("llamabot.bot.agentbot.stream_chunks") as mock_agentbot_stream_chunks,
    ):
        # Mock thought phase
        thought_message = AIMessage(content="Thought: I need to do something.")
        mock_make_response.return_value = thought_message
        mock_stream_chunks.return_value = thought_message
        mock_extract_content.return_value = "Thought: I need to do something."

        # Mock the completion call to prevent real API calls
        mock_completion.return_value = thought_message

        # Mock stream_chunks to return the message directly
        mock_stream_chunks.return_value = thought_message
        mock_agentbot_stream_chunks.return_value = thought_message

        # Mock extract_content to return the content directly
        def mock_extract_content_func(response):
            if hasattr(response, "content"):
                return response.content
            return "Thought: I need to do something."

        mock_extract_content.side_effect = mock_extract_content_func

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
        patch("llamabot.bot.agentbot.extract_content") as mock_extract_content,
        patch("litellm.completion") as mock_completion,
        patch("llamabot.bot.agentbot.stream_chunks") as mock_agentbot_stream_chunks,
    ):
        # Mock thought phase
        thought_message = AIMessage(content="Thought: I need to do something.")
        mock_make_response.return_value = thought_message
        mock_stream_chunks.return_value = thought_message
        mock_extract_content.return_value = "Thought: I need to do something."

        # Mock the completion call to prevent real API calls
        mock_completion.return_value = thought_message

        # Mock stream_chunks to return the message directly
        mock_stream_chunks.return_value = thought_message
        mock_agentbot_stream_chunks.return_value = thought_message

        # Mock extract_content to return the content directly
        def mock_extract_content_func(response):
            if hasattr(response, "content"):
                return response.content
            return "Thought: I need to do something."

        mock_extract_content.side_effect = mock_extract_content_func

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


@pytest.mark.asyncio
async def test_agent_bot_empty_tool_calls():
    """Test AgentBot automatically responds when no tools are selected."""
    bot = AgentBot(
        system_prompt="Test prompt",
        tools=[mock_tool],
    )

    with (
        patch("llamabot.bot.simplebot.make_response") as mock_make_response,
        patch("llamabot.bot.simplebot.stream_chunks") as mock_stream_chunks,
        patch("llamabot.bot.agentbot.extract_content") as mock_extract_content,
        patch("litellm.completion") as mock_completion,
        patch("llamabot.bot.agentbot.stream_chunks") as mock_agentbot_stream_chunks,
    ):
        # Mock thought phase
        thought_message = AIMessage(content="I need more information from the user.")
        mock_make_response.return_value = thought_message
        mock_stream_chunks.return_value = thought_message
        mock_extract_content.return_value = "I need more information from the user."

        mock_completion.return_value = thought_message
        mock_stream_chunks.return_value = thought_message
        mock_agentbot_stream_chunks.return_value = thought_message

        def mock_extract_content_func(response):
            if hasattr(response, "content"):
                return response.content
            return "I need more information from the user."

        mock_extract_content.side_effect = mock_extract_content_func

        # Mock ToolBot to return empty list
        with (
            patch("llamabot.bot.toolbot.make_response") as mock_toolbot_make_response,
            patch("llamabot.bot.toolbot.stream_chunks") as mock_toolbot_stream_chunks,
        ):
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message = MagicMock()
            mock_response.choices[0].message.tool_calls = []
            mock_toolbot_make_response.return_value = mock_response
            mock_toolbot_stream_chunks.return_value = mock_response

            result = bot("Test message")
            assert result.content == "I need more information from the user."
            assert bot.run_meta["tool_usage"]["respond_to_user"]["calls"] == 1


def test_agentbot_memory_with_string_input():
    """Test that AgentBot properly appends string user inputs to memory."""
    from llamabot.components.chat_memory import ChatMemory

    # Create a mock memory
    mock_memory = MagicMock(spec=ChatMemory)

    # Create bot with memory
    bot = AgentBot(
        name="test_bot",
        system_prompt="You are a helpful assistant.",
        tools=[mock_tool],
        memory=mock_memory,
    )

    with (
        patch("llamabot.bot.simplebot.make_response") as mock_make_response,
        patch("llamabot.bot.simplebot.stream_chunks") as mock_stream_chunks,
        patch("llamabot.bot.agentbot.extract_content") as mock_extract_content,
        patch("litellm.completion") as mock_completion,
        patch("llamabot.bot.agentbot.stream_chunks") as mock_agentbot_stream_chunks,
    ):
        # Mock thought phase
        thought_message = AIMessage(content="I need to process this.")
        mock_make_response.return_value = thought_message
        mock_stream_chunks.return_value = thought_message
        mock_extract_content.return_value = "I need to process this."
        mock_completion.return_value = thought_message
        mock_agentbot_stream_chunks.return_value = thought_message

        def mock_extract_content_func(response):
            if hasattr(response, "content"):
                return response.content
            return "I need to process this."

        mock_extract_content.side_effect = mock_extract_content_func

        # Mock ToolBot to return empty list (no tools selected)
        with (
            patch("llamabot.bot.toolbot.make_response") as mock_toolbot_make_response,
            patch("llamabot.bot.toolbot.stream_chunks") as mock_toolbot_stream_chunks,
        ):
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message = MagicMock()
            mock_response.choices[0].message.tool_calls = []
            mock_toolbot_make_response.return_value = mock_response
            mock_toolbot_stream_chunks.return_value = mock_response

            # Test with string input
            result = bot("Hello, process this string")

            # Verify that memory.append was called with the user message
            assert mock_memory.append.called
            # The first call should be the user message
            user_message = mock_memory.append.call_args_list[0][0][0]
            assert user_message.content == "Hello, process this string"
            assert user_message.role == "user"

            # Verify the result
            assert result.content == "I need to process this."


def test_agentbot_memory_with_list_input():
    """Test that AgentBot properly appends list user inputs to memory."""
    from llamabot.components.chat_memory import ChatMemory
    from llamabot.components.messages import HumanMessage

    # Create a mock memory
    mock_memory = MagicMock(spec=ChatMemory)

    # Create bot with memory
    bot = AgentBot(
        name="test_bot",
        system_prompt="You are a helpful assistant.",
        tools=[mock_tool],
        memory=mock_memory,
    )

    with (
        patch("llamabot.bot.simplebot.make_response") as mock_make_response,
        patch("llamabot.bot.simplebot.stream_chunks") as mock_stream_chunks,
        patch("llamabot.bot.agentbot.extract_content") as mock_extract_content,
        patch("litellm.completion") as mock_completion,
        patch("llamabot.bot.agentbot.stream_chunks") as mock_agentbot_stream_chunks,
    ):
        # Mock thought phase
        thought_message = AIMessage(content="I need to process this.")
        mock_make_response.return_value = thought_message
        mock_stream_chunks.return_value = thought_message
        mock_extract_content.return_value = "I need to process this."
        mock_completion.return_value = thought_message
        mock_agentbot_stream_chunks.return_value = thought_message

        def mock_extract_content_func(response):
            if hasattr(response, "content"):
                return response.content
            return "I need to process this."

        mock_extract_content.side_effect = mock_extract_content_func

        # Mock ToolBot to return empty list (no tools selected)
        with (
            patch("llamabot.bot.toolbot.make_response") as mock_toolbot_make_response,
            patch("llamabot.bot.toolbot.stream_chunks") as mock_toolbot_stream_chunks,
        ):
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message = MagicMock()
            mock_response.choices[0].message.tool_calls = []
            mock_toolbot_make_response.return_value = mock_response
            mock_toolbot_stream_chunks.return_value = mock_response

            # Test with list input containing HumanMessage
            user_msg = HumanMessage(content="Hello from list")
            result = bot([user_msg])

            # Verify that memory.append was called with the user message
            assert mock_memory.append.called
            # The first call should be the user message
            user_message = mock_memory.append.call_args_list[0][0][0]
            assert user_message.content == "Hello from list"
            assert user_message.role == "user"

            # Verify the result
            assert result.content == "I need to process this."
