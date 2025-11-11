"""Test suite for PocketFlow-based AgentBot."""

import sys
import pytest
from unittest.mock import MagicMock, patch

from llamabot.bot.agentbot import AgentBot
from llamabot.components.pocketflow import DecideNode
from llamabot.prompt_library.agentbot import decision_bot_system_prompt


def echo_function(text: str) -> str:
    """Echo back the provided text (test helper function).

    :param text: Text to echo
    :return: The echoed text
    """
    return text


def add_function(a: int, b: int) -> int:
    """Add two numbers (test helper function).

    :param a: First number
    :param b: Second number
    :return: Sum of a and b
    """
    return a + b


def test_agentbot_initialization():
    """Test AgentBot initialization with plain callables."""
    bot = AgentBot(tools=[echo_function])

    assert bot is not None
    assert hasattr(bot, "tools")
    assert hasattr(bot, "decide_node")
    assert hasattr(bot, "flow")
    assert hasattr(bot, "shared")


def test_agentbot_wraps_tools():
    """Test that tools are automatically wrapped with @tool and @nodeify."""
    bot = AgentBot(tools=[echo_function])

    # Check that tools are wrapped (should be FuncNode instances)
    assert len(bot.tools) > 0
    # Default tools + provided tools
    assert len(bot.tools) >= 3  # today_date, respond_to_user, echo_function

    # Check that wrapped tools have the expected attributes
    for wrapped_tool in bot.tools:
        assert hasattr(wrapped_tool, "func")
        assert hasattr(wrapped_tool, "loopback_name")
        assert hasattr(wrapped_tool, "name")


def test_agentbot_includes_default_tools():
    """Test that default tools (today_date, respond_to_user) are included."""
    bot = AgentBot(tools=[])

    # Should have at least the default tools
    tool_names = [tool.func.__name__ for tool in bot.tools]
    assert "today_date" in tool_names
    assert "respond_to_user" in tool_names


def test_respond_to_user_is_terminal():
    """Test that respond_to_user is marked as terminal (no loopback)."""
    bot = AgentBot(tools=[])

    # Find respond_to_user tool
    respond_tool = None
    for wrapped_tool in bot.tools:
        if wrapped_tool.func.__name__ == "respond_to_user":
            respond_tool = wrapped_tool
            break

    assert respond_tool is not None
    assert respond_tool.loopback_name is None


def test_agentbot_custom_decide_node():
    """Test custom decide_node parameter."""
    custom_decide = MagicMock(spec=DecideNode)
    bot = AgentBot(tools=[echo_function], decide_node=custom_decide)

    assert bot.decide_node is custom_decide


def test_agentbot_default_decide_node():
    """Test that default DecideNode is created when not provided."""
    bot = AgentBot(tools=[echo_function])

    assert isinstance(bot.decide_node, DecideNode)
    assert bot.decide_node.tools == bot.tools


@patch("llamabot.bot.agentbot.Flow")
def test_agentbot_call_execution(mock_flow_class):
    """Test __call__ method execution."""
    # Mock the flow
    mock_flow = MagicMock()
    mock_flow.run.return_value = "test result"
    mock_flow_class.return_value = mock_flow

    bot = AgentBot(tools=[echo_function])

    # Mock the flow attribute since it's created in __init__
    bot.flow = mock_flow

    result = bot("test query")

    assert result == "test result"
    assert mock_flow.run.called
    # Check that shared state was set up correctly
    call_args = mock_flow.run.call_args[0][0]
    assert "memory" in call_args
    assert call_args["memory"] == ["test query"]


@patch("llamabot.components.pocketflow.flow_to_mermaid")
def test_agentbot_display_with_marimo(mock_flow_to_mermaid):
    """Test _display_ method when marimo is available."""
    bot = AgentBot(tools=[echo_function])
    mock_flow_to_mermaid.return_value = "graph TD\nA[Test]"

    # Mock marimo module
    mock_mo = MagicMock()
    mock_mo.mermaid.return_value = "mermaid_diagram"

    with patch.dict("sys.modules", {"marimo": mock_mo}):
        result = bot._display_()

    assert result == "mermaid_diagram"
    mock_flow_to_mermaid.assert_called_once_with(bot.flow)
    mock_mo.mermaid.assert_called_once_with("graph TD\nA[Test]")


def test_agentbot_display_without_marimo():
    """Test _display_ method raises ImportError when marimo is not available."""
    bot = AgentBot(tools=[echo_function])

    # Remove marimo from sys.modules if it exists

    marimo_backup = sys.modules.pop("marimo", None)

    try:
        # Mock import to raise ImportError for marimo
        original_import = __import__

        def mock_import(name, *args, **kwargs):
            if name == "marimo":
                raise ImportError("No module named 'marimo'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError) as exc_info:
                bot._display_()

            assert "marimo is required" in str(exc_info.value)
            assert "install marimo" in str(exc_info.value).lower()
    finally:
        # Restore marimo if it was there
        if marimo_backup is not None:
            sys.modules["marimo"] = marimo_backup


def test_decision_bot_system_prompt():
    """Test that decision_bot_system_prompt is callable and returns BaseMessage."""
    from llamabot.components.messages import BaseMessage

    prompt_message = decision_bot_system_prompt()
    assert isinstance(prompt_message, BaseMessage)
    assert prompt_message.role == "system"
    assert isinstance(prompt_message.content, str)


def test_agentbot_with_multiple_tools():
    """Test AgentBot with multiple user-provided tools."""
    bot = AgentBot(tools=[echo_function, add_function])

    # Should have default tools + 2 user tools
    tool_names = [tool.func.__name__ for tool in bot.tools]
    assert "echo_function" in tool_names
    assert "add_function" in tool_names
    assert "today_date" in tool_names
    assert "respond_to_user" in tool_names


def test_agentbot_shared_state_reset():
    """Test that shared state is reset on each call."""
    bot = AgentBot(tools=[echo_function])

    with patch.object(bot, "flow") as mock_flow:
        mock_flow.run.return_value = "result1"
        bot("query1")

        # Check that shared state was reset
        call_args1 = mock_flow.run.call_args[0][0]
        assert call_args1["memory"] == ["query1"]

        mock_flow.run.return_value = "result2"
        bot("query2")

        # Check that shared state was reset again
        call_args2 = mock_flow.run.call_args[0][0]
        assert call_args2["memory"] == ["query2"]
