"""Test suite for PocketFlow-based AgentBot."""

import sys
import pytest
from unittest.mock import MagicMock, patch

from llamabot.bot.agentbot import AgentBot
from llamabot.components.pocketflow import DECIDE_NODE_ACTION, DecideNode, nodeify
from llamabot.components.tools import tool
from llamabot.prompt_library.agentbot import decision_bot_system_prompt


@nodeify(loopback_name=DECIDE_NODE_ACTION)
@tool
def echo_function(text: str) -> str:
    """Echo back the provided text (test helper function).

    :param text: Text to echo
    :return: The echoed text
    """
    return text


@nodeify(loopback_name=DECIDE_NODE_ACTION)
@tool
def add_function(a: int, b: int) -> int:
    """Add two numbers (test helper function).

    :param a: First number
    :param b: Second number
    :return: Sum of a and b
    """
    return a + b


def test_agentbot_initialization():
    """Test AgentBot initialization with properly decorated tools."""
    bot = AgentBot(tools=[echo_function])

    assert bot is not None
    assert hasattr(bot, "tools")
    assert hasattr(bot, "decide_node")
    assert hasattr(bot, "flow")
    assert hasattr(bot, "shared")


def test_agentbot_requires_decorated_tools():
    """Test that tools must be decorated with @tool and @nodeify."""
    bot = AgentBot(tools=[echo_function])

    # Check that tools are wrapped (should be FuncNode instances)
    assert len(bot.tools) > 0
    # Default tools + provided tools
    assert (
        len(bot.tools) >= 4
    )  # today_date, respond_to_user, return_object_to_user, echo_function

    # Check that wrapped tools have the expected attributes
    for wrapped_tool in bot.tools:
        assert hasattr(wrapped_tool, "func")
        assert hasattr(wrapped_tool, "loopback_name")
        assert hasattr(wrapped_tool, "name")


def test_agentbot_rejects_unwrapped_tools():
    """Test that AgentBot raises informative error for unwrapped tools."""

    def unwrapped_function(text: str) -> str:
        """Unwrapped test function without decorators."""
        return text

    with pytest.raises(ValueError) as exc_info:
        AgentBot(tools=[unwrapped_function])

    error_message = str(exc_info.value)
    assert "not properly decorated" in error_message
    assert "unwrapped_function" in error_message
    assert "missing @tool decorator" in error_message
    assert "missing @nodeify decorator" in error_message
    assert "To fix this" in error_message


def test_agentbot_includes_default_tools():
    """Test that default tools (today_date, respond_to_user, return_object_to_user) are included."""
    bot = AgentBot(tools=[])

    # Should have at least the default tools
    tool_names = [tool.func.__name__ for tool in bot.tools]
    assert "today_date" in tool_names
    assert "respond_to_user" in tool_names
    assert "return_object_to_user" in tool_names


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


def test_return_object_to_user_is_terminal():
    """Test that return_object_to_user is marked as terminal (no loopback)."""
    bot = AgentBot(tools=[])

    # Find return_object_to_user tool
    return_tool = None
    for wrapped_tool in bot.tools:
        if wrapped_tool.func.__name__ == "return_object_to_user":
            return_tool = wrapped_tool
            break

    assert return_tool is not None
    assert return_tool.loopback_name is None


def test_return_object_to_user_returns_object():
    """Test that return_object_to_user can return objects from globals_dict."""
    from llamabot.components.tools import return_object_to_user

    test_data = {"key": "value", "number": 42}
    result = return_object_to_user("key", _globals_dict=test_data)
    assert result == "value"

    result = return_object_to_user("number", _globals_dict=test_data)
    assert result == 42


def test_return_object_to_user_raises_error_when_not_found():
    """Test that return_object_to_user raises ValueError when variable not found."""
    from llamabot.components.tools import return_object_to_user

    test_data = {"key": "value"}
    with pytest.raises(ValueError) as exc_info:
        return_object_to_user("nonexistent", _globals_dict=test_data)

    error_message = str(exc_info.value)
    assert "nonexistent" in error_message
    assert "not found" in error_message.lower()
    assert "key" in error_message  # Should list available variables


def test_return_object_to_user_with_empty_globals():
    """Test that return_object_to_user handles empty globals_dict gracefully."""
    from llamabot.components.tools import return_object_to_user

    with pytest.raises(ValueError) as exc_info:
        return_object_to_user("any_var", _globals_dict={})

    error_message = str(exc_info.value)
    assert "not found" in error_message.lower()
    assert "none" in error_message.lower() or "Available variables" in error_message


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

    # flow.run() modifies shared state in place, so we need to set up the shared dict
    def mock_run(shared):
        shared["result"] = "test result"

    mock_flow.run.side_effect = mock_run
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
    assert "globals_dict" in call_args
    assert call_args["globals_dict"] == {}


@patch("llamabot.bot.agentbot.Flow")
def test_agentbot_call_with_globals_dict(mock_flow_class):
    """Test __call__ method with globals_dict parameter."""
    # Mock the flow
    mock_flow = MagicMock()

    # flow.run() modifies shared state in place, so we need to set up the shared dict
    def mock_run(shared):
        shared["result"] = "test result"

    mock_flow.run.side_effect = mock_run
    mock_flow_class.return_value = mock_flow

    bot = AgentBot(tools=[echo_function])

    # Mock the flow attribute since it's created in __init__
    bot.flow = mock_flow

    test_globals = {"my_var": "test_value", "my_number": 42}
    result = bot("test query", globals_dict=test_globals)

    assert result == "test result"
    assert mock_flow.run.called
    # Check that shared state includes globals_dict
    call_args = mock_flow.run.call_args[0][0]
    assert "memory" in call_args
    assert call_args["memory"] == ["test query"]
    assert "globals_dict" in call_args
    assert call_args["globals_dict"] == test_globals


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


def test_decision_bot_system_prompt_with_globals_dict():
    """Test that decision_bot_system_prompt accepts globals_dict and includes variable info."""
    from llamabot.components.messages import BaseMessage

    try:
        import pandas as pd
    except ImportError:
        pytest.skip("pandas not installed")

    test_globals = {
        "my_df": pd.DataFrame({"x": [1, 2, 3]}),
        "my_func": lambda x: x,
        "my_var": "test",
    }

    prompt_message = decision_bot_system_prompt(globals_dict=test_globals)
    assert isinstance(prompt_message, BaseMessage)
    assert prompt_message.role == "system"
    content = prompt_message.content
    assert isinstance(content, str)
    # Check that variable information is included
    assert "my_df" in content or "DataFrame" in content
    assert "Variable Name Matching" in content or "return_object_to_user" in content


def test_decision_bot_system_prompt_with_polars_dataframe():
    """Test that decision_bot_system_prompt works with Polars DataFrames without triggering __getitem__."""
    from llamabot.components.messages import BaseMessage

    try:
        import polars as pl
    except ImportError:
        pytest.skip("polars not installed")

    test_globals = {
        "my_polars_df": pl.DataFrame({"x": [1, 2, 3]}),
        "my_var": "test",
    }

    # This should not raise ColumnNotFoundError
    prompt_message = decision_bot_system_prompt(globals_dict=test_globals)
    assert isinstance(prompt_message, BaseMessage)
    assert prompt_message.role == "system"
    content = prompt_message.content
    assert isinstance(content, str)
    # Check that variable information is included
    assert "my_polars_df" in content or "DataFrame" in content


def test_decide_node_passes_globals_dict():
    """Test that DecideNode passes globals_dict to decision_bot_system_prompt."""
    from llamabot.components.pocketflow.nodes import DecideNode
    from unittest.mock import patch, MagicMock

    test_globals = {"my_var": "test_value"}

    with patch("llamabot.bot.toolbot.ToolBot") as mock_toolbot_class:
        mock_toolbot = MagicMock()
        mock_toolbot.return_value = [
            MagicMock(
                function=MagicMock(
                    name="respond_to_user", arguments='{"response": "test"}'
                )
            )
        ]
        mock_toolbot_class.return_value = mock_toolbot

        decide_node = DecideNode(tools=[], model_name="gpt-4.1")

        prep_res = {
            "memory": ["test query"],
            "globals_dict": test_globals,
        }

        with patch(
            "llamabot.prompt_library.agentbot.decision_bot_system_prompt"
        ) as mock_prompt:
            mock_prompt.return_value = MagicMock()
            try:
                decide_node.exec(prep_res)
            except (ValueError, AttributeError):
                # We expect this to fail, we just want to verify the prompt was called with globals_dict
                pass

            # Verify that decision_bot_system_prompt was called with globals_dict
            mock_prompt.assert_called_once()
            call_args = mock_prompt.call_args
            assert "globals_dict" in call_args.kwargs
            assert call_args.kwargs["globals_dict"] == test_globals
            # categorized_vars should be computed automatically by the prompt manager
            assert "categorized_vars" in call_args.kwargs


def test_agentbot_with_multiple_tools():
    """Test AgentBot with multiple user-provided tools."""
    bot = AgentBot(tools=[echo_function, add_function])

    # Should have default tools + 2 user tools
    tool_names = [tool.func.__name__ for tool in bot.tools]
    assert "echo_function" in tool_names
    assert "add_function" in tool_names
    assert "today_date" in tool_names
    assert "respond_to_user" in tool_names
    assert "return_object_to_user" in tool_names


def test_agentbot_memory_preserved():
    """Test that memory is preserved across calls."""
    bot = AgentBot(tools=[echo_function])

    with patch.object(bot, "flow") as mock_flow:

        def mock_run1(shared):
            shared["result"] = "result1"

        mock_flow.run.side_effect = mock_run1
        bot("query1")

        # Check that memory contains the first query
        call_args1 = mock_flow.run.call_args[0][0]
        assert call_args1["memory"] == ["query1"]
        assert call_args1["globals_dict"] == {}

        def mock_run2(shared):
            shared["result"] = "result2"

        mock_flow.run.side_effect = mock_run2
        bot("query2", globals_dict={"var": "value"})

        # Check that memory is preserved and contains both queries
        call_args2 = mock_flow.run.call_args[0][0]
        assert call_args2["memory"] == ["query1", "query2"]
        assert call_args2["globals_dict"] == {"var": "value"}


def test_agentbot_globals_dict_preserved():
    """Test that globals_dict is preserved if not provided."""
    bot = AgentBot(tools=[echo_function])

    with patch.object(bot, "flow") as mock_flow:

        def mock_run1(shared):
            shared["result"] = "result1"

        mock_flow.run.side_effect = mock_run1
        bot("query1", globals_dict={"var1": "value1"})

        # Check initial globals_dict
        call_args1 = mock_flow.run.call_args[0][0]
        assert call_args1["globals_dict"] == {"var1": "value1"}

        def mock_run2(shared):
            shared["result"] = "result2"

        mock_flow.run.side_effect = mock_run2
        bot("query2")  # No globals_dict provided

        # Check that globals_dict is preserved
        call_args2 = mock_flow.run.call_args[0][0]
        assert call_args2["globals_dict"] == {"var1": "value1"}
