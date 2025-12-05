"""Test suite for PocketFlow-based AgentBot."""

import sys
from unittest.mock import MagicMock, patch

import pytest

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
    """Test that default tools (today_date, respond_to_user, return_object_to_user, inspect_globals) are included."""
    bot = AgentBot(tools=[])

    # Should have at least the default tools
    tool_names = [tool.func.__name__ for tool in bot.tools]
    assert "today_date" in tool_names
    assert "respond_to_user" in tool_names
    assert "return_object_to_user" in tool_names
    assert "inspect_globals" in tool_names


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


def test_return_object_to_user_with_formatter():
    """Test that return_object_to_user uses formatter callback when provided."""
    from llamabot.components.tools import return_object_to_user

    # Test data
    test_data = {"my_var": {"key": "value", "number": 42}}

    # Formatter that doubles the number
    def formatter(result, globals_dict):
        """Test formatter that doubles the number in result dict."""
        if isinstance(result, dict) and "number" in result:
            result["number"] *= 2
        return result

    # Add formatter to globals_dict
    test_data["_return_object_formatter"] = formatter

    result = return_object_to_user("my_var", _globals_dict=test_data)
    assert result["number"] == 84  # Should be doubled by formatter
    assert result["key"] == "value"  # Other values unchanged


def test_return_object_to_user_formatter_receives_correct_args():
    """Test that formatter receives correct arguments (result and globals_dict)."""
    from llamabot.components.tools import return_object_to_user

    test_data = {"my_var": "test_value"}
    received_args = []

    def formatter(result, globals_dict):
        """Test formatter that records arguments and formats result."""
        received_args.append((result, globals_dict))
        return f"formatted_{result}"

    test_data["_return_object_formatter"] = formatter

    result = return_object_to_user("my_var", _globals_dict=test_data)

    assert len(received_args) == 1
    assert received_args[0][0] == "test_value"
    assert received_args[0][1] is test_data
    assert result == "formatted_test_value"


def test_return_object_to_user_formatter_fallback_on_error():
    """Test that return_object_to_user falls back to original result if formatter fails."""
    from llamabot.components.tools import return_object_to_user

    original_value = {"key": "value"}
    test_data = {"my_var": original_value}

    # Formatter that raises an exception
    def failing_formatter(result, globals_dict):
        """Test formatter that raises an exception to test error handling."""
        raise ValueError("Formatter error")

    test_data["_return_object_formatter"] = failing_formatter

    # Should return original value, not raise
    result = return_object_to_user("my_var", _globals_dict=test_data)
    assert result == original_value


def test_return_object_to_user_formatter_not_callable():
    """Test that return_object_to_user ignores non-callable formatter."""
    from llamabot.components.tools import return_object_to_user

    original_value = "test"
    test_data = {"my_var": original_value, "_return_object_formatter": "not a function"}

    # Should return original value, ignoring non-callable formatter
    result = return_object_to_user("my_var", _globals_dict=test_data)
    assert result == original_value


def test_return_object_to_user_no_formatter_backward_compatible():
    """Test that return_object_to_user works without formatter (backward compatibility)."""
    from llamabot.components.tools import return_object_to_user

    test_data = {"key": "value", "number": 42}

    # No formatter in globals_dict
    result = return_object_to_user("key", _globals_dict=test_data)
    assert result == "value"

    result = return_object_to_user("number", _globals_dict=test_data)
    assert result == 42


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


def test_decide_node_uses_system_prompt():
    """Test that DecideNode uses the provided system prompt string."""
    from unittest.mock import MagicMock, patch

    from llamabot.components.pocketflow.nodes import DecideNode

    custom_prompt = "You are a helpful assistant."

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

        decide_node = DecideNode(
            tools=[], system_prompt=custom_prompt, model_name="gpt-4.1"
        )

        prep_res = {
            "memory": ["test query"],
            "globals_dict": {},
        }

        try:
            decide_node.exec(prep_res)
        except (ValueError, AttributeError):
            # We expect this to fail, we just want to verify the prompt was used
            pass

        # Verify that ToolBot was called with the custom prompt
        mock_toolbot_class.assert_called_once()
        call_args = mock_toolbot_class.call_args
        assert call_args.kwargs["system_prompt"] == custom_prompt


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


def test_agentbot_max_iterations_terminates():
    """Test that AgentBot terminates after max_iterations is exceeded."""
    from unittest.mock import MagicMock, patch

    # Create a tool that always loops back (non-terminal)
    @nodeify(loopback_name=DECIDE_NODE_ACTION)
    @tool
    def looping_tool(message: str) -> str:
        """A tool that always loops back to decide node."""
        return f"Tool executed: {message}"

    bot = AgentBot(tools=[looping_tool], max_iterations=3)

    # Mock ToolBot to always return looping_tool
    with patch("llamabot.components.pocketflow.nodes.ToolBot") as mock_toolbot_class:
        mock_toolbot = MagicMock()
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "looping_tool"
        mock_tool_call.function.arguments = '{"message": "test"}'
        mock_toolbot.return_value = [mock_tool_call]
        mock_toolbot_class.return_value = mock_toolbot

        # Mock respond_to_user to capture when it's called
        respond_called = False

        def mock_respond(message: str) -> str:
            nonlocal respond_called
            respond_called = True
            return message

        # Find and replace respond_to_user in tools
        for i, tool_node in enumerate(bot.tools):
            if tool_node.func.__name__ == "respond_to_user":
                # Create a mock tool that replaces respond_to_user
                mock_respond_tool = MagicMock()
                mock_respond_tool.func.__name__ = "respond_to_user"
                mock_respond_tool.func = mock_respond
                mock_respond_tool.loopback_name = None
                mock_respond_tool.name = "respond_to_user"
                bot.tools[i] = mock_respond_tool
                break

        # Execute the bot
        bot("test query")

        # Verify that respond_to_user was eventually called
        assert (
            respond_called
        ), "respond_to_user should be called when max_iterations is exceeded"
        # Verify that iteration_count was tracked
        assert bot.shared.get("iteration_count", 0) > 3


def test_agentbot_max_iterations_tracks_count():
    """Test that AgentBot correctly tracks iteration_count."""
    from unittest.mock import MagicMock, patch

    @nodeify(loopback_name=DECIDE_NODE_ACTION)
    @tool
    def test_tool(message: str) -> str:
        """A test tool."""
        return f"Result: {message}"

    bot = AgentBot(tools=[test_tool], max_iterations=5)

    # Mock ToolBot to return test_tool
    with patch("llamabot.components.pocketflow.nodes.ToolBot") as mock_toolbot_class:
        mock_toolbot = MagicMock()
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = '{"message": "test"}'
        mock_toolbot.return_value = [mock_tool_call]
        mock_toolbot_class.return_value = mock_toolbot

        # Mock respond_to_user
        def mock_respond(message: str) -> str:
            return message

        # Replace respond_to_user in tools
        for i, tool_node in enumerate(bot.tools):
            if tool_node.func.__name__ == "respond_to_user":
                mock_respond_tool = MagicMock()
                mock_respond_tool.func.__name__ = "respond_to_user"
                mock_respond_tool.func = mock_respond
                mock_respond_tool.loopback_name = None
                mock_respond_tool.name = "respond_to_user"
                bot.tools[i] = mock_respond_tool
                break

        # Execute the bot
        bot("test query")

        # Verify iteration_count is tracked
        assert "iteration_count" in bot.shared
        # Should be at least 4 (1 initial + 3 tool calls before termination)
        assert bot.shared["iteration_count"] >= 4


def test_agentbot_max_iterations_none_no_limit():
    """Test that AgentBot with max_iterations=None has no limit."""

    @nodeify(loopback_name=DECIDE_NODE_ACTION)
    @tool
    def looping_tool(message: str) -> str:
        """A tool that always loops back."""
        return f"Tool: {message}"

    bot = AgentBot(tools=[looping_tool], max_iterations=None)

    # Verify max_iterations is None
    assert bot.max_iterations is None
    assert bot.decide_node.max_iterations is None


def test_agentbot_max_iterations_initializes_count():
    """Test that iteration_count is initialized in shared state."""
    from unittest.mock import MagicMock, patch

    @nodeify(loopback_name=DECIDE_NODE_ACTION)
    @tool
    def test_tool(message: str) -> str:
        """A test tool."""
        return f"Result: {message}"

    bot = AgentBot(tools=[test_tool], max_iterations=3)

    # Before calling, iteration_count should not exist
    assert "iteration_count" not in bot.shared

    # Mock ToolBot and respond_to_user
    with patch("llamabot.components.pocketflow.nodes.ToolBot") as mock_toolbot_class:
        mock_toolbot = MagicMock()
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = '{"message": "test"}'
        mock_toolbot.return_value = [mock_tool_call]
        mock_toolbot_class.return_value = mock_toolbot

        def mock_respond(message: str) -> str:
            return message

        for i, tool_node in enumerate(bot.tools):
            if tool_node.func.__name__ == "respond_to_user":
                mock_respond_tool = MagicMock()
                mock_respond_tool.func.__name__ = "respond_to_user"
                mock_respond_tool.func = mock_respond
                mock_respond_tool.loopback_name = None
                mock_respond_tool.name = "respond_to_user"
                bot.tools[i] = mock_respond_tool
                break

        # After calling, iteration_count should be initialized
        bot("test query")
        assert "iteration_count" in bot.shared
        assert isinstance(bot.shared["iteration_count"], int)
        assert bot.shared["iteration_count"] > 0


def test_agentbot_max_iterations_force_terminate_flag():
    """Test that _force_terminate flag is set when max_iterations is exceeded."""
    from unittest.mock import MagicMock, patch

    @nodeify(loopback_name=DECIDE_NODE_ACTION)
    @tool
    def looping_tool(message: str) -> str:
        """A tool that always loops back."""
        return f"Tool: {message}"

    bot = AgentBot(tools=[looping_tool], max_iterations=2)

    # Mock ToolBot
    with patch("llamabot.components.pocketflow.nodes.ToolBot") as mock_toolbot_class:
        mock_toolbot = MagicMock()
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "looping_tool"
        mock_tool_call.function.arguments = '{"message": "test"}'
        mock_toolbot.return_value = [mock_tool_call]
        mock_toolbot_class.return_value = mock_toolbot

        def mock_respond(message: str) -> str:
            return message

        # Replace respond_to_user
        for i, tool_node in enumerate(bot.tools):
            if tool_node.func.__name__ == "respond_to_user":
                mock_respond_tool = MagicMock()
                mock_respond_tool.func.__name__ = "respond_to_user"
                mock_respond_tool.func = mock_respond
                mock_respond_tool.loopback_name = None
                mock_respond_tool.name = "respond_to_user"
                bot.tools[i] = mock_respond_tool
                break

        # Execute the bot
        bot("test query")

        # Verify that iteration_count exceeded max_iterations
        assert bot.shared.get("iteration_count", 0) > bot.max_iterations
        # Verify that _force_terminate flag was set in shared state
        # The flag is set in DecideNode.prep() when max_iterations is exceeded
        assert bot.shared.get("_force_terminate", False) is True
