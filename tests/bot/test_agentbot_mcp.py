"""Smoke tests for AgentBot MCP integration."""

from types import SimpleNamespace
from unittest.mock import patch

from fastmcp import FastMCP

from llamabot.bot.agentbot import AgentBot
from llamabot.components.tools import tool
from llamabot.mcp.specs import MCPServerConfig, MCPIntegrationOptions, MCPStartupMode


@tool
def local_stub() -> str:
    """Unused locally defined tool for AgentBot validation."""
    return "stub"


@tool
def local_uppercase(text: str) -> str:
    """Uppercase text with a local Python tool.

    :param text: Input text.
    :return: Uppercased text.
    """
    return text.upper()


def test_agentbot_merges_inproc_mcp_tools() -> None:
    """AgentBot registers MCP-backed tools alongside Python tools."""
    mcp = FastMCP("agentbot_mcp_test")

    @mcp.tool()
    def ping() -> str:
        """Return pong.

        :return: Literal pong.
        """
        return "pong"

    bot = AgentBot(
        tools=[local_stub],
        mcp_servers=[MCPServerConfig(name="srv", transport="inproc", fastmcp=mcp)],
        mcp_options=MCPIntegrationOptions(startup_mode=MCPStartupMode.STRICT),
        model_name="gpt-4o-mini",
    )
    try:
        names: list[str] = []
        for t in bot.tools:
            if hasattr(t, "func"):
                names.append(t.func.__name__)
            else:
                names.append(getattr(t, "__name__", ""))
        assert "srv__ping" in names
    finally:
        bot.close_mcp()


def test_agentbot_notebook_like_mcp_flow_terminates_without_max_iterations() -> None:
    """Mirror notebook behavior: MCP + local tool calls then respond naturally.

    The first tool-selection call returns two tool calls in one model response
    (MCP greet + local uppercase). The second selection call should receive
    execution history and choose ``respond_to_user``.
    """
    mcp = FastMCP("agentbot_notebook_flow")

    @mcp.tool()
    def greet(name: str) -> str:
        """Return a short greeting.

        :param name: Who to greet.
        :return: Greeting string.
        """
        return f"Hello, {name}! (from MCP tool greet)"

    call_index = {"value": 0}
    observed_history_lengths: list[int] = []

    def select_tools(*messages, execution_history=None):
        """Mock ToolBot callable used by DecideNode.

        :param messages: Decision memory messages.
        :param execution_history: Structured prior tool execution context.
        :return: Tool-call objects with ``function.name`` and ``function.arguments``.
        """
        history = execution_history or []
        observed_history_lengths.append(len(history))
        call_index["value"] += 1

        if call_index["value"] == 1:
            return [
                SimpleNamespace(
                    function=SimpleNamespace(
                        name="srv__greet",
                        arguments='{"name":"Ada"}',
                    )
                ),
                SimpleNamespace(
                    function=SimpleNamespace(
                        name="local_uppercase",
                        arguments='{"text":"hello mcp"}',
                    )
                ),
            ]

        assert [entry["tool_name"] for entry in history[-2:]] == [
            "srv__greet",
            "local_uppercase",
        ]
        return [
            SimpleNamespace(
                function=SimpleNamespace(
                    name="respond_to_user",
                    arguments='{"response":"Done: greeted Ada and uppercased hello mcp."}',
                )
            )
        ]

    with patch("llamabot.bot.toolbot.ToolBot") as mock_toolbot_cls:
        mock_toolbot = mock_toolbot_cls.return_value
        mock_toolbot.side_effect = select_tools

        bot = AgentBot(
            tools=[local_uppercase],
            mcp_servers=[MCPServerConfig(name="srv", transport="inproc", fastmcp=mcp)],
            mcp_options=MCPIntegrationOptions(startup_mode=MCPStartupMode.STRICT),
            model_name="anthropic/claude-sonnet-4-5-20250929",
            max_iterations=8,
            system_prompt=(
                "You are a careful assistant. Prefer tools when useful. "
                "MCP tools are namespaced (for example demo__greet)."
            ),
        )

        try:
            result = bot(
                "Use MCP tool srv__greet to greet Ada, then use local_uppercase "
                "on the phrase hello mcp."
            )
        finally:
            bot.close_mcp()

    assert result == "Done: greeted Ada and uppercased hello mcp."
    assert call_index["value"] == 2
    assert observed_history_lengths == [0, 2]
    assert bot.shared["iteration_count"] == 3
