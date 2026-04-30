"""Smoke tests for AgentBot MCP integration."""

from fastmcp import FastMCP

from llamabot.bot.agentbot import AgentBot
from llamabot.components.tools import tool
from llamabot.mcp.specs import MCPServerSpec, MCPIntegrationOptions, MCPStartupMode


@tool
def local_stub() -> str:
    """Unused locally defined tool for AgentBot validation."""
    return "stub"


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
        mcp_servers=[MCPServerSpec(name="srv", transport="inproc", fastmcp=mcp)],
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
