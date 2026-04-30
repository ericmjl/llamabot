"""Integration tests for MCP client manager with in-process FastMCP."""

from fastmcp import FastMCP

from llamabot.mcp.manager import MCPClientManager
from llamabot.mcp.specs import MCPIntegrationOptions, MCPServerSpec, MCPStartupMode


def test_inproc_mcp_tool_invoke_roundtrip() -> None:
    """MCP tools discovered from an in-process server can be invoked synchronously."""
    mcp = FastMCP("pytest_inline")

    @mcp.tool()
    def echo(text: str) -> str:
        """Echo input unchanged.

        :param text: Message to echo.
        :return: Same text.
        """
        return text

    spec = MCPServerSpec(name="pytest_srv", transport="inproc", fastmcp=mcp)
    opts = MCPIntegrationOptions(startup_mode=MCPStartupMode.STRICT)
    mgr = MCPClientManager([spec], opts)
    try:
        mgr.start()
        tools = mgr.llamabot_tools()
        names = {t.__name__ for t in tools}
        assert "pytest_srv__echo" in names
        echo_tool = next(t for t in tools if t.__name__ == "pytest_srv__echo")
        assert echo_tool(text="hello") == "hello"
    finally:
        mgr.close()


def test_best_effort_records_failure() -> None:
    """Unreachable stdio server yields a failure entry without raising."""
    spec = MCPServerSpec(
        name="bad",
        transport="stdio",
        command="/nonexistent_binary_xyz",
        args=[],
    )
    opts = MCPIntegrationOptions(startup_mode=MCPStartupMode.BEST_EFFORT)
    mgr = MCPClientManager([spec], opts)
    try:
        mgr.start()
        assert mgr.failures
        assert mgr.llamabot_tools() == []
    finally:
        mgr.close()


def test_inproc_fastmcp_shortcut_is_coerced() -> None:
    """Passing FastMCP directly in ``mcp_servers`` is supported."""
    mcp = FastMCP("shortcut_srv")

    @mcp.tool()
    def ping() -> str:
        """Return pong.

        :return: Literal pong.
        """
        return "pong"

    mgr = MCPClientManager(
        [mcp], MCPIntegrationOptions(startup_mode=MCPStartupMode.STRICT)
    )
    try:
        mgr.start()
        tools = mgr.llamabot_tools()
        names = {tool.__name__ for tool in tools}
        assert "shortcut_srv__ping" in names
    finally:
        mgr.close()
