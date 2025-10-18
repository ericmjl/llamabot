"""Tests for LocalMCPServer component."""

from llamabot.components.local_mcp_server import LocalMCPServer
from llamabot.components.tools import add, today_date


def test_local_mcp_server_initialization():
    """Test LocalMCPServer initialization."""
    server = LocalMCPServer("test")
    assert server.mcp is not None
    assert len(server.registered_tools) == 0


def test_register_single_tool():
    """Test registering a single tool."""
    server = LocalMCPServer("test")
    tool = server.register_tool(add)

    assert len(server.registered_tools) == 1
    assert "add" in server.registered_tools
    assert tool is not None


def test_register_multiple_tools():
    """Test registering multiple tools."""
    server = LocalMCPServer("test")
    tools = [add, today_date]
    server.register_tools(tools)

    assert len(server.registered_tools) == 2
    assert "add" in server.registered_tools
    assert "today_date" in server.registered_tools


def test_get_server():
    """Test getting the FastMCP server instance."""
    server = LocalMCPServer("test")
    server.register_tool(add)

    mcp_server = server.get_server()
    assert mcp_server is not None
    assert hasattr(mcp_server, "tool")  # FastMCP server should have tool decorator
