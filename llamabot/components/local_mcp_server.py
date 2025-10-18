"""Local MCP Server for wrapping local Python functions as MCP tools."""

from fastmcp import FastMCP
from typing import List, Callable


class LocalMCPServer:
    """In-process FastMCP server for local tools.

    This class wraps local Python functions as MCP tools, allowing them
    to be used through the MCP protocol alongside remote tools.

    :param name: Name for the MCP server
    """

    def __init__(self, name: str = "local"):
        self.mcp = FastMCP(name)
        self.registered_tools = {}

    def register_tool(self, func: Callable):
        """Dynamically register a local Python function as MCP tool.

        :param func: The Python function to register as an MCP tool
        :return: The registered MCP tool
        """
        tool = self.mcp.tool()(func)
        self.registered_tools[func.__name__] = tool
        return tool

    def register_tools(self, tools: List[Callable]):
        """Register multiple tools at once.

        :param tools: List of Python functions to register as MCP tools
        """
        for tool in tools:
            self.register_tool(tool)

    def get_server(self) -> FastMCP:
        """Get the FastMCP server instance.

        :return: The FastMCP server instance
        """
        return self.mcp
