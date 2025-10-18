"""MCP Tools adapter for converting MCP tools to llamabot tool format.

This module provides functional adapters that convert MCP (Model Context Protocol)
tools into the format expected by llamabot's tool system.
"""

from typing import Any, Callable, Dict, List
from loguru import logger

from llamabot.components.tools import tool
from llamabot.components.mcp_client import MCPConnectionManager, run_async


def mcp_tool_to_llamabot_tool(
    mcp_tool: Dict[str, Any], server_name: str, connection_manager: MCPConnectionManager
) -> Callable:
    """Convert a single MCP tool to a llamabot-compatible tool.

    :param mcp_tool: MCP tool definition
    :param server_name: Name of the MCP server
    :param connection_manager: MCP connection manager instance
    :return: Callable tool function
    """
    tool_name = mcp_tool.get("name", "unknown_tool")
    tool_description = mcp_tool.get("description", "MCP tool")
    tool_input_schema = mcp_tool.get("inputSchema", {})

    # Create a unique name for the tool
    prefixed_name = f"{server_name}:{tool_name}"

    # Generate docstring from MCP tool description
    docstring = f"""{tool_description}

    This is an MCP tool from server '{server_name}'.

    Parameters:
    """

    # Add parameter descriptions from schema
    properties = tool_input_schema.get("properties", {})
    required = tool_input_schema.get("required", [])

    for param_name, param_schema in properties.items():
        param_type = param_schema.get("type", "string")
        param_desc = param_schema.get("description", "")
        is_required = param_name in required

        docstring += f"    :param {param_name}: {param_desc} ({param_type})"
        if not is_required:
            docstring += " (optional)"
        docstring += "\n"

    docstring += f"    :return: Result from MCP tool '{tool_name}'"

    def mcp_tool_wrapper(**kwargs):
        """Wrapper function for MCP tool execution."""
        try:
            # Call the MCP tool asynchronously
            result = run_async(
                connection_manager.call_tool(server_name, tool_name, kwargs)
            )
            return result
        except Exception as e:
            logger.error(f"Error calling MCP tool {prefixed_name}: {e}")
            return f"Error: {str(e)}"

    # Set the function metadata
    mcp_tool_wrapper.__name__ = prefixed_name
    mcp_tool_wrapper.__doc__ = docstring

    # Create the tool decorator with proper schema
    @tool
    def decorated_tool(**kwargs):
        """Decorated MCP tool function."""
        return mcp_tool_wrapper(**kwargs)

    # Update the decorated function's metadata
    decorated_tool.__name__ = prefixed_name
    decorated_tool.__doc__ = docstring

    return decorated_tool


def discover_mcp_tools(
    connection_manager: MCPConnectionManager, server_name: str
) -> List[Callable]:
    """Discover all tools from an MCP server and convert them to llamabot tools.

    :param connection_manager: MCP connection manager instance
    :param server_name: Name of the MCP server
    :return: List of converted tool functions
    """
    try:
        # Get tools from the server
        mcp_tools = run_async(connection_manager.list_tools(server_name))

        if not mcp_tools:
            logger.warning(f"No tools found on MCP server: {server_name}")
            return []

        # Convert each MCP tool to a llamabot tool
        llamabot_tools = []
        for mcp_tool in mcp_tools:
            try:
                llamabot_tool = mcp_tool_to_llamabot_tool(
                    mcp_tool, server_name, connection_manager
                )
                llamabot_tools.append(llamabot_tool)
                logger.debug(
                    f"Converted MCP tool: {server_name}:{mcp_tool.get('name', 'unknown')}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to convert MCP tool {mcp_tool.get('name', 'unknown')}: {e}"
                )
                continue

        logger.info(
            f"Discovered {len(llamabot_tools)} tools from MCP server: {server_name}"
        )
        return llamabot_tools

    except Exception as e:
        logger.error(f"Failed to discover tools from MCP server {server_name}: {e}")
        return []


def discover_all_mcp_tools(connection_manager: MCPConnectionManager) -> List[Callable]:
    """Discover tools from all configured MCP servers.

    :param connection_manager: MCP connection manager instance
    :return: List of all converted tool functions
    """
    all_tools = []

    for server in connection_manager.servers:
        server_name = server["name"]
        try:
            server_tools = discover_mcp_tools(connection_manager, server_name)
            all_tools.extend(server_tools)
        except Exception as e:
            logger.error(f"Failed to discover tools from server {server_name}: {e}")
            continue

    logger.info(f"Discovered {len(all_tools)} total MCP tools from all servers")
    return all_tools


def create_mcp_resource_tool(
    server_name: str, resource_uri: str, connection_manager: MCPConnectionManager
) -> Callable:
    """Create a tool for reading a specific MCP resource.

    :param server_name: Name of the MCP server
    :param resource_uri: URI of the resource
    :param connection_manager: MCP connection manager instance
    :return: Callable tool function for reading the resource
    """
    tool_name = f"{server_name}:read_resource"

    @tool
    def read_mcp_resource() -> str:
        """Read a resource from an MCP server.

        :return: Content of the MCP resource
        """
        try:
            content = run_async(
                connection_manager.read_resource(server_name, resource_uri)
            )
            return str(content)
        except Exception as e:
            logger.error(f"Error reading MCP resource {resource_uri}: {e}")
            return f"Error: {str(e)}"

    read_mcp_resource.__name__ = tool_name
    return read_mcp_resource


def create_mcp_prompt_tool(
    server_name: str, prompt_name: str, connection_manager: MCPConnectionManager
) -> Callable:
    """Create a tool for getting a specific MCP prompt.

    :param server_name: Name of the MCP server
    :param prompt_name: Name of the prompt
    :param connection_manager: MCP connection manager instance
    :return: Callable tool function for getting the prompt
    """
    tool_name = f"{server_name}:get_prompt_{prompt_name}"

    @tool
    def get_mcp_prompt(**kwargs) -> str:
        """Get a prompt from an MCP server.

        :param kwargs: Prompt arguments
        :return: Prompt content
        """
        try:
            prompt = run_async(
                connection_manager.get_prompt(server_name, prompt_name, kwargs)
            )
            return str(prompt)
        except Exception as e:
            logger.error(f"Error getting MCP prompt {prompt_name}: {e}")
            return f"Error: {str(e)}"

    get_mcp_prompt.__name__ = tool_name
    return get_mcp_prompt


def get_mcp_server_info(connection_manager: MCPConnectionManager) -> Dict[str, Any]:
    """Get information about all configured MCP servers.

    :param connection_manager: MCP connection manager instance
    :return: Dictionary with server information
    """
    server_info = {}

    for server in connection_manager.servers:
        server_name = server["name"]
        try:
            # Try to get basic info about the server
            tools = run_async(connection_manager.list_tools(server_name))
            resources = run_async(connection_manager.list_resources(server_name))
            prompts = run_async(connection_manager.list_prompts(server_name))

            server_info[server_name] = {
                "command": server["command"],
                "args": server.get("args", []),
                "tools_count": len(tools),
                "resources_count": len(resources),
                "prompts_count": len(prompts),
                "connected": connection_manager._connected.get(server_name, False),
            }
        except Exception as e:
            logger.error(f"Failed to get info for server {server_name}: {e}")
            server_info[server_name] = {
                "command": server["command"],
                "args": server.get("args", []),
                "error": str(e),
                "connected": False,
            }

    return server_info
