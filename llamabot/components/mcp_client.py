"""MCP Client component for connecting to external MCP servers.

This module provides functionality to connect ToolBot and AgentBot to external
MCP (Model Context Protocol) servers, enabling them to discover and use tools,
resources, and prompts from these servers.
"""

import asyncio
import subprocess
from typing import Any, Dict, List, Optional
from loguru import logger

from fastmcp import Client


class MCPConnectionManager:
    """Manages connections to multiple MCP servers.

    This class handles the lifecycle of MCP server connections, including
    lazy connection, caching, and cleanup. It uses fastmcp.Client internally
    to handle the MCP protocol details.

    :param servers: List of MCP server configurations
    """

    def __init__(self, servers: List[Dict[str, Any]]):
        # Initialize the connection manager with server configurations.
        # servers: List of server configurations, each containing:
        #     - name: Server identifier
        #     - command: Command to start the server
        #     - args: Command arguments (optional)
        #     - env: Environment variables (optional)
        self.servers = servers
        self._connections: Dict[str, Client] = {}
        self._connected: Dict[str, bool] = {server["name"]: False for server in servers}

    async def get_client(self, server_name: str) -> Optional[Client]:
        """Get a connected client for the specified server.

        :param server_name: Name of the server to connect to
        :return: Connected Client instance or None if connection failed
        """
        if server_name in self._connections and self._connected.get(server_name, False):
            return self._connections[server_name]

        # Find server configuration
        server_config = None
        for server in self.servers:
            if server["name"] == server_name:
                server_config = server
                break

        if not server_config:
            logger.error(f"Server configuration not found for: {server_name}")
            return None

        try:
            # Create client with stdio transport
            client = await self._connect_to_server(server_config)
            if client:
                self._connections[server_name] = client
                self._connected[server_name] = True
                logger.debug(f"Connected to MCP server: {server_name}")
                return client
        except Exception as e:
            logger.error(f"Failed to connect to MCP server {server_name}: {e}")
            self._connected[server_name] = False

        return None

    async def _connect_to_server(
        self, server_config: Dict[str, Any]
    ) -> Optional[Client]:
        """Connect to a specific MCP server.

        :param server_config: Server configuration dictionary
        :return: Connected Client instance or None if failed
        """
        command = server_config["command"]
        args = server_config.get("args", [])
        env = server_config.get("env", {})

        # Start the server process
        process = subprocess.Popen(
            [command] + args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**server_config.get("env", {}), **env},
        )

        # Create client with stdio transport
        client = Client(process.stdin, process.stdout)

        # Initialize the connection
        await client.initialize()

        return client

    async def list_tools(self, server_name: str) -> List[Dict[str, Any]]:
        """List available tools from a specific server.

        :param server_name: Name of the server
        :return: List of tool definitions
        """
        client = await self.get_client(server_name)
        if not client:
            return []

        try:
            tools = await client.list_tools()
            return tools
        except Exception as e:
            logger.error(f"Failed to list tools from {server_name}: {e}")
            return []

    async def call_tool(
        self, server_name: str, tool_name: str, arguments: Dict[str, Any]
    ) -> Any:
        """Call a tool on a specific server.

        :param server_name: Name of the server
        :param tool_name: Name of the tool to call
        :param arguments: Tool arguments
        :return: Tool result
        """
        client = await self.get_client(server_name)
        if not client:
            raise ConnectionError(f"Not connected to server: {server_name}")

        try:
            result = await client.call_tool(tool_name, arguments)
            return result
        except Exception as e:
            logger.error(f"Failed to call tool {tool_name} on {server_name}: {e}")
            raise

    async def list_resources(self, server_name: str) -> List[Dict[str, Any]]:
        """List available resources from a specific server.

        :param server_name: Name of the server
        :return: List of resource definitions
        """
        client = await self.get_client(server_name)
        if not client:
            return []

        try:
            resources = await client.list_resources()
            return resources
        except Exception as e:
            logger.error(f"Failed to list resources from {server_name}: {e}")
            return []

    async def read_resource(self, server_name: str, resource_uri: str) -> Any:
        """Read a resource from a specific server.

        :param server_name: Name of the server
        :param resource_uri: URI of the resource to read
        :return: Resource content
        """
        client = await self.get_client(server_name)
        if not client:
            raise ConnectionError(f"Not connected to server: {server_name}")

        try:
            content = await client.read_resource(resource_uri)
            return content
        except Exception as e:
            logger.error(
                f"Failed to read resource {resource_uri} from {server_name}: {e}"
            )
            raise

    async def list_prompts(self, server_name: str) -> List[Dict[str, Any]]:
        """List available prompts from a specific server.

        :param server_name: Name of the server
        :return: List of prompt definitions
        """
        client = await self.get_client(server_name)
        if not client:
            return []

        try:
            prompts = await client.list_prompts()
            return prompts
        except Exception as e:
            logger.error(f"Failed to list prompts from {server_name}: {e}")
            return []

    async def get_prompt(
        self, server_name: str, prompt_name: str, arguments: Dict[str, Any]
    ) -> str:
        """Get a prompt from a specific server.

        :param server_name: Name of the server
        :param prompt_name: Name of the prompt
        :param arguments: Prompt arguments
        :return: Prompt content
        """
        client = await self.get_client(server_name)
        if not client:
            raise ConnectionError(f"Not connected to server: {server_name}")

        try:
            prompt = await client.get_prompt(prompt_name, arguments)
            return prompt
        except Exception as e:
            logger.error(f"Failed to get prompt {prompt_name} from {server_name}: {e}")
            raise

    async def close_all(self):
        """Close all server connections."""
        for server_name, client in self._connections.items():
            try:
                await client.close()
                logger.debug(f"Closed connection to {server_name}")
            except Exception as e:
                logger.error(f"Error closing connection to {server_name}: {e}")

        self._connections.clear()
        self._connected.clear()

    def __del__(self):
        """Cleanup connections on destruction."""
        if self._connections:
            # Schedule cleanup in the event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.close_all())
            except RuntimeError:
                # No event loop running, can't schedule cleanup
                pass


def run_async(coro):
    """Run an async coroutine in a sync context.

    :param coro: Async coroutine to run
    :return: Result of the coroutine
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an event loop, we need to use a different approach
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop, create a new one
        return asyncio.run(coro)
