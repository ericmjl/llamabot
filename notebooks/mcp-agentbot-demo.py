# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "fastmcp==2.12.5",
#     "llamabot==0.13.11",
# ]
#
# [tool.uv.sources]
# llamabot = { path = "../", editable = true }
# ///
"""MCP AgentBot Demo - Demonstrating local and remote MCP tool integration.

This notebook shows how AgentBot can use both local tools (wrapped in a local MCP server)
and remote MCP tools through a unified MCP interface.
"""

import marimo

__generated_with = "0.8.0"
app = marimo.App(width="medium")


@app.cell
def __():
    """Import required modules."""
    import asyncio
    from llamabot import AgentBot
    from llamabot.components.tools import add, today_date, respond_to_user
    from llamabot.components.local_mcp_server import LocalMCPServer
    from fastmcp import Client, FastMCP

    print("✅ Imports successful")
    return (
        add,
        AgentBot,
        Client,
        FastMCP,
        LocalMCPServer,
        asyncio,
        respond_to_user,
        today_date,
    )


@app.cell
def __(LocalMCPServer, add, respond_to_user, today_date):
    """Create a local MCP server with some tools."""
    # Create local MCP server
    local_server = LocalMCPServer("local")

    # Register some local tools
    local_server.register_tools([add, today_date, respond_to_user])

    print(
        "✅ Local MCP server created with tools:",
        [tool.__name__ for tool in [add, today_date, respond_to_user]],
    )
    return (local_server,)


@app.cell
def __(FastMCP):
    """Create a mock remote MCP server for demonstration."""
    # Create a mock remote MCP server
    remote_server = FastMCP("remote")

    @remote_server.tool()
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers.

        :param a: First number
        :param b: Second number
        :return: Product of a and b
        """
        return a * b

    @remote_server.tool()
    def get_weather(city: str) -> str:
        """Get weather for a city.

        :param city: Name of the city
        :return: Weather information
        """
        return f"The weather in {city} is sunny and 72°F"

    print("✅ Mock remote MCP server created with tools: multiply, get_weather")
    return (remote_server,)


@app.cell
def __(Client, local_server, remote_server):
    """Create MCP clients for both local and remote servers."""
    # Create clients - pass FastMCP objects directly
    local_client = Client(local_server.get_server())
    remote_client = Client(remote_server)

    mcp_clients = [local_client, remote_client]

    print("✅ MCP clients created")
    return local_client, mcp_clients, remote_client


@app.cell
def __(mcp_clients):
    """Test tool discovery from both servers."""

    async def test_tool_discovery():
        """Test discovering tools from all MCP clients."""
        all_tools = []
        for i, client in enumerate(mcp_clients):
            try:
                tools = await client.list_tools()
                server_name = "local" if i == 0 else "remote"
                print(f"📋 {server_name} server tools:")
                for tool in tools:
                    print(f"  - {tool.name}: {tool.description}")
                    all_tools.append(tool.name)
            except Exception as e:
                print(f"❌ Failed to load tools from client {i}: {e}")

        return all_tools

    # Run the test
    discovered_tools = asyncio.run(test_tool_discovery())
    print(f"\n✅ Total tools discovered: {len(discovered_tools)}")
    return discovered_tools, test_tool_discovery


@app.cell
def __(mcp_clients):
    """Test tool execution from both servers."""

    async def test_tool_execution():
        """Test executing tools from both servers."""
        results = []

        # Test local tool (add)
        try:
            result = await mcp_clients[0].call_tool("add", {"a": 5, "b": 3})
            results.append(f"Local add(5, 3) = {result}")
        except Exception as e:
            results.append(f"❌ Local add failed: {e}")

        # Test remote tool (multiply)
        try:
            result = await mcp_clients[1].call_tool("multiply", {"a": 4, "b": 7})
            results.append(f"Remote multiply(4, 7) = {result}")
        except Exception as e:
            results.append(f"❌ Remote multiply failed: {e}")

        # Test remote tool (get_weather)
        try:
            result = await mcp_clients[1].call_tool(
                "get_weather", {"city": "San Francisco"}
            )
            results.append(f"Remote get_weather('San Francisco') = {result}")
        except Exception as e:
            results.append(f"❌ Remote get_weather failed: {e}")

        return results

    # Run the test
    execution_results = asyncio.run(test_tool_execution())
    for result in execution_results:
        print(f"🔧 {result}")
    return execution_results, test_tool_execution


@app.cell
def __(AgentBot, add, mcp_clients):
    """Create an AgentBot with MCP integration."""
    # Create AgentBot with local tools and MCP clients
    agent = AgentBot(
        tools=[add],  # Local tools
        # Note: In a real scenario, mcp_servers would be URLs to remote servers
        # For this demo, we're using the mock remote server
    )

    # Manually set the MCP clients (normally done in __init__)
    agent.toolbot.mcp_clients = mcp_clients

    print("✅ AgentBot created with MCP integration")
    return (agent,)


@app.cell
def __(agent):
    """Test AgentBot with a simple math question."""
    # Test the agent with a simple question
    question1 = "What is 15 + 27?"

    print(f"🤖 Question: {question1}")
    print("🤖 AgentBot is thinking...")

    try:
        response1 = agent(question1)
        print(f"🤖 Response: {response1.content}")
    except Exception as e:
        print(f"❌ AgentBot failed: {e}")
        import traceback as _traceback1

        _traceback1.print_exc()

    return question1, response1


@app.cell
def __(agent):
    """Test AgentBot with a question requiring remote tools."""
    # Test with a question that might use remote tools
    question2 = "What's the weather like in New York?"

    print(f"🤖 Question: {question2}")
    print("🤖 AgentBot is thinking...")

    try:
        response2 = agent(question2)
        print(f"🤖 Response: {response2.content}")
    except Exception as e:
        print(f"❌ AgentBot failed: {e}")
        import traceback as _traceback2

        _traceback2.print_exc()

    return question2, response2


@app.cell
def __(agent):
    """Test AgentBot with a complex question requiring multiple tools."""
    # Test with a complex question
    question3 = "Calculate 6 * 8 and then add 10 to the result"

    print(f"🤖 Question: {question3}")
    print("🤖 AgentBot is thinking...")

    try:
        response3 = agent(question3)
        print(f"🤖 Response: {response3.content}")
    except Exception as e:
        print(f"❌ AgentBot failed: {e}")
        import traceback as _traceback3

        _traceback3.print_exc()

    return question3, response3


@app.cell
def __():
    """Summary of the MCP integration."""
    summary = """
    ## MCP AgentBot Integration Summary

    ✅ **Local MCP Server**: Local Python functions wrapped as MCP tools
    ✅ **Remote MCP Server**: External MCP server with additional tools
    ✅ **Unified Interface**: ToolBot uses MCP clients for all tools
    ✅ **No Branching Logic**: All tools accessed through MCP protocol
    ✅ **Dynamic Discovery**: Tools discovered on-demand from all servers
    ✅ **Seamless Execution**: AgentBot doesn't know if tools are local or remote

    ### Key Benefits:
    - **Clean Architecture**: Single MCP interface for all tools
    - **Easy Extension**: Add new MCP servers without code changes
    - **Transparent Execution**: Tools work the same regardless of location
    - **Future-Proof**: Easy to add preferences, fallback, etc.
    """

    print(summary)
    return (summary,)


if __name__ == "__main__":
    app.run()
