# marimo
# title: MCP Integration Demo
# description: Demonstrates connecting ToolBot and AgentBot to MCP servers
# author: Eric Ma
# date: 2025-01-18
# version: 1.0.0
# tags: [MCP, ToolBot, AgentBot, FastMCP]

import marimo

__generated_with = "0.8.0"

app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import os
    import sys
    from pathlib import Path

    # Add the llamabot package to the path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from llamabot.bot.toolbot import ToolBot, toolbot_sysprompt
    from llamabot.bot.agentbot import AgentBot
    from llamabot.components.mcp_client import MCPConnectionManager
    from llamabot.components.mcp_tools import (
        discover_all_mcp_tools,
        get_mcp_server_info,
    )

    mo.md(
        """
        # MCP Integration Demo

        This notebook demonstrates how to connect ToolBot and AgentBot to external
        MCP (Model Context Protocol) servers.
        """
    )
    return (
        AgentBot,
        MCPConnectionManager,
        Path,
        ToolBot,
        discover_all_mcp_tools,
        get_mcp_server_info,
        mo,
        os,
        sys,
        toolbot_sysprompt,
    )


@app.cell
def __(mo):
    mo.md(
        """
        ## 1. Basic MCP Connection

        Let's start by connecting to llamabot's own MCP server to demonstrate
        the basic connection pattern.
        """
    )
    return


@app.cell
def __(MCPConnectionManager):
    # Configure the llamabot MCP server
    llamabot_mcp_server = {
        "name": "llamabot_docs",
        "command": "uvx",
        "args": ["--with", "llamabot[all]", "llamabot", "mcp", "launch"],
        "env": {},
    }

    # Create connection manager
    connection_manager = MCPConnectionManager([llamabot_mcp_server])
    return connection_manager, llamabot_mcp_server


@app.cell
def __(connection_manager, get_mcp_server_info, mo):
    # Get server information
    server_info = get_mcp_server_info(connection_manager)
    mo.display(server_info)
    return (server_info,)


@app.cell
def __(mo):
    mo.md(
        """
        ## 2. Tool Discovery

        Now let's discover what tools are available on the MCP server.
        """
    )
    return


@app.cell
def __(connection_manager, discover_all_mcp_tools, mo):
    # Discover tools from the MCP server
    mcp_tools = discover_all_mcp_tools(connection_manager)

    mo.md(f"**Discovered {len(mcp_tools)} MCP tools:**")
    for tool in mcp_tools:
        mo.md(
            f"- `{tool.__name__}`: {tool.__doc__.split('.')[0] if tool.__doc__ else 'No description'}"
        )
    return (mcp_tools,)


@app.cell
def __(mo):
    mo.md(
        """
        ## 3. ToolBot with MCP Integration

        Now let's create a ToolBot that can use both local tools and MCP tools.
        """
    )
    return


@app.cell
def __(ToolBot, llamabot_mcp_server, toolbot_sysprompt):
    # Create ToolBot with MCP server configuration
    toolbot = ToolBot(
        system_prompt=toolbot_sysprompt(globals_dict={}),
        model_name="ollama_chat/llama3.1:latest",
        mcp_servers=[llamabot_mcp_server],
    )
    return (toolbot,)


@app.cell
def __(mo, toolbot):
    # Test the ToolBot with a query that should use MCP tools
    mo.md("**Testing ToolBot with MCP integration:**")

    # This will trigger MCP tool discovery
    query = "Search for information about ToolBot in the llamabot documentation"
    mo.md(f"**Query:** {query}")

    # The bot will discover MCP tools on first call
    tool_calls = toolbot(query)
    mo.md(f"**Tool calls generated:** {len(tool_calls)}")

    for i, call in enumerate(tool_calls):
        mo.md(f"**Tool {i + 1}:** {call.function.name}")
        mo.md(f"**Arguments:** {call.function.arguments}")
    return call, i, query, tool_calls


@app.cell
def __(mo):
    mo.md(
        """
        ## 4. AgentBot with MCP Integration

        AgentBot can also use MCP tools in its ReAct loop.
        """
    )
    return


@app.cell
def __(AgentBot, llamabot_mcp_server):
    # Create AgentBot with MCP server configuration
    agentbot = AgentBot(
        model_name="ollama_chat/llama3.1:latest", mcp_servers=[llamabot_mcp_server]
    )
    return (agentbot,)


@app.cell
def __(agentbot, mo):
    mo.md("**Testing AgentBot with MCP integration:**")

    # Test with a complex query that requires multiple steps
    complex_query = (
        "Find information about how to use ToolBot and create a simple example"
    )
    mo.md(f"**Query:** {complex_query}")

    # AgentBot will use MCP tools in its reasoning loop
    try:
        result = agentbot(complex_query)
        mo.md(f"**Result:** {result.content}")
    except Exception as e:
        mo.md(f"**Error:** {e}")
    return complex_query, e, result


@app.cell
def __(mo):
    mo.md(
        """
        ## 5. Multiple MCP Servers

        You can connect to multiple MCP servers simultaneously.
        """
    )
    return


@app.cell
def __(mo):
    # Example configuration for multiple servers
    multiple_servers = [
        {
            "name": "llamabot_docs",
            "command": "uvx",
            "args": ["--with", "llamabot[all]", "llamabot", "mcp", "launch"],
            "env": {},
        },
        # Add more servers here as needed
        # {
        #     "name": "filesystem",
        #     "command": "npx",
        #     "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        #     "env": {}
        # }
    ]

    mo.md("**Multiple server configuration:**")
    for server in multiple_servers:
        mo.md(f"- **{server['name']}**: {server['command']} {' '.join(server['args'])}")
    return multiple_servers, server


@app.cell
def __(mo):
    mo.md(
        """
        ## 6. Best Practices and Error Handling

        ### Best Practices:

        1. **Lazy Connection**: MCP servers are connected only when first needed
        2. **Error Resilience**: If MCP connection fails, bot continues with local tools
        3. **Namespacing**: MCP tools are prefixed with server name (e.g., `server:tool_name`)
        4. **Resource Management**: Connections are automatically cleaned up

        ### Error Handling:

        The MCP integration includes comprehensive error handling:
        - Connection failures don't break the bot
        - Tool discovery errors are logged but don't stop execution
        - Individual tool call failures are handled gracefully
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        ## 7. Comparison with In-Runtime Tools

        | Feature | In-Runtime Tools | MCP Tools |
        |---------|------------------|-----------|
        | **Execution** | Same process | External process |
        | **Performance** | Fast (no I/O) | Slower (network/process) |
        | **Security** | Full access | Sandboxed |
        | **Scalability** | Limited | High |
        | **Maintenance** | Code changes | Server updates |
        | **Discovery** | Static | Dynamic |

        ### When to Use Each:

        - **In-Runtime Tools**: Fast, simple operations, data processing
        - **MCP Tools**: External services, file system access, specialized capabilities
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        ## 8. Troubleshooting

        ### Common Issues:

        1. **Connection Failed**: Check if MCP server is running and accessible
        2. **No Tools Found**: Verify server configuration and permissions
        3. **Tool Execution Error**: Check tool parameters and server logs
        4. **Performance Issues**: Consider connection pooling or caching

        ### Debug Information:

        Enable debug logging to see MCP connection details:
        ```python
        import logging
        logging.getLogger("llamabot.components.mcp_client").setLevel(logging.DEBUG)
        ```
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        ## Conclusion

        This demo shows how MCP integration enhances ToolBot and AgentBot with:

        - **External Tool Access**: Use tools from remote servers
        - **Dynamic Discovery**: Automatically find available tools
        - **Seamless Integration**: MCP tools work alongside local tools
        - **Error Resilience**: Graceful handling of connection issues

        The MCP integration complements the existing tool system, providing
        a bridge to external capabilities while maintaining the simplicity
        and reliability of the core llamabot functionality.
        """
    )
    return


if __name__ == "__main__":
    app.run()
