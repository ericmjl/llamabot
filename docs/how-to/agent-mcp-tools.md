# Use MCP tools with AgentBot

LlamaBot can merge **Python tools** (``@tool`` functions) with tools discovered from **MCP servers**
(stdio subprocesses or remote HTTP / SSE / streamable HTTP endpoints) using FastMCP's client.

## When to use this

- You already expose capabilities as an MCP server and want the same tools inside ``AgentBot``
  or ``ToolBot`` without rewriting them as Python callables.
- You mix a few in-repo Python tools with shared MCP-hosted tools (e.g. org-wide servers).

## Constructor parameters

- ``mcp_servers``: list of :class:`~llamabot.mcp.specs.MCPServerConfig`
- ``mcp_options``: optional :class:`~llamabot.mcp.specs.MCPIntegrationOptions`

MCP tools are registered **after** your Python ``tools=`` list (defaults from ``DEFAULT_TOOLS`` stay first).

## Lifespan

Each MCP server runs a **persistent session** on a background thread until you call:

- ``AgentBot.close_mcp()`` or ``AsyncAgentBot.close_mcp()``
- ``ToolBot.close_mcp()`` when ``ToolBot`` was constructed with ``mcp_servers=``

Call one of these when the bot is discarded (especially in long-lived apps) so subprocesses and HTTP sessions shut down cleanly.

## Example (in-process server for tests)

```python
from fastmcp import FastMCP

from llamabot.bot.agentbot import AgentBot
from llamabot.components.tools import tool
from llamabot.mcp.specs import MCPServerConfig, MCPIntegrationOptions, MCPStartupMode

mcp = FastMCP("demo")

@mcp.tool()
def greet(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}!"

@tool
def double(x: int) -> int:
    """Double an integer."""
    return x * 2

bot = AgentBot(
    tools=[double],
    mcp_servers=[MCPServerConfig(name="demo", transport="inproc", fastmcp=mcp)],
    mcp_options=MCPIntegrationOptions(startup_mode=MCPStartupMode.STRICT),
)

try:
    bot("Use greet via MCP with name Ada")  # model selects demo__greet
finally:
    bot.close_mcp()
```

## Remote URL server

```python
from llamabot.mcp.specs import MCPServerConfig

MCPServerConfig(
    name="remote",
    transport="http",
    url="http://localhost:8080/mcp",
    headers={"Authorization": "Bearer TOKEN"},
)
```

For SSE endpoints, set ``transport="sse"`` (or rely on FastMCP URL path inference when using ``transport="http"``).

## Stdio server (same shape as Cursor MCP config)

```python
MCPServerConfig(
    name="docs",
    transport="stdio",
    command="uvx",
    args=["--with", "llamabot[all]", "llamabot", "mcp", "launch"],
    env={},
)
```

## Tool naming and filtering

- Remote tool ``echo`` on server ``srv`` becomes Python symbol ``srv__echo`` by default (separator from ``MCPIntegrationOptions.tool_namespace_sep``).
- JSON Schema property names that are not valid Python identifiers are sanitized for the model; wire names are preserved when calling MCP.
- ``allow_tools`` / ``deny_tools`` on :class:`~llamabot.mcp.specs.MCPIntegrationOptions` filter by **prefixed** names (e.g. ``srv__echo``).

## Startup modes

- ``MCPStartupMode.BEST_EFFORT`` (default): failed servers are skipped; check ``MCPClientManager.failures`` if you construct the manager yourself.
- ``MCPStartupMode.STRICT``: any connection or discovery failure raises during bot construction.
