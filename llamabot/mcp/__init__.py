"""MCP client integration for LlamaBot agents.

Only the light ``specs`` module is imported eagerly. ``MCPClientManager``
is exposed lazily via module ``__getattr__`` because importing it pulls in
the pocketflow adapter chain (~200ms), which would otherwise tax every
``from llamabot import ToolBot`` even when no MCP servers are used.
"""

from typing import TYPE_CHECKING

from llamabot.mcp.specs import MCPIntegrationOptions, MCPServerConfig, MCPStartupMode

if TYPE_CHECKING:
    from llamabot.mcp.manager import MCPClientManager

__all__ = [
    "MCPClientManager",
    "MCPIntegrationOptions",
    "MCPServerConfig",
    "MCPStartupMode",
]


def __getattr__(name: str):
    """Lazily import MCPClientManager on first attribute access.

    :param name: Attribute being accessed on this package.
    :return: The requested attribute (only ``MCPClientManager`` is lazy).
    :raises AttributeError: If the attribute is not part of this package.
    """
    if name == "MCPClientManager":
        from llamabot.mcp.manager import MCPClientManager

        return MCPClientManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
