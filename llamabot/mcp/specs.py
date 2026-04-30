"""Transport-agnostic MCP server configuration for LlamaBot."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class MCPStartupMode(str, Enum):
    """How to behave when an MCP server fails during startup."""

    STRICT = "strict"
    BEST_EFFORT = "best_effort"


class MCPIntegrationOptions(BaseModel):
    """Options for merging MCP tools with Python tools."""

    startup_mode: MCPStartupMode = MCPStartupMode.BEST_EFFORT
    """If ``strict``, raise when any configured server fails to connect or list tools."""

    tool_namespace_sep: str = "__"
    """Separator between server name and remote tool name in generated Python identifiers."""

    connect_timeout_seconds: float = 60.0
    """Time to wait for the MCP session to become ready."""

    call_timeout_seconds: Optional[float] = None
    """Per-tool-call timeout in seconds; ``None`` uses the FastMCP client default."""

    allow_tools: Optional[list[str]] = None
    """If set, only register tools whose prefixed names are in this list."""

    deny_tools: Optional[list[str]] = None
    """If set, skip tools whose prefixed names appear here."""

    client_timeout_seconds: Optional[float] = None
    """Optional timeout passed to :class:`fastmcp.Client` (seconds)."""

    client_init_timeout_seconds: Optional[float] = None
    """Optional MCP initialize handshake timeout for :class:`fastmcp.Client` (seconds)."""


class MCPServerSpec(BaseModel):
    """Declares one MCP server for use with :class:`~llamabot.mcp.manager.MCPClientManager`.

    Supports stdio subprocess servers, remote HTTP / Streamable HTTP / SSE URLs, and
    in-process :class:`fastmcp.FastMCP` instances (primarily for tests).
    """

    name: str
    """Logical server name used for namespacing and logging."""

    transport: Literal["stdio", "http", "sse", "streamable-http", "inproc"] = "stdio"
    """Transport kind; ``inproc`` embeds a :class:`fastmcp.FastMCP` server in-process."""

    command: Optional[str] = None
    """Executable for ``stdio`` transport."""

    args: list[str] = Field(default_factory=list)
    """Arguments for ``stdio`` transport."""

    env: dict[str, Any] = Field(default_factory=dict)
    """Extra environment variables for ``stdio`` transport."""

    cwd: Optional[str] = None
    """Working directory for ``stdio`` transport."""

    url: Optional[str] = None
    """Endpoint URL for remote transports."""

    remote_transport: Optional[Literal["http", "sse", "streamable-http"]] = None
    """Explicit remote transport; if omitted, FastMCP infers from ``url``."""

    headers: dict[str, str] = Field(default_factory=dict)
    """HTTP headers for remote transports."""

    auth: Any = None
    """Optional auth for remote transports (Bearer string, ``\"oauth\"``, or ``httpx.Auth``)."""

    sse_read_timeout_seconds: Optional[float] = None
    """Optional SSE read timeout for remote transports (seconds)."""

    fastmcp: Any = Field(default=None, repr=False)
    """In-process :class:`fastmcp.FastMCP` instance when ``transport == \"inproc\"``."""

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def validate_transport_fields(self) -> MCPServerSpec:
        """Ensure required fields exist for each transport.

        :return: Validated spec.
        :raises ValueError: When configuration is inconsistent.
        """
        if self.transport == "stdio":
            if not self.command:
                raise ValueError("stdio transport requires 'command'")
        elif self.transport == "inproc":
            if self.fastmcp is None:
                raise ValueError("inproc transport requires 'fastmcp' instance")
        else:
            if not self.url:
                raise ValueError(f"{self.transport} transport requires 'url'")
        return self
