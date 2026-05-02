"""MCP client lifecycle: connect servers and expose LlamaBot tools."""

from __future__ import annotations

from typing import Any, Callable, List

from loguru import logger

from llamabot.mcp.adapter import mcp_tools_as_llamabot_tools
from llamabot.mcp.session import PersistentMCPClientSession, build_fastmcp_client
from llamabot.mcp.specs import (
    MCPIntegrationOptions,
    MCPStartupMode,
    coerce_mcp_server_configs,
)


class MCPClientManager:
    """Connect to declared MCP servers and expose LlamaBot-compatible tools.

    :param servers: MCP servers to connect at :meth:`start`.
    :param options: Optional integration settings.
    """

    def __init__(
        self,
        servers: List[Any],
        options: MCPIntegrationOptions | None = None,
    ) -> None:
        self._servers = coerce_mcp_server_configs(list(servers))
        self._options = options or MCPIntegrationOptions()
        self._sessions: dict[str, PersistentMCPClientSession] = {}
        self._failures: list[tuple[str, BaseException]] = []

    @property
    def failures(self) -> list[tuple[str, BaseException]]:
        """Servers that failed under ``best_effort`` startup.

        :return: List of ``(server_name, error)`` pairs.
        """
        return list(self._failures)

    def start(self) -> None:
        """Connect all configured servers.

        :raises TimeoutError: In strict mode when a server hits connect timeout.
        :raises RuntimeError: In strict mode when startup fails.
        """
        self._failures.clear()
        for config in self._servers:
            client = build_fastmcp_client(config, self._options)
            session = PersistentMCPClientSession(
                client,
                connect_timeout=float(self._options.connect_timeout_seconds),
            )
            try:
                session.start()
                self._sessions[config.name] = session
                logger.info("Connected MCP server {!r}", config.name)
            except BaseException as exc:
                self._failures.append((config.name, exc))
                logger.exception("Failed to connect MCP server {!r}", config.name)
                if self._options.startup_mode == MCPStartupMode.STRICT:
                    self.close()
                    raise

    def close(self) -> None:
        """Close all MCP sessions."""
        for sess in self._sessions.values():
            sess.close()
        self._sessions.clear()

    def llamabot_tools(self) -> List[Callable[..., Any]]:
        """Build ``@tool``-compatible callables for all discovered MCP tools.

        Call after :meth:`start`.

        :return: List of decorated tool functions.
        """
        tools: List[Callable[..., Any]] = []
        sep = self._options.tool_namespace_sep
        call_timeout = self._options.call_timeout_seconds
        for config in self._servers:
            session = self._sessions.get(config.name)
            if session is None:
                continue

            remote_tools = session.run_coroutine(
                session.client.list_tools(),
                timeout=call_timeout,
            )
            tools.extend(
                mcp_tools_as_llamabot_tools(
                    session,
                    config.name,
                    remote_tools,
                    namespace_sep=sep,
                    options=self._options,
                    call_timeout=call_timeout,
                )
            )
        return tools
