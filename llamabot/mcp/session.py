"""Low-level MCP session: FastMCP client construction and persistent asyncio session."""

from __future__ import annotations

import asyncio
import threading
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from loguru import logger

from llamabot.mcp.specs import MCPIntegrationOptions, MCPServerConfig

if TYPE_CHECKING:
    from fastmcp import Client


def build_fastmcp_client(
    config: MCPServerConfig, options: MCPIntegrationOptions
) -> Client:
    """Create a :class:`fastmcp.Client` from a server config.

    :param config: Declared MCP server configuration.
    :param options: Integration timeouts and client options.
    :return: Configured FastMCP client (not yet connected).
    """
    from fastmcp import Client
    from fastmcp.mcp_config import StdioMCPServer

    kwargs: dict[str, Any] = {}
    if options.client_timeout_seconds is not None:
        kwargs["timeout"] = timedelta(seconds=float(options.client_timeout_seconds))
    if options.client_init_timeout_seconds is not None:
        kwargs["init_timeout"] = float(options.client_init_timeout_seconds)

    if config.transport == "inproc":
        return Client(config.fastmcp, name=config.name, **kwargs)

    if config.transport == "stdio":
        stdio = StdioMCPServer(
            command=config.command,
            args=list(config.args),
            env=dict(config.env),
            cwd=config.cwd,
        )
        return Client(stdio.to_transport(), name=config.name, **kwargs)

    from fastmcp import Client
    from fastmcp.mcp_config import RemoteMCPServer

    remote_transport: str | None
    if config.remote_transport is not None:
        remote_transport = config.remote_transport
    elif config.transport == "http":
        remote_transport = None
    elif config.transport == "sse":
        remote_transport = "sse"
    elif config.transport == "streamable-http":
        remote_transport = "streamable-http"
    else:
        remote_transport = None

    sse_read_timeout = None
    if config.sse_read_timeout_seconds is not None:
        sse_read_timeout = timedelta(seconds=float(config.sse_read_timeout_seconds))

    remote = RemoteMCPServer(
        url=config.url,
        transport=remote_transport,
        headers=dict(config.headers),
        auth=config.auth,
        sse_read_timeout=sse_read_timeout,
    )
    return Client(remote.to_transport(), name=config.name, **kwargs)


class PersistentMCPClientSession:
    """Hold one connected :class:`fastmcp.Client` on a dedicated event-loop thread.

    :param client: FastMCP client to connect.
    :param connect_timeout: Seconds to wait for the session to become ready.
    """

    def __init__(self, client: Client, connect_timeout: float) -> None:
        self._client = client
        self._connect_timeout = connect_timeout
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stop_fut: asyncio.Future[Any] | None = None
        self._startup_error: BaseException | None = None

    @property
    def client(self) -> Client:
        """Return the wrapped FastMCP client.

        :return: Client instance.
        """
        return self._client

    @property
    def loop(self) -> asyncio.AbstractEventLoop | None:
        """Return the background event loop, if started.

        :return: Event loop or ``None``.
        """
        return self._loop

    def start(self) -> None:
        """Start the background thread and wait until the MCP session is ready.

        :raises TimeoutError: When connect timeout elapses before readiness.
        :raises RuntimeError: When the session fails to start.
        """

        def runner() -> None:
            """Run the MCP client session on a dedicated asyncio event loop."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop

            async def hold_session() -> None:
                """Enter ``async with client``, signal readiness, await shutdown."""
                try:
                    async with self._client:
                        self._ready.set()
                        stop_wait: asyncio.Future[Any] = loop.create_future()
                        self._stop_fut = stop_wait
                        await stop_wait
                except BaseException as exc:
                    self._startup_error = exc
                    self._ready.set()
                    raise

            try:
                loop.run_until_complete(hold_session())
            except BaseException:
                # Failure is surfaced via ``_startup_error`` after ``_ready``;
                # swallow here so the worker thread exits without threading leaks.
                pass
            finally:
                try:
                    pending = asyncio.all_tasks(loop)
                    for t in pending:
                        t.cancel()
                    if pending:
                        loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                except Exception:
                    logger.exception("Error cancelling MCP background tasks")
                loop.close()

        self._thread = threading.Thread(
            target=runner,
            daemon=True,
            name=f"mcp-session-{self._client.name}",
        )
        self._thread.start()
        if not self._ready.wait(timeout=self._connect_timeout):
            self.close()
            raise TimeoutError(
                f"MCP server {self._client.name!r} did not become ready within "
                f"{self._connect_timeout} seconds"
            )
        if self._startup_error is not None:
            err = self._startup_error
            self.close()
            raise RuntimeError(
                f"MCP server {self._client.name!r} failed to start"
            ) from err

    def run_coroutine(self, coro: Any, *, timeout: float | None) -> Any:
        """Run an awaitable on the session loop from another thread.

        :param coro: Coroutine produced by the FastMCP client.
        :param timeout: Optional timeout in seconds for the operation.
        :return: Coroutine result.
        """
        if self._loop is None:
            raise RuntimeError("MCP session loop is not running")
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=timeout)

    def close(self) -> None:
        """Signal shutdown and join the background thread."""
        if self._loop is not None and self._stop_fut is not None:
            if not self._stop_fut.done():

                def resolve() -> None:
                    """Complete the stop future so the session context can exit."""
                    if self._stop_fut is not None and not self._stop_fut.done():
                        self._stop_fut.set_result(None)

                self._loop.call_soon_threadsafe(resolve)
        if self._thread is not None:
            self._thread.join(timeout=15.0)
            self._thread = None
