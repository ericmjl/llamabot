"""SSE streaming utilities for LlamaBot."""

from typing import Any, AsyncGenerator, Protocol, runtime_checkable


@runtime_checkable
class SupportsStreamAsync(Protocol):
    """Object that exposes an async text stream (e.g. :meth:`~llamabot.bot.simplebot.AsyncSimpleBot.stream_async`)."""

    def stream_async(self, *args: Any, **kwargs: Any) -> AsyncGenerator[str, None]:
        """Yield incremental assistant text chunks."""
        ...


async def sse_stream(
    bot: SupportsStreamAsync,
    messages: list[Any],
    event_type: str = "message",
    done_event: str = "done",
) -> AsyncGenerator[dict[str, str], None]:
    """Convert bot streaming to SSE event format.

    Yields dictionaries compatible with sse-starlette's EventSourceResponse.

    Example usage with FastAPI:
        from llamabot import AsyncSimpleBot
        from llamabot.sse import sse_stream
        from sse_starlette.sse import EventSourceResponse

        @app.post("/api/chat")
        async def chat(request: ChatRequest):
            bot = AsyncSimpleBot(system_prompt="You are helpful.")
            return EventSourceResponse(sse_stream(bot, request.messages))

    :param bot: An object with ``stream_async`` (e.g. :class:`~llamabot.bot.simplebot.AsyncSimpleBot`).
    :param messages: Message list passed as ``*messages`` to ``stream_async`` (e.g. ``list[str]``).
    :param event_type: SSE event type for content chunks (default: "message").
    :param done_event: SSE event type for completion signal (default: "done").
    :return: Async generator yielding SSE event dictionaries.
    """
    try:
        async for chunk in bot.stream_async(*messages):
            yield {"event": event_type, "data": chunk}
        yield {"event": done_event, "data": ""}
    except Exception as e:
        yield {"event": "error", "data": str(e)}
