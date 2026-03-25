"""SSE streaming utilities for LlamaBot."""

from typing import AsyncGenerator


async def sse_stream(
    bot, messages, event_type: str = "message", done_event: str = "done"
) -> AsyncGenerator[dict, None]:
    """Convert bot streaming to SSE event format.

    Yields dictionaries compatible with sse-starlette's EventSourceResponse.

    Example usage with FastAPI:
        from llamabot import SimpleBot
        from llamabot.sse import sse_stream
        from sse_starlette.sse import EventSourceResponse

        @app.post("/api/chat")
        async def chat(request: ChatRequest):
            bot = SimpleBot(system_prompt="You are helpful.")
            return EventSourceResponse(sse_stream(bot, request.messages))

    :param bot: A SimpleBot instance with stream_async() method.
    :param messages: Messages to send to the bot.
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
