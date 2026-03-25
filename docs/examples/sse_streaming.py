# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "marimo",
#     "llamabot>=0.17.0",
#     "sse-starlette>=1.6.1",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # SSE Streaming with LlamaBot

    This notebook demonstrates how to use LlamaBot's Server-Sent Events (SSE) streaming capabilities for real-time chat applications.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Basic Streaming

    `AsyncSimpleBot` (parallel to `SimpleBot`) exposes `stream_async()`, which yields content chunks asynchronously—ideal for SSE or WebSocket streaming.
    """
    )
    return


@app.cell
def _():
    from llamabot import AsyncSimpleBot

    bot = AsyncSimpleBot(
        system_prompt="You are a helpful assistant.",
        model_name="ollama/phi3",  # Use your preferred model
        stream_target="none",  # We'll handle streaming ourselves
    )
    return (bot,)


@app.cell
async def _(bot):

    async def stream_example():
        chunks = []
        async for chunk in bot.stream_async("Tell me a short joke about programming."):
            chunks.append(chunk)
            print(chunk, end="", flush=True)  # Print as it streams
        print("\n\n---\n")
        print(f"Total chunks: {len(chunks)}")
        print(f"Full response: {''.join(chunks)}")

    await stream_example()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## SSE Format Conversion

    The `sse_stream()` utility converts bot streaming to SSE event format, ready for FastAPI's `EventSourceResponse`.
    """
    )
    return


@app.cell
async def _(bot):
    from llamabot.sse import sse_stream

    async def sse_example():
        events = []
        async for event in sse_stream(
            bot, ["What is Python?"], event_type="message", done_event="done"
        ):
            events.append(event)
            print(f"Event: {event['event']}, Data: {repr(event['data'])}")

        print("\n---\n")
        print(f"Total events: {len(events)}")
        print(f"Last event type: {events[-1]['event']}")

    await sse_example()
    return (sse_stream,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## FastAPI Integration

    Here's how to use SSE streaming in a FastAPI endpoint:
    """
    )
    return


@app.cell
def _(bot, sse_stream):
    from fastapi import FastAPI
    from pydantic import BaseModel
    from sse_starlette.sse import EventSourceResponse

    fastapi_app = FastAPI()

    class ChatRequest(BaseModel):
        messages: list[str]

    @fastapi_app.post("/chat")
    async def chat(request: ChatRequest):
        """Stream chat responses using Server-Sent Events."""
        return EventSourceResponse(
            sse_stream(bot, request.messages, event_type="message", done_event="done")
        )

    print("FastAPI endpoint defined!")
    print("\nTo run this server:")
    print("  uvicorn <module>:fastapi_app --reload")
    print("\nSee scripts/fastapi_sse_example.py for a complete example.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Error Handling

    The `sse_stream()` function automatically handles errors and yields error events:
    """
    )
    return


@app.cell
async def _(sse_stream):
    from unittest.mock import MagicMock

    # Create a bot that will raise an error
    error_bot = MagicMock()

    async def mock_stream_with_error(*args):
        yield "some"
        raise Exception("API error occurred")

    error_bot.stream_async = mock_stream_with_error

    async def error_example():
        events = []
        async for event in sse_stream(error_bot, ["test"]):
            events.append(event)
            print(f"Event: {event['event']}, Data: {event['data']}")

        print(f"\nTotal events: {len(events)}")
        print(f"Error event: {events[-1]}")

    await error_example()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## `stream_async()` vs sync `__call__()`

    Both paths can update **memory**, **SQLite logging**, and **spans** when streaming completes.

    - **`SimpleBot.__call__()`**: Synchronous; uses `stream_target` (stdout, panel, api, none) for where incremental output goes.
    - **`AsyncSimpleBot.stream_async()`**: Async token deltas via `litellm.acompletion`; intended for FastAPI/SSE. Does not use `stream_target` for those deltas.

    Use **`SimpleBot()`** / **`__call__()`** for a single blocking `AIMessage` in scripts and notebooks. Use **`AsyncSimpleBot`** with **`stream_async()`** (or `await AsyncSimpleBot(...)`) when you need async iteration (SSE, WebSockets, etc.).
    """
    )
    return


@app.cell
def _():
    print("\nComparison:")
    print("\nSimpleBot().__call__() - Synchronous completion:")
    print("  ✓ Returns AIMessage when done")
    print("  ✓ Uses stream_target for console/panel/api")
    print("\nAsyncSimpleBot.stream_async() - Async streaming:")
    print("  ✓ Real-time text chunks (async iterator)")
    print("  ✓ Memory / SQLite / spans on completion (same family as sync __call__)")
    print("  ✓ SSE / WebSocket friendly")
    return


if __name__ == "__main__":
    app.run()
