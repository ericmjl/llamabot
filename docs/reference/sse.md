# SSE Streaming API Reference

The SSE (Server-Sent Events) streaming utilities provide a simple way to convert LlamaBot's async streaming into SSE event format for FastAPI and other web frameworks.

## `sse_stream`

```python
async def sse_stream(
    bot,
    messages,
    event_type: str = "message",
    done_event: str = "done"
) -> AsyncGenerator[dict, None]
```

Convert bot streaming to SSE event format. Yields dictionaries compatible with `sse-starlette`'s `EventSourceResponse`.

### Parameters

- **bot**: An object with a `stream_async()` method (typically `AsyncSimpleBot` from `llamabot`) that accepts messages and returns an `AsyncGenerator[str, None]`.
- **messages**: Messages to send to the bot. Can be a list of strings, or unpacked as `*messages`.
- **event_type** (`str`, default: `"message"`): SSE event type for content chunks.
- **done_event** (`str`, default: `"done"`): SSE event type for completion signal.

### Returns

- **AsyncGenerator[dict, None]**: An async generator yielding SSE event dictionaries with `"event"` and `"data"` keys.

### Event Format

The function yields dictionaries in this format:

```python
{"event": "message", "data": "chunk content"}
{"event": "message", "data": " more content"}
{"event": "done", "data": ""}
```

If an error occurs:

```python
{"event": "error", "data": "Error message here"}
```

### Example Usage

#### Basic FastAPI Endpoint

```python
from fastapi import FastAPI
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from llamabot import AsyncSimpleBot
from llamabot.sse import sse_stream

app = FastAPI()
bot = AsyncSimpleBot("You are a helpful assistant.", stream_target="none")

class ChatRequest(BaseModel):
    messages: list[str]

@app.post("/chat")
async def chat(request: ChatRequest):
    """Stream chat responses using Server-Sent Events."""
    return EventSourceResponse(
        sse_stream(bot, request.messages, event_type="message", done_event="done")
    )
```

#### Custom Event Types

```python
# Use custom event names
async for event in sse_stream(
    bot,
    ["Hello!"],
    event_type="content",
    done_event="complete"
):
    # event["event"] will be "content" or "complete"
    print(f"{event['event']}: {event['data']}")
```

#### Error Handling

The function automatically catches exceptions and yields error events:

```python
async for event in sse_stream(bot, ["test"]):
    if event["event"] == "error":
        print(f"Error occurred: {event['data']}")
        break
    elif event["event"] == "done":
        print("Streaming complete")
        break
    else:
        # Process content chunk
        print(event["data"], end="", flush=True)
```

### Integration with AsyncSimpleBot

This function is designed to work with `AsyncSimpleBot.stream_async()`:

```python
from llamabot import AsyncSimpleBot
from llamabot.sse import sse_stream

bot = AsyncSimpleBot("You are helpful.", stream_target="none")

# sse_stream internally calls bot.stream_async(*messages)
async for event in sse_stream(bot, ["Hello!"]):
    print(event)
```

### Client-Side Consumption

The browser `EventSource` API only supports **GET** by default and cannot send a JSON body. For a **POST** chat endpoint (as in the FastAPI example above), use `fetch` with `ReadableStream` and parse SSE frames, or expose a **GET** SSE route that reads session or query parameters.

Example using `fetch` (POST + streaming body):

```javascript
const response = await fetch("/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ messages: ["Hello!"] }),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();
let buffer = "";

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  buffer += decoder.decode(value, { stream: true });
  const lines = buffer.split("\n");
  buffer = lines.pop() ?? "";
  for (const line of lines) {
    if (line.startsWith("data:")) {
      console.log("data:", line.slice(5).trim());
    }
  }
}
```

For **named** events (`event: message`, `event: done`), extend the parser to accumulate `event:` and `data:` lines per SSE event (see the MDN documentation for the SSE wire format).

### See Also

- [Async streaming reference](streaming_async.md) - `stream_async` contract and helpers
- [SimpleBot API Reference](./bots/simplebot.md) - Sync chat bot; use `AsyncSimpleBot` for `stream_async`
- [SSE Streaming Example](../../examples/sse_streaming.py) - Marimo notebook (`uvx marimo run --sandbox docs/examples/sse_streaming.py`)
- [FastAPI SSE Example](../../../scripts/fastapi_sse_example.py) - JSON POST + ``EventSourceResponse`` only
- [FastAPI + HTMX + SSE demo](../../../scripts/async_simplebot_htmx_demo.py) - Browser UI (HTMX form + EventSource) using the same ``sse_stream`` / ``AsyncSimpleBot`` stack; static assets in ``scripts/async_simplebot_htmx_assets/``
