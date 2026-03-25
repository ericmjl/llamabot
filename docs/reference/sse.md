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

- **bot**: A SimpleBot instance (or any object with a `stream_async()` method that accepts messages and returns an `AsyncGenerator[str, None]`).
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
from llamabot import SimpleBot
from llamabot.sse import sse_stream

app = FastAPI()
bot = SimpleBot("You are a helpful assistant.", stream_target="none")

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

### Integration with SimpleBot

This function is designed to work with `SimpleBot.stream_async()`:

```python
from llamabot import SimpleBot
from llamabot.sse import sse_stream

bot = SimpleBot("You are helpful.", stream_target="none")

# sse_stream internally calls bot.stream_async(*messages)
async for event in sse_stream(bot, ["Hello!"]):
    print(event)
```

### Client-Side Consumption

On the client side (JavaScript), consume the SSE stream like this:

```javascript
const eventSource = new EventSource('/chat', {
    method: 'POST',
    body: JSON.stringify({ messages: ["Hello!"] })
});

eventSource.addEventListener('message', (event) => {
    console.log('Content chunk:', event.data);
});

eventSource.addEventListener('done', (event) => {
    console.log('Streaming complete');
    eventSource.close();
});

eventSource.addEventListener('error', (event) => {
    console.error('Error:', event.data);
    eventSource.close();
});
```

### See Also

- [SimpleBot API Reference](./bots/simplebot.md#stream_async) - The `stream_async()` method
- [SSE Streaming Example](../../examples/sse_streaming.ipynb) - Interactive notebook
- [FastAPI SSE Example](../../../scripts/fastapi_sse_example.py) - Complete FastAPI example
