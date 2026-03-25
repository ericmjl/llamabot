"""FastAPI example demonstrating SSE streaming with LlamaBot.

This example shows how to use LlamaBot's SSE streaming capabilities
to create a real-time chat API endpoint.

Run with:
    uvicorn scripts.fastapi_sse_example:app --reload

Then visit http://localhost:8000/docs to test the endpoint.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from llamabot import AsyncSimpleBot
from llamabot.sse import sse_stream

app = FastAPI(title="LlamaBot SSE Streaming Example")

# Initialize the bot
bot = AsyncSimpleBot(
    system_prompt="You are a helpful assistant. Be concise and friendly.",
    model_name="ollama/phi3",  # Change to your preferred model
    stream_target="none",  # We handle streaming via SSE
)


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    messages: list[str]
    """List of messages to send to the bot."""


@app.get("/")
async def root():
    """Root endpoint with instructions."""
    return {
        "message": "LlamaBot SSE Streaming Example",
        "endpoints": {
            "/chat": "POST - Stream chat responses via SSE",
            "/docs": "GET - API documentation",
        },
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    """Stream chat responses using Server-Sent Events (SSE).

    This endpoint accepts a list of messages and streams the bot's
    response in real-time using SSE format.

    Example request:
        POST /chat
        {
            "messages": ["Hello!", "How are you?"]
        }

    The response is streamed as SSE events:
        event: message
        data: Hello

        event: message
        data:  there

        event: done
        data:

    If an error occurs:
        event: error
        data: Error message here
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty")

    # Use sse_stream to convert bot streaming to SSE format
    return EventSourceResponse(
        sse_stream(bot, request.messages, event_type="message", done_event="done")
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
