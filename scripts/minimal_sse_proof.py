"""Minimal runnable proof that ``AsyncSimpleBot`` + ``sse_stream`` work over HTTP.

Uses ``mock_response`` so no API key or network LLM is required.

Run from the repo root::

    pixi run python scripts/minimal_sse_proof.py

In a second terminal (port defaults to 9877; override with ``MINIMAL_SSE_PORT``)::

    curl -N http://127.0.0.1:9877/sse

You should see ``event: message`` lines with the mock text, then ``event: done``.
"""

import os

import uvicorn
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse

from llamabot import AsyncSimpleBot
from llamabot.sse import sse_stream

PORT = int(os.environ.get("MINIMAL_SSE_PORT", "9877"))

app = FastAPI(title="LlamaBot minimal SSE proof")

bot = AsyncSimpleBot(
    system_prompt="You are a test assistant.",
    model_name="gpt-4o-mini",
    mock_response="SSE proof OK",
    stream_target="none",
)


@app.get("/sse")
async def sse() -> EventSourceResponse:
    """Stream one assistant reply as SSE (no real provider call)."""
    return EventSourceResponse(sse_stream(bot, ["ping"]))


def main() -> None:
    """Print instructions and start uvicorn."""
    url = f"http://127.0.0.1:{PORT}/sse"
    print("LlamaBot minimal SSE proof (mock_response; no API key).")
    print(f"  curl -N {url}")
    print("  (set MINIMAL_SSE_PORT to use another port)")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
