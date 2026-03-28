"""FastAPI + HTMX + SSE demo using :class:`~llamabot.bot.async_bots.AsyncSimpleBot`.

The page POSTs chat text with HTMX. The server echoes the user bubble and an
assistant container; a small external script opens ``EventSource`` to ``/sse/{id}``,
which streams tokens from :func:`~llamabot.sse.sse_stream` (backed by
``AsyncSimpleBot.stream_async``).

Run (from repository root)::

    pixi run uvicorn docs.examples.async_simplebot_htmx_demo:app --reload

Then open http://127.0.0.1:8000/ . Requires a working LiteLLM model (default
``ollama/phi3`` — adjust in ``create_app``).
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from llamabot import AsyncSimpleBot
from llamabot.sse import sse_stream

ASSETS_DIR = Path(__file__).resolve().parent / "async_simplebot_htmx_assets"


def create_app(bot: AsyncSimpleBot | None = None) -> FastAPI:
    """Build the demo FastAPI application.

    :param bot: Optional bot instance (for tests). When ``None``, a default
        :class:`~llamabot.bot.async_bots.AsyncSimpleBot` is constructed.
    :return: Configured :class:`~fastapi.FastAPI` app.
    """
    app = FastAPI(
        title="AsyncSimpleBot HTMX + SSE",
        description="Minimal chat UI: HTMX form + EventSource SSE streaming.",
    )

    app.mount(
        "/static/async-simplebot-htmx",
        StaticFiles(directory=ASSETS_DIR),
        name="static_demo",
    )
    templates = Jinja2Templates(directory=ASSETS_DIR)

    resolved_bot: Any = bot or AsyncSimpleBot(
        system_prompt="You are a helpful assistant. Be concise.",
        model_name="ollama/phi3",
        stream_target="none",
    )

    # One-shot pending prompts keyed by stream id (UUID). Consumed when ``/sse/{id}`` connects.
    pending: dict[str, str] = {}

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        """Serve the demo page."""
        return templates.TemplateResponse(request, "index.html")

    @app.post("/send", response_class=HTMLResponse)
    async def send(request: Request, message: str = Form(...)) -> HTMLResponse:
        """Append a user row plus an assistant shell; client opens SSE for the shell."""
        text = message.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Empty message")

        stream_id = str(uuid.uuid4())
        pending[stream_id] = text

        return templates.TemplateResponse(
            request,
            "_turn.html",
            {
                "user_message": text,
                "stream_id": stream_id,
            },
        )

    @app.get("/sse/{stream_id}")
    async def sse(stream_id: str) -> EventSourceResponse:
        """Stream assistant tokens for the pending message, then ``done``."""
        if stream_id not in pending:
            raise HTTPException(
                status_code=404,
                detail="Unknown or expired stream id (refresh and send again).",
            )
        user_text = pending.pop(stream_id)

        return EventSourceResponse(
            sse_stream(
                resolved_bot,
                [user_text],
                event_type="message",
                done_event="done",
            )
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
