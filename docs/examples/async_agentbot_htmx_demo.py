"""FastAPI + HTMX + SSE demo using :class:`~llamabot.bot.agentbot.AgentBot`.

:class:`~llamabot.bot.agentbot.AgentBot` runs a PocketFlow graph (decide → tool → …)
synchronously. There is no token stream from the graph itself, so this demo
streams a short status line, then the **final result** and a **memory trace**
after :meth:`~llamabot.bot.agentbot.AgentBot.__call__` finishes (via
:func:`asyncio.to_thread`).

Run (from repository root)::

    pixi run uvicorn docs.examples.async_agentbot_htmx_demo:app --reload

Then open http://127.0.0.1:8000/ . Uses :func:`~llamabot.config.default_language_model`
and a small demo tool (``inch_to_cm``). Set ``DEFAULT_LANGUAGE_MODEL`` / API keys
as for other LlamaBot apps.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator, Callable
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from llamabot.bot.agentbot import AgentBot
from llamabot.components.tools import tool
from llamabot.config import default_language_model

ASSETS_DIR = Path(__file__).resolve().parent / "async_agentbot_htmx_assets"

AGENT_DEMO_SYSTEM_PROMPT = """You are a small HTMX demo agent using PocketFlow.
Use tools when they fit the user's request, then call respond_to_user with a
brief final message for the user. For inch/cm conversion questions, use inch_to_cm."""


@tool
def inch_to_cm(inches: float) -> float:
    """Convert inches to centimeters.

    Use when the user asks to convert a length from inches to cm or millimeters
    (approximate via cm).

    :param inches: Length in inches.
    :return: Equivalent length in centimeters, rounded to two decimals.
    """
    return round(inches * 2.54, 2)


def build_default_agentbot() -> AgentBot:
    """Build a demo :class:`AgentBot` with a single extra tool and a step cap.

    :return: New agent instance (call once per user request in the SSE handler).
    """
    return AgentBot(
        tools=[inch_to_cm],
        system_prompt=AGENT_DEMO_SYSTEM_PROMPT,
        model_name=default_language_model(),
        max_iterations=8,
    )


async def agentbot_sse_stream(
    bot: AgentBot, query: str
) -> AsyncGenerator[dict[str, str], None]:
    """Run the agent in a worker thread and stream result text over SSE.

    :param bot: Fresh :class:`AgentBot` instance for this request.
    :param query: User message.
    :return: Async generator of sse-starlette event dicts.
    """
    try:
        yield {
            "event": "message",
            "data": "Running AgentBot (PocketFlow)…\n\n",
        }
        result = await asyncio.to_thread(bot, query)
        parts: list[str] = ["--- result ---\n\n"]
        if result is not None:
            parts.append(str(result))
        else:
            parts.append("(no result; check memory trace below)")
        mem = bot.shared.get("memory", [])
        if mem:
            parts.append("\n\n--- memory trace ---\n\n")
            parts.append("\n".join(str(m) for m in mem))
        yield {"event": "message", "data": "".join(parts)}
        yield {"event": "done", "data": ""}
    except Exception as e:
        yield {"event": "error", "data": str(e)}


def create_app(
    bot_factory: Callable[[], AgentBot] | None = None,
) -> FastAPI:
    """Build the demo FastAPI application.

    :param bot_factory: Returns a new :class:`AgentBot` per request. When
        ``None``, :func:`build_default_agentbot` is used.
    :return: Configured :class:`~fastapi.FastAPI` application.
    """
    app = FastAPI(
        title="AgentBot HTMX + SSE",
        description="PocketFlow agent: status + result stream after each run.",
    )

    app.mount(
        "/static/async-agentbot-htmx",
        StaticFiles(directory=ASSETS_DIR),
        name="static_demo",
    )
    templates = Jinja2Templates(directory=ASSETS_DIR)

    factory: Callable[[], AgentBot] = bot_factory or build_default_agentbot

    pending: dict[str, str] = {}

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        """Serve the demo page."""
        return templates.TemplateResponse(request, "index.html")

    @app.post("/send", response_class=HTMLResponse)
    async def send(request: Request, message: str = Form(...)) -> HTMLResponse:
        """Return a user bubble and assistant shell; client opens SSE for the shell."""
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
        """Stream status, then agent result and memory trace."""
        if stream_id not in pending:
            raise HTTPException(
                status_code=404,
                detail="Unknown or expired stream id (refresh and send again).",
            )
        user_text = pending.pop(stream_id)
        bot = factory()
        return EventSourceResponse(agentbot_sse_stream(bot, user_text))

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
