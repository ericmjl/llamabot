"""FastAPI + HTMX + SSE demo using :class:`~llamabot.bot.toolbot.AsyncToolBot`.

The UI posts a natural-language request. The server uses a small custom SSE
stream (not generic :func:`~llamabot.sse.sse_stream`) because tool completions
often have **empty** ``delta.content``; the shared helper only forwards text
chunks. Here we stream assistant **text** when present, then append one JSON
block with the **assembled** tool call. The bot includes
:class:`~llamabot.components.tools.DEFAULT_TOOLS` plus demo tools
``reverse_text`` and ``word_count``.

Run (from repository root)::

    pixi run uvicorn docs.examples.async_toolbot_htmx_demo:app --reload

Then open http://127.0.0.1:8000/ . Use a model that supports function calling
(see ``DEFAULT_LANGUAGE_MODEL`` / :func:`~llamabot.config.default_language_model`).
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from litellm import ModelResponse, stream_chunk_builder

from llamabot import AsyncToolBot
from llamabot.bot.simplebot import (
    extract_content,
    extract_tool_calls,
    make_async_response,
    model_supports_token_streaming,
)
from llamabot.components.tools import tool
from llamabot.config import default_language_model

ASSETS_DIR = Path(__file__).resolve().parent / "async_toolbot_htmx_assets"

DEMO_SYSTEM_PROMPT = """You are the ToolBot HTMX demo. Your job is to choose
the single best function call for the user's request. Prefer the demo tools
`reverse_text` when the user wants text reversed, and `word_count` when they
ask how many words are in a phrase. You may use other available tools if they
clearly fit better. Output only valid tool calls as required by the API."""


@tool
def reverse_text(text: str) -> str:
    """Return *text* with character order reversed.

    Use when the user asks to reverse, flip, or spell backwards a string.

    :param text: Input string.
    :return: Reversed characters.
    """
    return text[::-1]


@tool
def word_count(text: str) -> int:
    """Return the number of whitespace-separated words in *text*.

    :param text: Input text.
    :return: Word count.
    """
    return len(text.split())


def tool_calls_json_payload(tool_calls: list) -> str:
    """Serialize tool calls for display in the demo panel.

    :param tool_calls: List from :func:`~llamabot.bot.simplebot.extract_tool_calls`.
    :return: Pretty-printed JSON string.
    """
    return json.dumps(
        [
            {"name": t.function.name, "arguments": t.function.arguments}
            for t in tool_calls
        ],
        indent=2,
    )


async def toolbot_sse_stream(
    bot: AsyncToolBot, user_text: str
) -> AsyncGenerator[dict[str, str], None]:
    """Yield SSE events including assistant text and tool-call deltas.

    :class:`~llamabot.bot.simplebot.stream_tokens_for_messages` only forwards
    ``delta.content``.     Tool-heavy completions often leave content empty; per-chunk ``tool_calls``
    deltas are noisy in the UI, so we stream **text** deltas only and append
    one assembled **tool call** JSON block after the stream completes.

    :param bot: Configured async tool bot.
    :param user_text: User message to pass to :meth:`~llamabot.bot.toolbot.ToolBot.compose_tool_messages`.
    :return: Async generator of ``sse-starlette`` event dictionaries.
    """
    try:
        message_list, _ = bot.compose_tool_messages(user_text)
        stream = model_supports_token_streaming(bot.model_name)
        response = await make_async_response(bot, message_list, stream=stream)

        chunks: list = []

        if isinstance(response, ModelResponse):
            text = extract_content(response) or ""
            tools = extract_tool_calls(response)
            parts: list[str] = []
            if text.strip():
                parts.append(text)
            if tools:
                parts.append(tool_calls_json_payload(tools))
            if parts:
                yield {"event": "message", "data": "\n\n".join(parts)}
            yield {"event": "done", "data": ""}
            return

        text_emitted = False
        async for chunk in response:
            chunks.append(chunk)
            choice = chunk.choices[0]
            delta_obj = getattr(choice, "delta", None)
            if delta_obj is None:
                continue
            if isinstance(delta_obj, dict):
                delta = delta_obj.get("content")
            else:
                delta = getattr(delta_obj, "content", None)
            if delta:
                text_emitted = True
                yield {"event": "message", "data": delta}

        final = stream_chunk_builder(chunks)
        text = extract_content(final) or ""
        tools = extract_tool_calls(final)
        if not text_emitted and (text.strip() or tools):
            parts: list[str] = []
            if text.strip():
                parts.append(text)
            if tools:
                parts.append(tool_calls_json_payload(tools))
            if parts:
                yield {"event": "message", "data": "\n\n".join(parts)}
        elif tools:
            yield {
                "event": "message",
                "data": "\n\n--- tool call ---\n\n" + tool_calls_json_payload(tools),
            }
        yield {"event": "done", "data": ""}
    except Exception as e:
        yield {"event": "error", "data": str(e)}


def build_default_bot() -> AsyncToolBot:
    """Construct an :class:`AsyncToolBot` with demo tools and no streaming to stdout.

    :return: Bot wired for SSE-only streaming.
    """
    return AsyncToolBot(
        system_prompt=DEMO_SYSTEM_PROMPT,
        model_name=default_language_model(),
        tools=[reverse_text, word_count],
        stream_target="none",
    )


def create_app(bot: AsyncToolBot | None = None) -> FastAPI:
    """Build the demo FastAPI application.

    :param bot: Optional :class:`AsyncToolBot`. When ``None``, :func:`build_default_bot` is used.
    :return: Configured :class:`~fastapi.FastAPI` application.
    """
    app = FastAPI(
        title="AsyncToolBot HTMX + SSE",
        description="Tool-selection streaming: HTMX form + EventSource over sse_stream.",
    )

    app.mount(
        "/static/async-toolbot-htmx",
        StaticFiles(directory=ASSETS_DIR),
        name="static_demo",
    )
    templates = Jinja2Templates(directory=ASSETS_DIR)

    resolved: Any = bot or build_default_bot()

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
        """Stream model tokens (tool-call JSON) for the pending message, then ``done``."""
        if stream_id not in pending:
            raise HTTPException(
                status_code=404,
                detail="Unknown or expired stream id (refresh and send again).",
            )
        user_text = pending.pop(stream_id)

        return EventSourceResponse(toolbot_sse_stream(resolved, user_text))

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
