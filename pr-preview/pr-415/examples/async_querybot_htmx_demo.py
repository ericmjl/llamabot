"""FastAPI + HTMX + SSE demo using :class:`~llamabot.bot.querybot.AsyncQueryBot`.

Documents are loaded from ``async_querybot_htmx_assets/corpus/*.md`` into a
:class:`~llamabot.components.docstore.BM25DocStore` (lexical retrieval, no
embedding download). Each question triggers retrieval plus streaming via
:func:`~llamabot.sse.sse_stream`.

Run (from repository root)::

    pixi run uvicorn docs.examples.async_querybot_htmx_demo:app --reload

Then open http://127.0.0.1:8000/ . Requires a working LiteLLM model (default
``ollama/phi3`` — adjust ``model_name`` in :func:`create_app`).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from llamabot import AsyncQueryBot
from llamabot.components.docstore import BM25DocStore
from llamabot.sse import sse_stream

ASSETS_DIR = Path(__file__).resolve().parent / "async_querybot_htmx_assets"
CORPUS_DIR = ASSETS_DIR / "corpus"

SYSTEM_PROMPT = """You are answering questions about the LlamaBot Python library
using only the retrieved context passages shown with the user message.
If the passages do not contain the answer, say so briefly and suggest a
rephrased question. Keep answers short and clear."""


def load_corpus_documents(corpus_dir: Path) -> list[str]:
    """Read every ``*.md`` file in *corpus_dir* as UTF-8 text.

    :param corpus_dir: Directory containing markdown shards for BM25.
    :return: List of document strings, one per file.
    :raises FileNotFoundError: When *corpus_dir* does not exist.
    :raises ValueError: When no ``*.md`` files are found.
    """
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")
    paths = sorted(corpus_dir.glob("*.md"))
    if not paths:
        raise ValueError(f"No .md files under {corpus_dir}")
    return [p.read_text(encoding="utf-8") for p in paths]


class QuerySseAdapter:
    """Adapts :class:`AsyncQueryBot` for :func:`~llamabot.sse.sse_stream` with a fixed ``n_results``."""

    def __init__(self, bot: AsyncQueryBot, n_results: int = 8) -> None:
        self._bot = bot
        self._n_results = n_results

    async def stream_async(self, *messages: str) -> AsyncGenerator[str, None]:
        """Stream assistant tokens for the first string in *messages*.

        :param messages: Query strings; only the first is used (same contract as other async bots).
        :return: Async generator of text chunks.
        """
        if not messages:
            raise ValueError("SSE stream requires at least one message")
        query = messages[0]
        async for chunk in self._bot.stream_async(query, n_results=self._n_results):
            yield chunk


def build_default_bot(model_name: str = "ollama/phi3") -> QuerySseAdapter:
    """Construct a :class:`QuerySseAdapter` over a fresh BM25 store and corpus.

    :param model_name: LiteLLM model id passed to :class:`AsyncQueryBot`.
    :return: Adapter ready for :func:`~llamabot.sse.sse_stream`.
    """
    texts = load_corpus_documents(CORPUS_DIR)
    store = BM25DocStore()
    store.extend(texts)
    inner = AsyncQueryBot(
        system_prompt=SYSTEM_PROMPT,
        docstore=store,
        model_name=model_name,
        stream_target="none",
    )
    return QuerySseAdapter(inner, n_results=8)


def create_app(stream_bot: QuerySseAdapter | None = None) -> FastAPI:
    """Build the demo FastAPI application.

    :param stream_bot: Optional adapter wrapping :class:`AsyncQueryBot`. When
        ``None``, a default bot is built from the bundled corpus.
    :return: Configured :class:`~fastapi.FastAPI` application.
    """
    app = FastAPI(
        title="AsyncQueryBot HTMX + SSE",
        description=(
            "RAG chat demo: questions stream via SSE after BM25 retrieval "
            "over bundled markdown docs."
        ),
    )

    app.mount(
        "/static/async-querybot-htmx",
        StaticFiles(directory=ASSETS_DIR),
        name="static_demo",
    )
    templates = Jinja2Templates(directory=ASSETS_DIR)

    resolved: Any = stream_bot or build_default_bot()

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
        """Stream assistant tokens for the pending message, then ``done``."""
        if stream_id not in pending:
            raise HTTPException(
                status_code=404,
                detail="Unknown or expired stream id (refresh and send again).",
            )
        user_text = pending.pop(stream_id)

        return EventSourceResponse(
            sse_stream(
                resolved,
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
