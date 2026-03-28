"""FastAPI + HTMX + SSE demo using :class:`~llamabot.bot.async_bots.AsyncStructuredBot`.

The page accepts a free-form career description. The server runs
:class:`~llamabot.bot.async_bots.AsyncStructuredBot` to extract structured
fields (name, e-mail, skills, …) and streams the raw JSON via SSE.  When the
stream finishes the client-side script parses the accumulated JSON and
populates the read-only profile form.

Run (from repository root)::

    pixi run uvicorn docs.examples.async_structuredbot_htmx_demo:app --reload

Then open http://127.0.0.1:8000/ .  Requires a working LiteLLM model (default
``ollama_chat/phi3`` for local Ollama — adjust ``model_name`` in :func:`create_app`).
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, List

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator
from sse_starlette.sse import EventSourceResponse

from llamabot.bot.async_bots import AsyncStructuredBot
from llamabot.sse import sse_stream

ASSETS_DIR = Path(__file__).resolve().parent / "async_structuredbot_htmx_assets"

SYSTEM_PROMPT = """You are a career-profile extractor.
Given a free-form career description, return a valid JSON object matching the
schema you will be provided. Be conservative: if information is absent use
sensible defaults (empty string / empty list / 0).
Return ONLY the JSON object — no prose, no markdown fences."""


class JobProfile(BaseModel):
    """Structured job-applicant profile extracted from free-form text."""

    full_name: str = Field(description="Full name of the applicant.")
    email: str = Field(
        default="",
        description="Email address. Empty string if not mentioned.",
    )
    phone: str = Field(
        default="",
        description="Phone number. Empty string if not mentioned.",
    )
    years_of_experience: int = Field(
        default=0,
        ge=0,
        le=60,
        description="Total years of professional experience. 0 if not mentioned.",
    )
    top_skills: List[str] = Field(
        description="Between 1 and 6 key technical or soft skills.",
    )
    desired_role: str = Field(
        description="The role or job title the applicant is targeting.",
    )
    professional_summary: str = Field(
        description=(
            "A 2-3 sentence professional summary written in third person, "
            "suitable for a résumé."
        ),
    )

    @field_validator("top_skills")
    @classmethod
    def validate_skills_count(cls, v: List[str]) -> List[str]:
        """Ensure between 1 and 6 skills are present.

        :param v: List of skill strings.
        :return: Validated list.
        :raises ValueError: When the count is outside [1, 6].
        """
        if not (1 <= len(v) <= 6):
            raise ValueError("top_skills must contain between 1 and 6 items")
        return v


def create_app(bot: AsyncStructuredBot | None = None) -> FastAPI:
    """Build the demo FastAPI application.

    :param bot: Optional bot instance (for tests / custom models). When
        ``None``, a default :class:`~llamabot.bot.async_bots.AsyncStructuredBot`
        targeting ``ollama_chat/phi3`` is constructed.
    :return: Configured :class:`~fastapi.FastAPI` application.
    """
    app = FastAPI(
        title="AsyncStructuredBot HTMX + SSE",
        description=(
            "Form-filling demo: paste a career blurb, " "watch fields populate via SSE."
        ),
    )

    app.mount(
        "/static/async-structuredbot-htmx",
        StaticFiles(directory=ASSETS_DIR),
        name="static_demo",
    )
    templates = Jinja2Templates(directory=ASSETS_DIR)

    resolved_bot: Any = bot or AsyncStructuredBot(
        system_prompt=SYSTEM_PROMPT,
        pydantic_model=JobProfile,
        model_name="ollama_chat/phi3",
        stream_target="none",
    )

    # Pending prompts keyed by UUID stream id; consumed on first SSE connect.
    pending: dict[str, str] = {}

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        """Serve the main demo page."""
        return templates.TemplateResponse(request, "index.html")

    @app.post("/extract", response_class=HTMLResponse)
    async def extract(request: Request, description: str = Form(...)) -> HTMLResponse:
        """Store the pending description and return an SSE-ready panel fragment.

        :param request: Incoming HTTP request (needed by Jinja2Templates).
        :param description: Free-form career text from the HTML form.
        :return: HTML fragment containing the stream panel with the new stream id.
        :raises HTTPException: 400 when *description* is blank.
        """
        text = description.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Empty description")

        stream_id = str(uuid.uuid4())
        pending[stream_id] = text

        return templates.TemplateResponse(
            request,
            "_stream_panel.html",
            {"stream_id": stream_id},
        )

    @app.get("/sse/{stream_id}")
    async def sse_endpoint(stream_id: str) -> EventSourceResponse:
        """Stream structured JSON tokens for the pending description.

        Events:
        - ``token`` — one text chunk of the JSON being built
        - ``done``  — stream finished; client should parse and render

        :param stream_id: UUID issued by :func:`extract`.
        :return: SSE response that streams JSON tokens from the bot.
        :raises HTTPException: 404 when *stream_id* is unknown or already consumed.
        """
        if stream_id not in pending:
            raise HTTPException(
                status_code=404,
                detail="Unknown or expired stream id (refresh and try again).",
            )
        user_text = pending.pop(stream_id)

        return EventSourceResponse(
            sse_stream(
                resolved_bot,
                [user_text],
                event_type="token",
                done_event="done",
            )
        )

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
