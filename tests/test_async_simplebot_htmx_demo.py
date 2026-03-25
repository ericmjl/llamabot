"""Tests for :mod:`scripts.async_simplebot_htmx_demo` (no live LLM)."""

from __future__ import annotations

import re
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from scripts.async_simplebot_htmx_demo import create_app


@pytest.fixture
def mock_bot() -> MagicMock:
    """Return a bot whose ``stream_async`` yields predictable chunks."""
    bot = MagicMock()

    async def stream(*args, **kwargs):
        yield "chunk-a"
        yield "chunk-b"

    bot.stream_async = stream
    return bot


def test_index_serves_demo_page(mock_bot: MagicMock) -> None:
    """GET ``/`` returns HTML with HTMX."""
    client = TestClient(create_app(bot=mock_bot))
    response = client.get("/")
    assert response.status_code == 200
    assert "htmx.org" in response.text
    assert "async-simplebot-htmx" in response.text


def test_send_returns_user_row_and_stream_shell(mock_bot: MagicMock) -> None:
    """POST ``/send`` returns escaped user text and a ``data-sse-id`` target."""
    client = TestClient(create_app(bot=mock_bot))
    response = client.post("/send", data={"message": "hello"})
    assert response.status_code == 200
    assert "hello" in response.text
    assert re.search(r'data-sse-id="[0-9a-f-]{36}"', response.text)


def test_sse_streams_then_done(mock_bot: MagicMock) -> None:
    """GET ``/sse/{id}`` streams message events and ``done`` after POST ``/send``."""
    client = TestClient(create_app(bot=mock_bot))
    send_resp = client.post("/send", data={"message": "prompt"})
    match = re.search(r'data-sse-id="([^"]+)"', send_resp.text)
    assert match is not None
    stream_id = match.group(1)
    sse_resp = client.get(f"/sse/{stream_id}")
    assert sse_resp.status_code == 200
    body = sse_resp.text
    assert "event: message" in body
    assert "chunk-a" in body
    assert "chunk-b" in body
    assert "event: done" in body


def test_sse_unknown_id_returns_404(mock_bot: MagicMock) -> None:
    """GET ``/sse/{id}`` without a prior ``/send`` returns 404."""
    client = TestClient(create_app(bot=mock_bot))
    response = client.get("/sse/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404
