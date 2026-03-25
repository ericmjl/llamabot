"""Tests for async streaming helpers and Async* bot classes."""

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from llamabot.bot.async_bots import (
    AsyncQueryBot,
    AsyncSimpleBot,
    AsyncStructuredBot,
    AsyncToolBot,
)
from llamabot.bot.simplebot import (
    SimpleBot,
    completion_kwargs_for_messages,
    model_supports_token_streaming,
)
from llamabot.components.messages import HumanMessage
from llamabot.sse import sse_stream


def test_model_supports_token_streaming() -> None:
    """O1 models disable token streaming in LlamaBot."""
    assert model_supports_token_streaming("gpt-4o-mini") is True
    assert model_supports_token_streaming("o1-preview") is False
    assert model_supports_token_streaming("o1-mini") is False


def test_completion_kwargs_for_messages_basic() -> None:
    """completion_kwargs_for_messages builds LiteLLM kwargs."""
    bot = SimpleBot("system text", model_name="gpt-4o-mini")
    messages = [bot.system_prompt, HumanMessage(content="hi")]
    kwargs = completion_kwargs_for_messages(bot, messages, stream=True)
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["stream"] is True
    assert len(kwargs["messages"]) == 2


@pytest.mark.asyncio
async def test_simplebot_stream_async_mock_response() -> None:
    """stream_async yields deltas that assemble to the mock response."""
    bot = AsyncSimpleBot(
        system_prompt="Test",
        model_name="gpt-4o-mini",
        mock_response="hello world",
        stream_target="none",
    )
    chunks: list[str] = []
    async for delta in bot.stream_async("hi"):
        chunks.append(delta)
    assert "".join(chunks) == "hello world"


@pytest.mark.asyncio
async def test_sse_stream_with_simplebot_stream_async() -> None:
    """sse_stream drives AsyncSimpleBot.stream_async via mock_response."""
    bot = AsyncSimpleBot(
        system_prompt="Test",
        model_name="gpt-4o-mini",
        mock_response="ab",
        stream_target="none",
    )
    events: list[dict[str, str]] = []
    async for event in sse_stream(bot, ["x"]):
        events.append(event)
    assert events[-1]["event"] == "done"
    data_parts = [e["data"] for e in events if e["event"] == "message"]
    assert "".join(data_parts) == "ab"


class StreamSchema(BaseModel):
    """Minimal schema for AsyncStructuredBot.stream_async tests."""

    value: str


@pytest.mark.asyncio
async def test_toolbot_stream_async_mock_response() -> None:
    """AsyncToolBot.stream_async yields assembled mock content."""
    bot = AsyncToolBot(
        system_prompt="You choose tools.",
        model_name="gpt-4o-mini",
        mock_response="[]",
        stream_target="none",
    )
    chunks: list[str] = []
    async for delta in bot.stream_async("hello"):
        chunks.append(delta)
    assert "".join(chunks) == "[]"


@pytest.mark.asyncio
async def test_querybot_stream_async_mock_response() -> None:
    """AsyncQueryBot.stream_async uses docstore retrieval then streams."""
    docstore = MagicMock()
    docstore.retrieve.return_value = set()
    bot = AsyncQueryBot(
        system_prompt="You answer from docs.",
        docstore=docstore,
        model_name="gpt-4o-mini",
        mock_response="rag answer",
        stream_target="none",
    )
    chunks: list[str] = []
    async for delta in bot.stream_async("what is up"):
        chunks.append(delta)
    assert "".join(chunks) == "rag answer"
    docstore.retrieve.assert_called()


@pytest.mark.asyncio
async def test_structuredbot_stream_async_mock_response() -> None:
    """AsyncStructuredBot.stream_async streams one structured completion attempt."""
    bot = AsyncStructuredBot(
        system_prompt="Return JSON matching the schema.",
        pydantic_model=StreamSchema,
        model_name="gpt-4o-mini",
        mock_response='{"value": "ok"}',
        stream_target="none",
    )
    chunks: list[str] = []
    async for delta in bot.stream_async("give me data"):
        chunks.append(delta)
    assert "".join(chunks) == '{"value": "ok"}'
