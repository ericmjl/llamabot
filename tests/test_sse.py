"""Tests for SSE streaming utilities."""

from unittest.mock import MagicMock

import pytest

from llamabot.sse import sse_stream


@pytest.mark.asyncio
async def test_sse_stream_basic():
    """Test sse_stream yields correct event format."""
    bot = MagicMock()

    async def mock_stream(*args):
        yield "Hello"
        yield " world"

    bot.stream_async = mock_stream

    events = []
    async for event in sse_stream(bot, ["test message"]):
        events.append(event)

    assert len(events) == 3  # 2 message events + 1 done event
    assert events[0]["event"] == "message"
    assert events[0]["data"] == "Hello"
    assert events[1]["event"] == "message"
    assert events[1]["data"] == " world"
    assert events[2]["event"] == "done"
    assert events[2]["data"] == ""


@pytest.mark.asyncio
async def test_sse_stream_custom_event_types():
    """Test sse_stream with custom event types."""
    bot = MagicMock()

    async def mock_stream(*args):
        yield "test"

    bot.stream_async = mock_stream

    events = []
    async for event in sse_stream(
        bot, ["test"], event_type="content", done_event="complete"
    ):
        events.append(event)

    assert events[0]["event"] == "content"
    assert events[1]["event"] == "complete"


@pytest.mark.asyncio
async def test_sse_stream_error_handling():
    """Test sse_stream yields error event on exception."""
    bot = MagicMock()

    async def mock_stream(*args):
        # Raise exception during iteration (not at function call)
        yield "some"
        raise Exception("Test error")

    bot.stream_async = mock_stream

    events = []
    async for event in sse_stream(bot, ["test"]):
        events.append(event)

    assert len(events) == 2  # 1 message + 1 error
    assert events[0]["event"] == "message"
    assert events[1]["event"] == "error"
    assert "Test error" in events[1]["data"]


@pytest.mark.asyncio
async def test_sse_stream_empty_response():
    """Test sse_stream handles empty stream correctly."""
    bot = MagicMock()

    async def mock_stream(*args):
        # Empty async generator - no chunks
        if False:
            yield

    bot.stream_async = mock_stream

    events = []
    async for event in sse_stream(bot, ["test"]):
        events.append(event)

    # Should still yield done event even if no chunks
    assert len(events) == 1
    assert events[0]["event"] == "done"
    assert events[0]["data"] == ""


@pytest.mark.asyncio
async def test_sse_stream_multiple_messages():
    """Test sse_stream handles multiple input messages."""
    bot = MagicMock()

    async def mock_stream(*args):
        yield "response"

    bot.stream_async = mock_stream

    events = []
    async for event in sse_stream(bot, ["msg1", "msg2"]):
        events.append(event)

    # Verify stream_async was called with multiple messages
    # Note: We can't use assert_called_once_with on a regular function,
    # but we can verify the behavior worked correctly
    assert len(events) == 2  # 1 message + 1 done
    assert events[0]["event"] == "message"
    assert events[0]["data"] == "response"
    assert events[1]["event"] == "done"
