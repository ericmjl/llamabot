"""Tests for :mod:`llamabot.bot.async_agentbot`."""

from __future__ import annotations

import pytest
from pocketflow import AsyncFlow

from llamabot.bot.async_agentbot import AsyncAgentBot, AsyncDecideNode
from llamabot.components.tools import tool


@tool
def demo_echo(text: str) -> str:
    """Echo *text* for AsyncAgentBot wiring tests.

    :param text: Input string.
    :return: Same string.
    """
    return text


def test_async_agentbot_flow_is_asyncflow() -> None:
    """AsyncAgentBot exposes an :class:`~pocketflow.AsyncFlow`."""
    bot = AsyncAgentBot(tools=[demo_echo])
    assert isinstance(bot.flow, AsyncFlow)


def test_async_decide_and_func_nodes_wrap_sync() -> None:
    """Wrappers reference inner PocketFlow nodes."""
    bot = AsyncAgentBot(tools=[demo_echo])
    assert isinstance(bot.flow.start_node, AsyncDecideNode)
    assert bot.flow.start_node._inner is bot.decide_node


@pytest.mark.asyncio
async def test_arun_awaits_flow_run_async() -> None:
    """:meth:`AsyncAgentBot.arun` awaits :meth:`~pocketflow.AsyncFlow.run_async`."""
    bot = AsyncAgentBot(tools=[demo_echo])

    async def fake_run_async(shared: dict) -> object:
        """Stub :meth:`~pocketflow.AsyncFlow.run_async` for the test.

        :param shared: PocketFlow shared store.
        :return: ``None`` after setting ``result``.
        """
        shared["result"] = "mocked"
        return None

    bot.flow.run_async = fake_run_async  # type: ignore[method-assign]

    result = await bot.arun("ping")
    assert result == "mocked"
