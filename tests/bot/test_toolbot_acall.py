"""Tests for :meth:`~llamabot.bot.toolbot.ToolBot.acall`."""

from __future__ import annotations

import json

import pytest
from litellm import ModelResponse

from llamabot.bot.toolbot import ToolBot
from llamabot.components.tools import tool


@tool
def demo_add(a: int, b: int) -> int:
    """Return *a* + *b*.

    :param a: First summand.
    :param b: Second summand.
    :return: Sum.
    """
    return a + b


@pytest.mark.asyncio
async def test_toolbot_acall_returns_tool_calls_from_mock() -> None:
    """acall awaits async completion and parses tool calls (mock_response, no network)."""
    args = json.dumps({"a": 1, "b": 2})
    mock = ModelResponse(
        id="mock",
        choices=[
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "demo_add",
                                "arguments": args,
                            },
                        }
                    ],
                }
            }
        ],
        created=0,
        model="mock",
        object="chat.completion",
    )

    bot = ToolBot(
        system_prompt="You pick tools.",
        model_name="gpt-4.1",
        tools=[demo_add],
        mock_response=mock,
        stream_target="none",
    )

    calls = await bot.acall("add 1 and 2")
    assert len(calls) == 1
    assert calls[0].function.name == "demo_add"
    assert json.loads(calls[0].function.arguments) == {"a": 1, "b": 2}


@pytest.mark.asyncio
async def test_decide_node_aexec_uses_acall(monkeypatch: pytest.MonkeyPatch) -> None:
    """DecideNode.aexec routes through ToolBot.acall."""
    from llamabot.components.pocketflow.nodes import DecideNode

    captured: dict = {}

    class FakeToolBot:
        """Minimal stand-in for :class:`~llamabot.bot.toolbot.ToolBot`.

        Constructor accepts arbitrary arguments and ignores them.

        :param args: Ignored.
        :param kwargs: Ignored.
        """

        def __init__(self, *args, **kwargs) -> None:
            pass

        async def acall(self, memory, **kwargs):
            """Record *memory* and return one ``demo_add`` tool call.

            :param memory: Same as :meth:`~llamabot.bot.toolbot.ToolBot.acall`.
            :param kwargs: Ignored.
            :return: List of one tool call.
            """
            captured["memory"] = memory
            from litellm import ChatCompletionMessageToolCall, Function

            return [
                ChatCompletionMessageToolCall(
                    id="x",
                    type="function",
                    function=Function(
                        name="demo_add",
                        arguments=json.dumps({"a": 1, "b": 2}),
                    ),
                )
            ]

    monkeypatch.setattr("llamabot.bot.toolbot.ToolBot", FakeToolBot)

    node = DecideNode(
        tools=[demo_add],
        system_prompt="sys",
        model_name="gpt-4.1",
    )
    prep = {"memory": ["hi"], "iteration_count": 1}
    name = await node.aexec(prep)
    assert name == "demo_add"
    assert prep["func_call"] == {"a": 1, "b": 2}
    assert captured["memory"] == ["hi"]
