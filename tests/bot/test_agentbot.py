"""Test suite for AgentBot using native tool_calls (no ToolBot)."""

import pytest
from unittest.mock import MagicMock

from llamabot.bot.agentbot import AgentBot, hash_result
from llamabot.components.tools import tool
from llamabot.components.messages import HumanMessage


@tool
def echo_tool(text: str) -> str:
    """Echo back the provided text (test helper tool)."""
    return text


def _mk_response(content: str | None = None, tool_calls: list | None = None):
    """Create a MagicMock response object with optional content and tool_calls."""
    resp = MagicMock()
    choice = MagicMock()
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    choice.message = msg
    resp.choices = [choice]
    return resp


def _mk_tool_call(name: str, args: dict, call_id: str = "call_1"):
    """Create a MagicMock tool_call with function name, args and id."""
    call = MagicMock()
    func = MagicMock()
    func.name = name
    func.arguments = json_dumps(args)
    call.function = func
    call.id = call_id
    return call


def json_dumps(d: dict) -> str:
    """JSON-dump helper to keep arguments as strings in mocks."""
    import json

    return json.dumps(d)


def test_hash_result():
    """Hashing is stable and order-insensitive for dicts."""
    assert hash_result("test") == hash_result("test")
    assert len(hash_result("test")) == 8
    assert hash_result({"a": 1, "b": 2}) == hash_result({"b": 2, "a": 1})


def test_openai_like_tool_then_final(monkeypatch):
    """Model first returns tool_calls then a final assistant message."""
    bot = AgentBot(system_prompt="You are a helper.")

    # Turn 1: tool_calls only
    tcall = _mk_tool_call("today_date", {})
    r1 = _mk_response(content="", tool_calls=[tcall])
    # Turn 2: final content
    r2 = _mk_response(content="2025-10-29", tool_calls=None)

    calls = iter([r1, r2])
    monkeypatch.setattr("llamabot.bot.agentbot.completion", lambda **_: next(calls))

    out = bot("What's today's date?")
    assert out.content == "2025-10-29"


def test_ollama_like_tool_then_final(monkeypatch):
    """Ollama-like flow: tool_calls with empty content then final content."""
    bot = AgentBot(system_prompt="You are a helper.")

    # First return tool_calls with empty content, then final text
    tcall = _mk_tool_call("today_date", {}, call_id="call_xyz")
    r1 = _mk_response(content="", tool_calls=[tcall])
    r2 = _mk_response(content="2025-10-29", tool_calls=None)
    calls = iter([r1, r2])
    monkeypatch.setattr("llamabot.bot.agentbot.completion", lambda **_: next(calls))

    out = bot("date?")
    assert out.content == "2025-10-29"


def test_no_tool_path(monkeypatch):
    """Model responds directly without any tool_calls."""
    bot = AgentBot(system_prompt="You are a helper.")

    # Model answers directly with no tool_calls
    r1 = _mk_response(content="hi", tool_calls=None)
    monkeypatch.setattr("llamabot.bot.agentbot.completion", lambda **_: r1)

    out = bot("say hi")
    assert out.content == "hi"


def test_max_cycles(monkeypatch):
    """Exceeding max_iterations raises a RuntimeError."""
    bot = AgentBot(system_prompt="You are a helper.")
    # Keep returning a tool_call to force loop until limit
    tcall = _mk_tool_call("today_date", {})
    r = _mk_response(content="", tool_calls=[tcall])

    def _resp(**kwargs):
        """Return the same mocked response to force repeated tool_calls."""
        return r

    monkeypatch.setattr("llamabot.bot.agentbot.completion", _resp)
    with pytest.raises(RuntimeError):
        bot("loop please", max_iterations=2)


def test_memory_appends_user(monkeypatch):
    """User message is appended to memory before model call."""
    from llamabot.components.chat_memory import ChatMemory

    mem = MagicMock(spec=ChatMemory)
    bot = AgentBot(system_prompt="You are a helper.", memory=mem)

    r1 = _mk_response(content="hello", tool_calls=None)
    monkeypatch.setattr("llamabot.bot.agentbot.completion", lambda **_: r1)

    out = bot("hi")
    assert out.content == "hello"
    assert mem.append.called
    # first append should be last user msg
    first_appended = mem.append.call_args_list[0][0][0]
    assert isinstance(first_appended, HumanMessage)
    assert first_appended.content == "hi"


def test_qwen25_simple_date(monkeypatch):
    """Simple execution with qwen2.5:0.5b: tool then final content."""
    bot = AgentBot(
        system_prompt="You are a helper.", model_name="ollama_chat/qwen2.5:0.5b"
    )

    # Turn 1: tool_calls only
    tcall = _mk_tool_call("today_date", {})
    r1 = _mk_response(content="", tool_calls=[tcall])
    # Turn 2: final content
    r2 = _mk_response(content="2025-10-29", tool_calls=None)

    calls = iter([r1, r2])
    monkeypatch.setattr("llamabot.bot.agentbot.completion", lambda **_: next(calls))

    out = bot("What's today's date?")
    assert out.content == "2025-10-29"


@tool
def run_shell_command(command: str) -> str:
    """Return a fake directory listing for testing the shell tool path.

    :param command: Command string (ignored in test)
    :return: Mocked output
    """
    return "README.md\npyproject.toml\nllamabot/\ntests/\n"


def test_qwen25_with_shell_tool(monkeypatch):
    """qwen2.5:0.5b flow: tool call then final summarized content."""
    bot = AgentBot(
        system_prompt="You are a helper.",
        model_name="ollama_chat/qwen2.5:0.5b",
        tools=[run_shell_command],
    )

    # Turn 1: decide to call run_shell_command
    tcall = _mk_tool_call("run_shell_command", {"command": "ls"})
    r1 = _mk_response(content="", tool_calls=[tcall])
    # Turn 2: return final text referencing files
    final_text = "Found files: README.md, pyproject.toml, llamabot/, tests/."
    r2 = _mk_response(content=final_text, tool_calls=None)

    calls = iter([r1, r2])
    monkeypatch.setattr("llamabot.bot.agentbot.completion", lambda **_: next(calls))

    out = bot("What files are here? Use the shell tool.")
    assert "README.md" in out.content
    assert "pyproject.toml" in out.content
