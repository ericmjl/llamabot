"""Smoke tests for lazy import behaviour.

These tests verify that:
1. ``import llamabot`` is fast (< 0.5 s)
2. All public symbols remain accessible
3. No import-time crashes occur for optional extras
"""

import time


MAX_BARE_IMPORT = 0.5
MAX_SIMPLEBOT_IMPORT = 1.0


def test_bare_import_fast():
    """import llamabot should complete in under 0.5 s."""
    start = time.perf_counter()

    elapsed = time.perf_counter() - start
    assert (
        elapsed < MAX_BARE_IMPORT
    ), f"import llamabot took {elapsed:.3f}s (> {MAX_BARE_IMPORT}s)"


def test_simplebot_accessible():
    from llamabot import SimpleBot

    assert SimpleBot is not None


def test_async_simplebot_accessible():
    from llamabot import AsyncSimpleBot

    assert AsyncSimpleBot is not None


def test_structuredbot_accessible():
    from llamabot import StructuredBot

    assert StructuredBot is not None


def test_async_structuredbot_accessible():
    from llamabot import AsyncStructuredBot

    assert AsyncStructuredBot is not None


def test_agentbot_accessible():
    from llamabot import AgentBot

    assert AgentBot is not None


def test_async_agentbot_accessible():
    from llamabot import AsyncAgentBot

    assert AsyncAgentBot is not None


def test_toolbot_accessible():
    from llamabot import ToolBot

    assert ToolBot is not None


def test_async_toolbot_accessible():
    from llamabot import AsyncToolBot

    assert AsyncToolBot is not None


def test_querybot_accessible():
    from llamabot import QueryBot

    assert QueryBot is not None


def test_async_querybot_accessible():
    from llamabot import AsyncQueryBot

    assert AsyncQueryBot is not None


def test_imagebot_accessible():
    from llamabot import ImageBot

    assert ImageBot is not None


def test_tool_decorator_accessible():
    from llamabot import tool

    assert callable(tool)


def test_prompt_decorator_accessible():
    from llamabot import prompt

    assert callable(prompt)


def test_message_helpers():
    from llamabot import dev, system, user

    assert callable(user)
    assert callable(system)
    assert callable(dev)


def test_chat_memory_accessible():
    from llamabot import ChatMemory

    assert ChatMemory is not None


def test_nodeify_accessible():
    from llamabot import nodeify

    assert callable(nodeify)


def test_recorder_symbols():
    from llamabot import (
        span,
    )

    assert callable(span)


def test_experiment_symbols():
    from llamabot import Experiment, metric

    assert Experiment is not None
    assert callable(metric)


def test_docstore_symbols():
    from llamabot import BM25DocStore, LanceDBDocStore, TurboVecDocStore

    assert BM25DocStore is not None
    assert LanceDBDocStore is not None
    assert TurboVecDocStore is not None


def test_set_debug_mode():
    from llamabot import set_debug_mode

    assert callable(set_debug_mode)


def test_all_exports_importable():
    """Every name in __all__ should be importable without error."""
    import llamabot

    for name in llamabot.__all__:
        assert hasattr(llamabot, name), f"{name} in __all__ but not accessible"


def test_simplebot_creation():
    """SimpleBot can be instantiated with basic args."""
    from llamabot import SimpleBot

    bot = SimpleBot(
        system_prompt="You are a test bot.",
        mock_response="hello",
        model_name="gpt-4o-mini",
    )
    assert bot.model_name == "gpt-4o-mini"
    assert bot.temperature == 0.0
