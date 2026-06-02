"""Top-level API for llamabot.

This is the file from which you can do:

    from llamabot import some_function

All bot classes, components, and utilities are lazily loaded via
``lazy_loader`` so that ``import llamabot`` is near-instant.
Heavy dependencies are only pulled in when you actually access a symbol.

The module provides several high-level functions and classes for working with LLMs:

- Message creation functions: `user()` and `system()`
- Bot classes: SimpleBot, AsyncSimpleBot, StructuredBot, AsyncStructuredBot, ChatBot, ImageBot, QueryBot, AsyncQueryBot, ToolBot, AsyncToolBot, AgentBot, AsyncAgentBot
- Prompt management: `prompt` decorator
- Experimentation: `Experiment` and `metric`
- Recording: `PromptRecorder`
"""

import os
from pathlib import Path

from loguru import logger

import lazy_loader


def set_debug_mode(enabled: bool = True) -> None:
    """Set debug mode for llamabot.

    When enabled, this will show debug-level logging messages from llamabot.
    This is useful for debugging model interactions and tool calls.

    :param enabled: Whether to enable debug mode
    """
    level = "DEBUG" if enabled else "WARNING"
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level=level)


log_level = os.getenv("LOG_LEVEL", "WARNING").upper()
level_map = {
    "DEBUG": "DEBUG",
    "INFO": "INFO",
    "WARNING": "WARNING",
    "ERROR": "ERROR",
    "CRITICAL": "CRITICAL",
}

logger.add(lambda msg: print(msg, end=""), level=level_map.get(log_level, "WARNING"))

__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules=[],
    submod_attrs={
        "bot.agentbot": ["AgentBot"],
        "bot.async_agentbot": ["AsyncAgentBot"],
        "bot.imagebot": ["ImageBot"],
        "bot.querybot": ["AsyncQueryBot", "QueryBot"],
        "bot.simplebot": ["AsyncSimpleBot", "SimpleBot"],
        "bot.structuredbot": ["AsyncStructuredBot", "StructuredBot"],
        "bot.toolbot": ["AsyncToolBot", "ToolBot"],
        "components.chat_memory": ["ChatMemory"],
        "components.docstore": ["BM25DocStore", "LanceDBDocStore", "TurboVecDocStore"],
        "components.messages": ["dev", "system", "user"],
        "components.pocketflow": ["nodeify"],
        "components.tools": ["tool"],
        "experiments": ["Experiment", "metric"],
        "prompt_manager": ["prompt"],
        "recorder": [
            "SpanList",
            "dict_to_span",
            "enable_span_recording",
            "get_current_span",
            "get_span_tree",
            "get_spans",
            "span",
        ],
    },
)

__all__.append("set_debug_mode")

(Path.home() / ".llamabot").mkdir(parents=True, exist_ok=True)
