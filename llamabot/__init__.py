"""Top-level API for llamabot.

This is the file from which you can do:

    from llamabot import some_function

Use it to control the top-level API of your Python data science project.

The module provides several high-level functions and classes for working with LLMs:

- Message creation functions: `user()` and `system()`
- Bot classes: SimpleBot, StructuredBot, ChatBot, ImageBot, QueryBot
- Prompt management: `prompt` decorator
- Experimentation: `Experiment` and `metric`
- Recording: `PromptRecorder`
"""

import os
from pathlib import Path

from loguru import logger

from .bot.agentbot import AgentBot
from .bot.imagebot import ImageBot
from .bot.querybot import QueryBot
from .bot.simplebot import SimpleBot
from .bot.structuredbot import StructuredBot
from .bot.toolbot import ToolBot
from .experiments import Experiment, metric
from .prompt_manager import prompt
from .components.messages import user, system, dev
from .components.tools import tool
from .components.docstore import BM25DocStore, LanceDBDocStore
from .components.chat_memory import ChatMemory
from .components.pocketflow import nodeify


def set_debug_mode(enabled: bool = True) -> None:
    """Set debug mode for llamabot.

    When enabled, this will show debug-level logging messages from llamabot.
    This is useful for debugging model interactions and tool calls.

    :param enabled: Whether to enable debug mode
    """
    level = "DEBUG" if enabled else "WARNING"
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level=level)


# Configure logger
log_level = os.getenv("LOG_LEVEL", "WARNING").upper()
level_map = {
    "DEBUG": "DEBUG",
    "INFO": "INFO",
    "WARNING": "WARNING",
    "ERROR": "ERROR",
    "CRITICAL": "CRITICAL",
}

# Remove default logger configuration and set the desired level
logger.add(lambda msg: print(msg, end=""), level=level_map.get(log_level, "WARNING"))

__all__ = [
    "AgentBot",
    "ImageBot",
    "SimpleBot",
    "QueryBot",
    "StructuredBot",
    "ToolBot",
    "prompt",
    "Experiment",
    "metric",
    "tool",
    "user",
    "system",
    "dev",
    "BM25DocStore",
    "LanceDBDocStore",
    "set_debug_mode",
    "ChatMemory",
    "nodeify",
]

# Ensure ~/.llamabot directory exists
(Path.home() / ".llamabot").mkdir(parents=True, exist_ok=True)
