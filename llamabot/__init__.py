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
from .bot.chatbot import ChatBot
from .bot.imagebot import ImageBot
from .bot.querybot import QueryBot
from .bot.simplebot import SimpleBot
from .bot.structuredbot import StructuredBot
from .experiments import Experiment, metric
from .prompt_manager import prompt
from .recorder import PromptRecorder
from .components.messages import user, system, dev
from .components.tools import tool
from .components.docstore import BM25DocStore, LanceDBDocStore, ChromaDBDocStore


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
logger.remove()
logger.add(lambda msg: print(msg, end=""), level=level_map.get(log_level, "WARNING"))

__all__ = [
    "AgentBot",
    "ChatBot",
    "ImageBot",
    "SimpleBot",
    "QueryBot",
    "PromptRecorder",
    "StructuredBot",
    "prompt",
    "Experiment",
    "metric",
    "tool",
    "user",
    "system",
    "dev",
    "BM25DocStore",
    "LanceDBDocStore",
    "ChromaDBDocStore",
]

# Ensure ~/.llamabot directory exists
(Path.home() / ".llamabot").mkdir(parents=True, exist_ok=True)
