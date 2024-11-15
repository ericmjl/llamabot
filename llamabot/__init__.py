"""Top-level API for llamabot.

This is the file from which you can do:

    from llamabot import some_function

Use it to control the top-level API of your Python data science project.
"""

# Ensure that ~/.llamabotrc exists.
from pathlib import Path

from .bot.chatbot import ChatBot
from .bot.imagebot import ImageBot
from .bot.querybot import QueryBot
from .bot.simplebot import SimpleBot
from .bot.structuredbot import StructuredBot
from .experiments import Experiment, metric
from .prompt_manager import prompt
from .recorder import PromptRecorder

__all__ = [
    "ChatBot",
    "ImageBot",
    "SimpleBot",
    "QueryBot",
    "PromptRecorder",
    "StructuredBot",
    "prompt",
    "Experiment",
    "metric",
]


(Path.home() / ".llamabot").mkdir(parents=True, exist_ok=True)
