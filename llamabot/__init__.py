"""Top-level API for llamabot.

This is the file from which you can do:

    from llamabot import some_function

Use it to control the top-level API of your Python data science project.
"""
# Ensure that ~/.llamabotrc exists.
from pathlib import Path

from .bot import ChatBot, QueryBot, SimpleBot
from .recorder import PromptRecorder

__all__ = ["ChatBot", "SimpleBot", "QueryBot", "PromptRecorder"]


(Path.home() / ".llamabot").mkdir(parents=True, exist_ok=True)
