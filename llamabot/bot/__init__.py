"""Bot abstractions that let me quickly build new GPT-based applications."""

import os
import warnings

import panel as pn
from dotenv import load_dotenv

from llamabot.config import llamabotrc_path

from .chatbot import ChatBot
from .querybot import QueryBot
from .simplebot import SimpleBot
from .imagebot import ImageBot

pn.extension()
load_dotenv()

if llamabotrc_path.exists():
    load_dotenv(llamabotrc_path)

api_key = os.getenv("OPENAI_API_KEY", None)
if api_key is None:
    warnings.warn(
        "No OpenAI API key found. Please set OPENAI_API_KEY in your environment.",
        category=RuntimeWarning,
    )


__all__ = ["SimpleBot", "ChatBot", "QueryBot", "ImageBot"]
