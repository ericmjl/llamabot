"""Bot abstractions that let me quickly build new GPT-based applications."""

import os
import warnings
from pathlib import Path

import openai
import panel as pn
from dotenv import load_dotenv

from .chatbot import ChatBot
from .querybot import QueryBot
from .simplebot import SimpleBot

pn.extension()
load_dotenv()

config_path = Path.home() / ".llamabotrc"
if config_path.exists():
    load_dotenv(config_path)

api_key = os.getenv("OPENAI_API_KEY", None)
if api_key is None:
    warnings.warn(
        "No OpenAI API key found. Please set OPENAI_API_KEY in your environment.",
        category=RuntimeWarning,
    )
openai.api_key = api_key


__all__ = ["SimpleBot", "ChatBot", "QueryBot"]
