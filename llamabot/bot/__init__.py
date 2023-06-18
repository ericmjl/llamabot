"""Bot abstractions that let me quickly build new GPT-based applications."""

import os
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
    raise ValueError("OPENAI_API_KEY environment variable must be set.")
openai.api_key = api_key


__all__ = ["SimpleBot", "ChatBot", "QueryBot"]
