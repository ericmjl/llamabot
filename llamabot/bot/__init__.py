"""Bot abstractions that let me quickly build new GPT-based applications."""

import os

import openai
import panel as pn
from dotenv import load_dotenv

from .chatbot import ChatBot
from .querybot import QueryBot
from .simplebot import SimpleBot

pn.extension()
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


__all__ = ["SimpleBot", "ChatBot", "QueryBot"]
