"""Bot abstractions that let me quickly build new GPT-based applications."""


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


__all__ = ["SimpleBot", "ChatBot", "QueryBot", "ImageBot"]
