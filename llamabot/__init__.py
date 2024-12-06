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
from typing import Union
import mimetypes

import httpx

from loguru import logger

from .bot.chatbot import ChatBot
from .bot.imagebot import ImageBot
from .bot.querybot import QueryBot
from .bot.simplebot import SimpleBot
from .bot.structuredbot import StructuredBot
from .experiments import Experiment, metric
from .prompt_manager import prompt
from .recorder import PromptRecorder
from .components.messages import HumanMessage, ImageMessage, SystemMessage

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

# Ensure ~/.llamabot directory exists
(Path.home() / ".llamabot").mkdir(parents=True, exist_ok=True)


# High-level API
def user(
    *content: Union[str, Path]
) -> Union[HumanMessage, ImageMessage, list[Union[HumanMessage, ImageMessage]]]:
    """Create one or more user messages from the given content.

    This function provides a flexible way to create user messages from various types of content:
    - Plain text strings become HumanMessages
    - Image file paths become ImageMessages
    - URLs to images become ImageMessages
    - Text file paths become HumanMessages with the file contents
    - Multiple inputs return a list of messages

    Examples:
        >>> user("Hello, world!")  # Simple text message
        HumanMessage(content="Hello, world!")

        >>> user("image.png")  # Local image file
        ImageMessage(content="<base64-encoded-content>")

        >>> user("https://example.com/image.jpg")  # Image URL
        ImageMessage(content="<base64-encoded-content>")

        >>> user("text.txt")  # Text file
        HumanMessage(content="<file-contents>")

        >>> user("msg1", "msg2")  # Multiple messages
        [HumanMessage(content="msg1"), HumanMessage(content="msg2")]

    :param content: One or more pieces of content to convert into messages.
        Can be strings (text/URLs) or Paths to files.
    :return: Either a single message or list of messages depending on input type
    :raises FileNotFoundError: If a specified file path doesn't exist
    :raises ValueError: If an image file is invalid
    :raises httpx.HTTPError: If an image URL can't be accessed
    """

    def _handle_path(path: Path) -> Union[HumanMessage, ImageMessage]:
        """Handle Path objects by checking if they are images or text files."""
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type and mime_type.startswith("image/"):
            return ImageMessage(content=path)
        return HumanMessage(content=path.read_text())

    def _handle_url(url: str) -> Union[HumanMessage, ImageMessage]:
        """Handle URL strings by attempting to load as image, falling back to text."""
        try:
            return ImageMessage(content=url)
        except (httpx.HTTPError, ValueError):
            return HumanMessage(content=url)

    def _handle_single_content(
        item: Union[str, Path]
    ) -> Union[HumanMessage, ImageMessage]:
        """Handle a single content item and convert it to an appropriate message type.

        This helper function processes a single piece of content and determines whether it should
        be treated as a Path, URL, or plain text content.

        :param item: The content item to process, either a string or Path object
        :return: Either a HumanMessage or ImageMessage depending on the content type
        :raises FileNotFoundError: If a specified file path doesn't exist
        :raises ValueError: If an image file is invalid
        :raises httpx.HTTPError: If an image URL can't be accessed
        """
        # Handle Path objects directly
        if isinstance(item, Path):
            return _handle_path(item)

        # Handle string content
        if isinstance(item, str):
            # Check if string is a URL
            if item.startswith(("http://", "https://")):
                return _handle_url(item)

            # Check if string is a path that exists
            path = Path(item)
            if path.exists():
                return _handle_path(path)

        return HumanMessage(content=item)

    # Handle single input
    if len(content) == 1:
        return _handle_single_content(content[0])

    # Handle multiple inputs
    return [_handle_single_content(item) for item in content]


def system(content: str) -> SystemMessage:
    """Create a system message for instructing the LLM.

    System messages are used to set the behavior, role, or context for the LLM.
    They act as high-level instructions that guide the model's responses.

    Examples:
        >>> system("You are a helpful assistant.")
        SystemMessage(content="You are a helpful assistant.")

        >>> system("Respond in the style of Shakespeare.")
        SystemMessage(content="Respond in the style of Shakespeare.")

    :param content: The instruction or context to give to the LLM
    :return: A SystemMessage containing the provided content
    """
    return SystemMessage(content=content)
