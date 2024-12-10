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
from .components.messages import HumanMessage, ImageMessage, SystemMessage, BaseMessage
from .components.tools import tool


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
    "tool",
]

# Ensure ~/.llamabot directory exists
(Path.home() / ".llamabot").mkdir(parents=True, exist_ok=True)


# High-level API
def _handle_single_content(
    item: Union[str, Path, BaseMessage]
) -> Union[HumanMessage, ImageMessage, BaseMessage]:
    """Handle a single content item and convert it to an appropriate message type.

    This helper function processes a single piece of content and determines whether it should
    be treated as a Path, URL, plain text content, or passed through as a BaseMessage.

    :param item: The content item to process, either a string, Path object, or BaseMessage
    :return: Either a HumanMessage, ImageMessage, or the original BaseMessage
    :raises FileNotFoundError: If a specified file path doesn't exist
    :raises ValueError: If an image file is invalid
    :raises httpx.HTTPError: If an image URL can't be accessed
    """
    # Pass through BaseMessage objects unchanged
    if isinstance(item, BaseMessage):
        return item

    # Handle Path objects directly
    if isinstance(item, Path):
        return _handle_path(item)

    # Handle string content
    if isinstance(item, str):
        # Check if string is a URL
        if item.startswith(("http://", "https://")):
            return _handle_url(item)

        # Check if string is a path that exists
        try:
            path = Path(item)
            if path.exists():
                return _handle_path(path)
        except OSError:
            # Skip if path is invalid (e.g. too long)
            pass

    return HumanMessage(content=item)


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


def user(*content: Union[str, Path, BaseMessage]) -> Union[
    HumanMessage,
    ImageMessage,
    BaseMessage,
    list[Union[HumanMessage, ImageMessage, BaseMessage]],
]:
    """Create one or more user messages from the given content.

    This function provides a flexible way to create user messages from various types of content:
    - Plain text strings become HumanMessages
    - Image file paths become ImageMessages
    - URLs to images become ImageMessages
    - Text file paths become HumanMessages with the file contents
    - BaseMessage objects are passed through unchanged
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

        >>> user(HumanMessage(content="existing message"))  # Pass through BaseMessage
        HumanMessage(content="existing message")

        >>> user("msg1", "msg2")  # Multiple messages
        [HumanMessage(content="msg1"), HumanMessage(content="msg2")]

    :param content: One or more pieces of content to convert into messages.
        Can be strings (text/URLs), Paths to files, or BaseMessage objects.
    :return: Either a single message or list of messages depending on input type
    :raises FileNotFoundError: If a specified file path doesn't exist
    :raises ValueError: If an image file is invalid
    :raises httpx.HTTPError: If an image URL can't be accessed
    """
    # Handle single input
    if len(content) == 1:
        return _handle_single_content(content[0])

    # Handle multiple inputs
    return [_handle_single_content(item) for item in content]


def system(*content: Union[str, Path, BaseMessage]) -> SystemMessage:
    """Create a system message for instructing the LLM.

    System messages are used to set the behavior, role, or context for the LLM.
    They act as high-level instructions that guide the model's responses.
    Multiple inputs will be concatenated into a single system message.

    Examples:
        >>> system("You are a helpful assistant.")
        SystemMessage(content="You are a helpful assistant.")

        >>> system("Respond in the style of Shakespeare.")
        SystemMessage(content="Respond in the style of Shakespeare.")

        >>> system("Be helpful", "Be concise")
        SystemMessage(content="Be helpful Be concise")

        >>> system(Path("prompt.txt"))  # File containing "You are an expert coder"
        SystemMessage(content="You are an expert coder")

        >>> system(SystemMessage(content="Be factual"), "and precise")
        SystemMessage(content="Be factual and precise")

        >>> system("Be helpful", Path("style.txt"), SystemMessage(content="Be concise"))
        SystemMessage(content="Be helpful Use formal language Be concise")

        >>> # Combining different types of inputs
        >>> system(
        ...     "You are an assistant",
        ...     Path("role.txt"),  # Contains "specializing in Python"
        ...     SystemMessage(content="Focus on best practices")
        ... )
        SystemMessage(content="You are an assistant specializing in Python Focus on best practices")

    :param content: One or more pieces of content to convert into a system message.
        Can be strings, Paths to text files, or BaseMessage objects.
    :return: A single SystemMessage containing all content
    :raises FileNotFoundError: If a specified file path doesn't exist
    """

    def _process_item(item: Union[str, Path, BaseMessage]) -> str:
        """Process a single content item into a string.

        :param item: The item to process. Can be a string, Path to a text file,
                    or BaseMessage object
        :return: The string content of the item
        :raises FileNotFoundError: If a Path item points to a non-existent file
        """
        if isinstance(item, SystemMessage):
            return item.content
        if isinstance(item, BaseMessage):
            return item.content
        if isinstance(item, Path):
            if not item.exists():
                raise FileNotFoundError(f"File not found: {item}")
            return item.read_text()
        return str(item)

    # Process all items and join with spaces
    combined_content = " ".join(_process_item(item) for item in content)
    return SystemMessage(content=combined_content)
