"""Definitions for the different types of messages that can be sent."""

import base64
import mimetypes
from pathlib import Path
from typing import Iterable, Union

import httpx
from pydantic import BaseModel, Field


class BaseMessage(BaseModel):
    """A base message class."""

    role: str
    content: str
    prompt_hash: str | None = Field(default=None)
    tool_calls: list = Field(default=list())

    # Implement slicing for message contents so that I can get content[:-i].
    def __getitem__(self, index):
        """Get the content of the message at the given index."""
        return self.__class__(content=self.content[index], role=self.role)

    def __len__(self):
        """Get the length of the message."""
        return len(self.content)

    def __radd__(self, other: str) -> "BaseMessage":
        """Right add operation for BaseMessage.

        :param other: The string to add to the content.
        :returns: A new BaseMessage with the updated content.
        """
        if isinstance(other, str):
            return self.__class__(content=other + self.content, role=self.role)

    def __add__(self, other: str) -> "BaseMessage":
        """Left add operation for BaseMessage.

        :param other: The string to add to the content.
        :returns: A new BaseMessage with the updated content.
        """
        if isinstance(other, str):
            return self.__class__(content=self.content + other, role=self.role)


class SystemMessage(BaseMessage):
    """A message from the system.

    Remains in the library for legacy compatibility.
    """

    content: str
    role: str = "system"


class DeveloperMessage(BaseMessage):
    """A message from the developer."""

    content: str
    role: str = "developer"


class HumanMessage(BaseMessage):
    """A message from a human."""

    content: str
    role: str = "user"


class AIMessage(BaseMessage):
    """A message from the AI."""

    content: str
    role: str = "assistant"


class ThoughtMessage(AIMessage):
    """A message representing the agent's reasoning/thought process."""

    content: str
    role: str = "assistant"


class ObservationMessage(AIMessage):
    """A message representing the agent's observation of tool results."""

    content: str
    role: str = "assistant"


class ToolMessage(BaseMessage):
    """A message from the AI."""

    content: str
    role: str = "tool"


class RetrievedMessage(BaseMessage):
    """A message retrieved from the history."""

    content: str
    role: str = "system"


class ImageMessage(BaseMessage):
    """A message containing an image.

    :param content: Path to image file or URL of image
    :param role: Role of the message sender, defaults to "user"
    """

    content: str
    role: str = "user"

    def __init__(self, content: Union[str, Path], role: str = "user"):
        if isinstance(content, Path):
            path = content
        elif content.startswith(("http://", "https://")):
            # Download image from URL to temporary bytes

            response = httpx.get(content)
            image_bytes = response.content
            mime_type = response.headers["content-type"]
            encoded = base64.b64encode(image_bytes).decode("utf-8")
            super().__init__(content=encoded, role=role)
            self._mime_type = mime_type
            return
        else:
            path = Path(content)

        # Handle local file
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        mime_type = mimetypes.guess_type(path)[0]
        if not mime_type or not mime_type.startswith("image/"):
            raise ValueError(f"Not a valid image file: {path}")

        with open(path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("utf-8")

        super().__init__(content=encoded, role=role)
        self._mime_type = mime_type

    def model_dump(self):
        """Convert message to format expected by LiteLLM and OpenAI."""
        return {
            "role": self.role,
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{self._mime_type};base64,{self.content}"
                    },
                }
            ],
        }


def retrieve_messages_up_to_budget(
    messages: list[BaseMessage], character_budget: int
) -> list[BaseMessage]:
    """Retrieve messages up to the character budget.

    :param messages: The messages to retrieve.
    :param character_budget: The character budget to use.
    :returns: The retrieved messages.
    """
    used_chars = 0
    retrieved_messages = []
    for message in messages:
        if not isinstance(message, (BaseMessage, str)):
            raise ValueError(
                f"Expected message to be of type BaseMessage or str, got {type(message)}"
            )
        used_chars += len(message)
        if used_chars > character_budget:
            # append whatever is left
            retrieved_messages.append(message[: used_chars - character_budget])
            break
        retrieved_messages.append(message)
    return retrieved_messages


def to_basemessage(
    messages: Iterable[Union[str, BaseMessage, list[Union[str, BaseMessage]]]],
) -> list[BaseMessage]:
    """Process a tuple of messages into a list of BaseMessage objects.

    Handles nested lists and converts strings to HumanMessages.

    :param messages: Tuple of messages to process
    :return: List of BaseMessage objects
    """
    processed_messages = []

    def process_message(msg: Union[str, BaseMessage, list]) -> None:
        """Process a single message or list of messages into BaseMessage objects.

        Recursively processes nested lists and converts strings to HumanMessages.
        Appends processed messages to the outer scope processed_messages list.

        :param msg: Message to process - can be a string, BaseMessage, or list of messages
        """
        if isinstance(msg, list):
            for m in msg:
                process_message(m)
        elif isinstance(msg, str):
            processed_messages.append(HumanMessage(content=msg))
        elif isinstance(msg, BaseMessage):
            # Ensure content is stringified before creating BaseMessage
            content = msg.content
            if not isinstance(content, str):
                content = str(content)
            # Create a new message with stringified content
            processed_messages.append(
                msg.__class__(
                    content=content,
                    role=msg.role,
                    prompt_hash=msg.prompt_hash,
                    tool_calls=msg.tool_calls,
                )
            )
        else:
            # For any other type, convert to string and create HumanMessage
            processed_messages.append(HumanMessage(content=str(msg)))

    for msg in messages:
        process_message(msg)

    return processed_messages


# High-level API


def _handle_single_content(
    item: Union[str, Path, BaseMessage],
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


def user(
    *content: Union[str, Path, BaseMessage],
) -> Union[
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


def dev(*content: Union[str, Path, BaseMessage]) -> DeveloperMessage:
    """Create a DeveloperMessage from one or more pieces of content.

    Similar to the system() function, this combines multiple inputs into a single
    DeveloperMessage. Content can be provided as strings, file paths, or existing messages.

    Examples:
        >>> from pathlib import Path
        >>> import llamabot as lmb
        >>> lmb.dev("Write tests for this function")
        DeveloperMessage(content="Write tests for this function")

        >>> dev("Write tests", "with good coverage")
        DeveloperMessage(content="Write tests with good coverage")

        >>> dev(Path("dev_notes.txt"))  # File containing "Add error handling"
        DeveloperMessage(content="Add error handling")

        >>> dev(DeveloperMessage(content="Refactor code"), "to be more modular")
        DeveloperMessage(content="Refactor code to be more modular")

        >>> dev("Add docstrings", Path("style.txt"), DeveloperMessage(content="Follow PEP8"))
        DeveloperMessage(content="Add docstrings Use formal language Follow PEP8")

    :param content: One or more pieces of content to convert into a developer message.
        Can be strings, Paths to text files, or BaseMessage objects.
    :return: A single DeveloperMessage containing all content
    :raises FileNotFoundError: If a specified file path doesn't exist
    """

    def _process_item(item: Union[str, Path, BaseMessage]) -> str:
        """Process a single content item into a string.

        :param item: The item to process. Can be a string, Path to a text file,
                    or BaseMessage object
        :return: The string content of the item
        :raises FileNotFoundError: If a Path item points to a non-existent file
        """
        if isinstance(item, DeveloperMessage):
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
    return DeveloperMessage(content=combined_content)


def assistant(*content: Union[str, Path, BaseMessage]) -> AIMessage:
    """Create an AIMessage from one or more pieces of content.

    Similar to the system() and dev() functions, this combines multiple inputs into a single
    AIMessage. Content can be provided as strings, file paths, or existing messages.

    Examples:
        >>> from pathlib import Path
        >>> import llamabot as lmb
        >>> lmb.assistant("Hello! How can I help you today?")
        AIMessage(content="Hello! How can I help you today?")

        >>> assistant("I can help you", "with Python programming")
        AIMessage(content="I can help you with Python programming")

        >>> assistant(Path("response.txt"))  # File containing "Here's the answer"
        AIMessage(content="Here's the answer")

        >>> assistant(AIMessage(content="Let me explain"), "step by step")
        AIMessage(content="Let me explain step by step")

        >>> assistant("Here's a solution", Path("code.txt"), AIMessage(content="Hope this helps"))
        AIMessage(content="Here's a solution def hello(): print('world') Hope this helps")

    :param content: One or more pieces of content to convert into an AI message.
        Can be strings, Paths to text files, or BaseMessage objects.
    :return: A single AIMessage containing all content
    :raises FileNotFoundError: If a specified file path doesn't exist
    """

    def _process_item(item: Union[str, Path, BaseMessage]) -> str:
        """Process a single content item into a string.

        :param item: The item to process. Can be a string, Path to a text file,
                    or BaseMessage object
        :return: The string content of the item
        :raises FileNotFoundError: If a Path item points to a non-existent file
        """
        if isinstance(item, AIMessage):
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
    return AIMessage(content=combined_content)


def thought(*content: Union[str, Path, BaseMessage]) -> ThoughtMessage:
    """Create a thought message from content.

    :param content: The content of the thought message. Can be strings, Path objects,
                   or BaseMessage objects
    :return: A ThoughtMessage object
    """
    if not content:
        return ThoughtMessage(content="")

    def _process_item(item: Union[str, Path, BaseMessage]) -> str:
        """Process a single content item into a string.

        :param item: The item to process. Can be a string, Path to a text file,
                    or BaseMessage object
        :return: The string content of the item
        :raises FileNotFoundError: If a Path item points to a non-existent file
        """
        if isinstance(item, (AIMessage, ThoughtMessage)):
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
    return ThoughtMessage(content=combined_content)


def observation(*content: Union[str, Path, BaseMessage]) -> ObservationMessage:
    """Create an observation message from content.

    :param content: The content of the observation message. Can be strings, Path objects,
                   or BaseMessage objects
    :return: An ObservationMessage object
    """
    if not content:
        return ObservationMessage(content="")

    def _process_item(item: Union[str, Path, BaseMessage]) -> str:
        """Process a single content item into a string.

        :param item: The item to process. Can be a string, Path to a text file,
                    or BaseMessage object
        :return: The string content of the item
        :raises FileNotFoundError: If a Path item points to a non-existent file
        """
        if isinstance(item, (AIMessage, ObservationMessage)):
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
    return ObservationMessage(content=combined_content)
