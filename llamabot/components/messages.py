"""Definitions for the different types of messages that can be sent."""

import base64
import mimetypes
from pathlib import Path
from typing import Union

import httpx
from pydantic import BaseModel, Field


class BaseMessage(BaseModel):
    """A base message class."""

    role: str
    content: str
    prompt_hash: str | None = Field(default=None)

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
    """A message from the system."""

    content: str
    role: str = "system"


class HumanMessage(BaseMessage):
    """A message from a human."""

    content: str
    role: str = "user"


class AIMessage(BaseMessage):
    """A message from the AI."""

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


def process_messages(
    messages: tuple[Union[str, BaseMessage, list[Union[str, BaseMessage]], ...]]
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
        else:
            processed_messages.append(msg)

    for msg in messages:
        process_message(msg)

    return processed_messages
