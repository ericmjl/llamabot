"""Definitions for the different types of messages that can be sent."""
from pydantic import BaseModel


class BaseMessage(BaseModel):
    """Base class for all messages."""

    content: str
    role: str

    # Implement slicing for message contents so that I can get content[:-i].
    def __getitem__(self, index):
        """Get the content of the message at the given index."""
        return self.__class__(content=self.content[index], role=self.role)

    def __len__(self):
        """Get the length of the message."""
        return len(self.content)


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
    role: str = "assistant"
