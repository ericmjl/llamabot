"""Data models for chat memory."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from llamabot.components.messages import BaseMessage


class MessageSummary(BaseModel):
    """Summary of a message for better threading."""

    title: str = Field(..., description="Title of the message")
    summary: str = Field(..., description="Summary of the message. Two sentences max.")


@dataclass
class ConversationNode:
    """A node in the conversation graph."""

    id: int  # Auto-incremented based on number of nodes in graph
    message: BaseMessage  # Single message (not conversation turn)
    summary: Optional[MessageSummary] = None
    parent_id: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate node data after initialization."""
        if self.id <= 0:
            raise ValueError("Node ID must be positive")

        if self.parent_id is not None and self.parent_id <= 0:
            raise ValueError("Parent ID must be positive if not None")
