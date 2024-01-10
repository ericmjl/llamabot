"""Tests for the message classes."""
from llamabot.components.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)


def test_system_message_slicing():
    """Test slicing for system messages."""
    msg = SystemMessage(content="System Alert")
    assert msg[:6].content == "System"
    assert msg[7:].content == "Alert"
    assert msg[-5:].content == "Alert"
    assert msg[:-7].content == "Syste"


def test_human_message_slicing():
    """Test slicing for human messages."""
    msg = HumanMessage(content="User Message")
    assert msg[:4].content == "User"
    assert msg[5:].content == "Message"
    assert msg[-7:].content == "Message"
    assert msg[:-8].content == "User"


def test_ai_message_slicing():
    """Test slicing for AI messages."""
    msg = AIMessage(content="AI Response")
    assert msg[:2].content == "AI"
    assert msg[3:].content == "Response"
    assert msg[-8:].content == "Response"
    assert msg[:-9].content == "AI"


def test_tool_message_slicing():
    """Test slicing for tool messages."""
    msg = ToolMessage(content="Tool Output")
    assert msg[:4].content == "Tool"
    assert msg[5:].content == "Output"
    assert msg[-6:].content == "Output"
    assert msg[:-7].content == "Tool"
