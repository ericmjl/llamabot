"""Tests for the message classes."""

from llamabot.components.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    process_messages,
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


def test_process_messages_with_nested_types():
    """Test that process_messages handles various input types correctly."""
    from llamabot.components.messages import (
        HumanMessage,
        SystemMessage,
    )

    # Test with a mix of types
    test_message = SystemMessage(content="system message")
    messages = (
        "human message",
        test_message,
        ["nested message 1", "nested message 2"],
    )

    result = process_messages(messages)

    assert len(result) == 4
    assert isinstance(result[0], HumanMessage)
    assert result[0].content == "human message"
    assert result[1] == test_message
    assert isinstance(result[2], HumanMessage)
    assert result[2].content == "nested message 1"
    assert isinstance(result[3], HumanMessage)
    assert result[3].content == "nested message 2"
