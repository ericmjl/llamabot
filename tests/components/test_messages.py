"""Tests for the message classes."""

from llamabot.components.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    to_basemessage,
    dev,
    DeveloperMessage,
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

    result = to_basemessage(messages)

    assert len(result) == 4
    assert isinstance(result[0], HumanMessage)
    assert result[0].content == "human message"
    assert result[1] == test_message
    assert isinstance(result[2], HumanMessage)
    assert result[2].content == "nested message 1"
    assert isinstance(result[3], HumanMessage)
    assert result[3].content == "nested message 2"


def test_dev_message_creation(tmp_path):
    """Test that the dev() function creates DeveloperMessages correctly.

    Tests various input types including strings, paths, and existing messages.
    """
    from pytest import raises

    # Create a temporary file for testing
    test_file = tmp_path / "dev_notes.txt"
    test_file.write_text("Add error handling")

    # Test single string input
    msg1 = dev("Write tests")
    assert isinstance(msg1, DeveloperMessage)
    assert msg1.content == "Write tests"

    # Test multiple string inputs
    msg2 = dev("Write tests", "with good coverage")
    assert isinstance(msg2, DeveloperMessage)
    assert msg2.content == "Write tests with good coverage"

    # Test file path input
    msg3 = dev(test_file)
    assert isinstance(msg3, DeveloperMessage)
    assert msg3.content == "Add error handling"

    # Test existing DeveloperMessage input
    existing_msg = DeveloperMessage(content="Refactor code")
    msg4 = dev(existing_msg, "to be more modular")
    assert isinstance(msg4, DeveloperMessage)
    assert msg4.content == "Refactor code to be more modular"

    # Test mixed input types
    msg5 = dev("Add docstrings", test_file, DeveloperMessage(content="Follow PEP8"))
    assert isinstance(msg5, DeveloperMessage)
    assert msg5.content == "Add docstrings Add error handling Follow PEP8"

    # Test with non-existent file
    with raises(FileNotFoundError):
        dev(tmp_path / "nonexistent.txt")

    # Test with other message types
    human_msg = HumanMessage(content="Test message")
    msg6 = dev(human_msg)
    assert isinstance(msg6, DeveloperMessage)
    assert msg6.content == "Test message"
