"""Tests for the ChatBot class."""

import pytest

from llamabot.bot.chatbot import ChatBot


@pytest.fixture
def system_prompt():
    """Pytest fixture for the system prompt.

    :return: The system prompt to use for the chatbot.
    """
    return "Test prompt"


@pytest.fixture
def session_name():
    """Pytest fixture for the session name.
    :return: A session name for testing purposes.
    """
    return "test_session"


def test_chatbot_initialization(system_prompt, session_name):
    """Test initialization of the chatbot.
    This test verifies the correct instantiation of the ChatBot object.
    :param system_prompt: The system prompt to use for the chatbot.
    :param session_name: The session name for the chatbot.
    """
    chatbot = ChatBot(system_prompt, session_name)
    assert chatbot.system_prompt.content == system_prompt
    assert chatbot.session_name == session_name
    assert isinstance(chatbot.messages, list)


def test_chatbot_repr(system_prompt, session_name):
    """Test that the repr of the chatbot is correct.

    This test ensures that the string representation of ChatBot
    includes both human and AI messages.
    It also serves as an execution test for `__call__`.

    :param system_prompt: The system prompt to use for the chatbot.
    :param session_name: The session name for the chatbot.
    """
    # Initialize ChatBot and simulate a human message
    chatbot = ChatBot(
        system_prompt=system_prompt,
        session_name=session_name,
        stream_target="stdout",
        mock_response="Mocked AI response",
    )
    response = chatbot("Hello, ChatBot!")  # Send a message to create some chat history.

    # Verify the string representation
    representation = str(chatbot)
    assert "[Human]" in representation
    assert "[AI]" in representation
    assert response.content in representation
