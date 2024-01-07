"""Tests for the ChatBot class."""
import pytest

from llamabot.bot.chatbot import ChatBot
from llamabot.components.messages import AIMessage


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


def test_chatbot_call(mocker, system_prompt, session_name):
    """Test the call method of the ChatBot.
    This test verifies that the ChatBot correctly processes a human message using pytest-mock.
    :param mocker: The pytest-mock mocker fixture.
    :param system_prompt: The system prompt to use for the chatbot.
    :param session_name: The session name for the chatbot.
    """
    # Set up the mocks
    mock_retrieve = mocker.patch(
        "llamabot.bot.chatbot.ChatBot.retrieve", return_value=[]
    )
    mock_generate_response = mocker.patch(
        "llamabot.bot.chatbot.ChatBot.generate_response",
        return_value=AIMessage(content="Mocked AI response"),
    )

    # Initialize the ChatBot and send a message
    chatbot = ChatBot(system_prompt, session_name)
    human_message = "Hello, ChatBot!"
    response = chatbot(human_message)

    # Assertions
    mock_retrieve.assert_called_once()
    mock_generate_response.assert_called_once()
    assert isinstance(response, AIMessage)
    assert response.content == "Mocked AI response"
    assert len(chatbot.messages) > 0  # Chat history should have entries now.


def test_chatbot_repr(system_prompt, session_name):
    """Test that the repr of the chatbot is correct.
    This test ensures that the string representation of ChatBot includes both system and human messages.
    :param system_prompt: The system prompt to use for the chatbot.
    :param session_name: The session name for the chatbot.
    """
    chatbot = ChatBot(system_prompt=system_prompt, session_name=session_name)
    chatbot("Hello, ChatBot!")  # Send a message to create some chat history.
    representation = str(chatbot)
    assert "[Human]" in representation
    assert "[AI]" in representation
