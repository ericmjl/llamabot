"""Tests for the ChatBot class."""
import pytest

from llamabot import ChatBot


@pytest.fixture
def system_prompt():
    """Pytest fixture for the system prompt.

    :return: The system prompt to use for the chatbot.
    """
    return "Test prompt"


def test_chatbot_initialization(system_prompt):
    """Test initialization of the chatbot.

    This is just an execution test to make sure there are no errors.

    :param system_prompt: The system prompt to use for the chatbot.
        This is a pytest fixture.
    """
    ChatBot(system_prompt)
    assert True


def test_chatbot_repr(system_prompt):
    """Test that the repr of the chatbot is correct.

    Our correctness test is that the system prompt is included in the repr.

    :param system_prompt: The system prompt to use for the chatbot.
        This is a pytest fixture.
    """
    cb = ChatBot(system_prompt=system_prompt)
    assert system_prompt in str(cb)


def test_chatbot_panel(system_prompt):
    """Test that a panel app is returned when show=False is passed.

    :param system_prompt: The system prompt to use for the chatbot.
        This is a pytest fixture.
    """
    cb = ChatBot(system_prompt=system_prompt)
    panel_app = cb.panel(show=False)
    assert panel_app is not None  # we expect a panel app to be returned all the time!
