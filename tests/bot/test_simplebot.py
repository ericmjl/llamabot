"""
This module contains tests for the SimpleBot class.

It includes tests for the following:
- SimpleBot initialization
- SimpleBot call method

Classes:
- SimpleBot

Functions:
- test_simple_bot_init
- test_simple_bot_call
"""
from unittest.mock import MagicMock

from hypothesis import given, settings, strategies as st

from llamabot.bot.simplebot import SimpleBot
import pytest
from llamabot.components.messages import BaseMessage, AIMessage


@given(
    system_prompt=st.text(),
    temperature=st.floats(min_value=0, max_value=1),
    model_name=st.text(),
    stream=st.booleans(),
    json_mode=st.booleans(),
)
@settings(deadline=None)
def test_simple_bot_init(system_prompt, temperature, model_name, stream, json_mode):
    """Test that the SimpleBot is initialized correctly.

    :param system_prompt: The system prompt to use.
    :param temperature: The model temperature to use.
    :param model_name: The name of the OpenAI model to use.
    :param stream: Whether to stream output.
    :param json_mode: Whether to enable JSON mode.
    """
    bot = SimpleBot(system_prompt, temperature, model_name, stream, json_mode)
    assert bot.system_prompt.content == system_prompt
    assert bot.temperature == temperature
    assert bot.model_name == model_name
    assert bot.stream == stream
    assert bot.json_mode == json_mode


@given(system_prompt=st.text(min_size=1), human_message=st.text(min_size=1))
@settings(deadline=None)
def test_simple_bot_call(system_prompt, human_message):
    """Test that the SimpleBot is called correctly.

    :param system_prompt: The system prompt to use.
    :param human_message: The human message to use.
    """
    bot = SimpleBot(system_prompt, stream=False)
    bot.generate_response = MagicMock()
    response = AIMessage(content="Test response")
    bot.generate_response.return_value = response

    result = bot(human_message)

    bot.generate_response.assert_called_once()
    assert isinstance(result, AIMessage)
    assert result.content == "Test response"


def test_simple_bot_stream_response():
    """Test that the SimpleBot streams responses correctly."""
    bot = SimpleBot(
        system_prompt="", temperature=0.5, model_name="", stream=True, json_mode=False
    )
    messages = [
        BaseMessage(content="Hello"),
        BaseMessage(content="How are you?"),
        BaseMessage(content="Goodbye"),
    ]
    response_generator = bot.stream_response(messages)

    response = next(response_generator)
    assert response == "Hello"

    response = next(response_generator)
    assert response == "How are you?"

    response = next(response_generator)
    assert response == "Goodbye"

    with pytest.raises(StopIteration):
        next(response_generator)
