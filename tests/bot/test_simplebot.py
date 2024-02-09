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

from hypothesis import given, settings, strategies as st

from llamabot.bot.simplebot import SimpleBot
from llamabot.components.messages import AIMessage


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
    bot = SimpleBot(system_prompt, stream=False, mock_response="hello")
    result = bot(human_message)
    assert isinstance(result, AIMessage)
    assert result.content == "hello"
