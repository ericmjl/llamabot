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

from hypothesis import given
from hypothesis import strategies as st

from llamabot.bot.simplebot import AIMessage, HumanMessage, SimpleBot, SystemMessage


@given(
    system_prompt=st.text(),
    temperature=st.floats(min_value=0, max_value=1),
    model_name=st.text(),
)
def test_simple_bot_init(system_prompt, temperature, model_name):
    """Test that the SimpleBot is initialized correctly.

    :param system_prompt: The system prompt to use.
    :param temperature: The model temperature to use.
    :param model_name: The name of the OpenAI model to use.
    """
    bot = SimpleBot(system_prompt, temperature, model_name)
    assert bot.system_prompt == system_prompt
    assert bot.model.temperature == temperature
    assert bot.model.model_name == model_name
    assert bot.chat_history == []


@given(system_prompt=st.text(), human_message=st.text())
def test_simple_bot_call(system_prompt, human_message):
    """Test that the SimpleBot is called correctly.

    :param system_prompt: The system prompt to use.
    :param human_message: The human message to use.
    """
    bot = SimpleBot(system_prompt)
    bot.model = MagicMock()
    response = AIMessage(content="Test response")
    bot.model.return_value = response

    result = bot(human_message)

    bot.model.assert_called_once_with(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_message)]
    )
    assert result == response
    assert bot.chat_history == [HumanMessage(content=human_message), response]
