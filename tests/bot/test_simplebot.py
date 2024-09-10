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
    stream_target=st.one_of(st.just("stdout"), st.just("panel"), st.just("api")),
    json_mode=st.booleans(),
)
@settings(deadline=None)
def test_simple_bot_init(
    system_prompt, temperature, model_name, stream_target, json_mode
):
    """Test that the SimpleBot is initialized correctly.

    :param system_prompt: The system prompt to use.
    :param temperature: The model temperature to use.
    :param model_name: The name of the OpenAI model to use.
    :param stream_target: The target to stream the response to.
    :param json_mode: Whether to enable JSON mode.
    """
    bot = SimpleBot(system_prompt, temperature, model_name, stream_target, json_mode)
    assert bot.system_prompt.content == system_prompt
    assert bot.temperature == temperature
    assert bot.model_name == model_name
    assert bot.stream_target == stream_target
    assert bot.json_mode == json_mode


@given(st.data())
@settings(deadline=None)
def test_simple_bot_stream_stdout(data):
    """Test that SimpleBot stream API exists and returns agenerator."""
    system_prompt, human_message, mock_response = data.draw(
        st.tuples(st.text(min_size=1), st.text(min_size=1), st.text(min_size=1))
    )
    bot = SimpleBot(system_prompt, stream_target="stdout", mock_response=mock_response)
    result = bot(human_message)
    assert isinstance(result, AIMessage)
    assert result.content in mock_response


@given(
    system_prompt=st.text(min_size=1),
    human_message=st.text(min_size=1),
    mock_response=st.text(min_size=1),
    stream_target=st.one_of(st.just("panel"), st.just("api")),
)
@settings(deadline=None)
def test_simple_bot_stream_panel_or_api(
    system_prompt, human_message, mock_response, stream_target
):
    """Test that SimpleBot stream API exists and returns agenerator."""
    bot = SimpleBot(
        system_prompt, stream_target=stream_target, mock_response=mock_response
    )
    result = bot(human_message)
    for r in result:
        assert isinstance(r, str)
        assert r in mock_response
