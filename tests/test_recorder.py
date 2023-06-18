"""
This module provides a set of tests for the PromptRecorder and autorecord functions in the llamabot.recorder module.

Functions:
- test_init: Test that the PromptRecorder is initialized correctly.
- test_log: Test that the PromptRecorder logs prompts and responses correctly.
- test_repr: Test that the PromptRecorder is represented correctly.
- test_dataframe: Test that the PromptRecorder is represented correctly as a dataframe.
- test_autorecord_no_context: Test that autorecord works without a context.
- test_autorecord_with_context: Test that autorecord works with a context.
- test_autorecord_multiple_with_context: Test that autorecord works with a context multiple times.
"""
import pandas as pd

from llamabot.recorder import PromptRecorder, autorecord


def test_init():
    """Test that the PromptRecorder is initialized correctly."""
    recorder = PromptRecorder()
    assert recorder.prompts_and_responses == []


def test_log():
    """Test that the PromptRecorder logs prompts and responses correctly."""
    recorder = PromptRecorder()
    recorder.log("prompt1", "response1")
    assert recorder.prompts_and_responses == [
        {"prompt": "prompt1", "response": "response1"}
    ]


def test_repr():
    """Test that the PromptRecorder is represented correctly."""
    recorder = PromptRecorder()
    recorder.log("prompt1", "response1")
    assert isinstance(recorder.__repr__(), str)


def test_dataframe():
    """Test that the PromptRecorder is represented correctly as a dataframe."""
    recorder = PromptRecorder()
    recorder.log("prompt1", "response1")
    assert isinstance(recorder.dataframe(), pd.DataFrame)


def test_autorecord_no_context():
    """Test that autorecord works without a context."""
    prompt = "Hello"
    response = "Hi"
    autorecord(prompt, response)


def test_autorecord_with_context():
    """Test that autorecord works with a context."""
    prompt = "Hello"
    response = "Hi"
    recorder = PromptRecorder()
    with recorder:
        autorecord(prompt, response)
    assert len(recorder.prompts_and_responses) == 1
    assert recorder.prompts_and_responses[0]["prompt"] == prompt
    assert recorder.prompts_and_responses[0]["response"] == response


def test_autorecord_multiple_with_context():
    """Test that autorecord works with a context multiple times."""
    prompts = ["Hello", "How are you?"]
    responses = ["Hi", "I'm good, thanks!"]
    recorder = PromptRecorder()
    with recorder:
        for p, r in zip(prompts, responses):
            autorecord(p, r)
    assert len(recorder.prompts_and_responses) == 2
    for i, (p, r) in enumerate(zip(prompts, responses)):
        assert recorder.prompts_and_responses[i]["prompt"] == p
        assert recorder.prompts_and_responses[i]["response"] == r
