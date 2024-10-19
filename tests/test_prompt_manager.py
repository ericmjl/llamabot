"""Tests for the prompt decorator."""

import pytest
from llamabot.components.messages import BaseMessage
from llamabot.prompt_manager import prompt
from llamabot.recorder import hash_template


def test_prompt_no_args():
    """Test that the decorated prompt function returns the correct string."""

    @prompt(role="system")
    def test_func():
        """This is a test function."""

    assert test_func() == BaseMessage(
        role="system",
        content=test_func.__doc__,
        prompt_hash=hash_template(test_func.__doc__),
    )


def test_prompt_with_args():
    """Test that the decorated prompt function returns the correct string with args."""

    @prompt(role="system")
    def test_func(a, b):
        """This function takes two arguments: {{a}} and {{b}}. # noqa: DAR101"""

    assert test_func(1, 2) == BaseMessage(
        role="system",
        content="This function takes two arguments: 1 and 2. # noqa: DAR101",
        prompt_hash=hash_template(test_func.__doc__),
    )


def test_prompt_with_kwargs():
    """Test that the decorated prompt function
    returns the correct string with kwargs."""

    @prompt(role="system")
    def test_func(a=1, b=2):
        """This function takes two arguments: {{a}} and {{b}}. # noqa: DAR101"""

    assert test_func(a=3, b=4) == BaseMessage(
        role="system",
        content="This function takes two arguments: 3 and 4. # noqa: DAR101",
        prompt_hash=hash_template(test_func.__doc__),
    )


def test_prompt_with_args_and_kwargs():
    """Test that the decorated prompt function
    returns the correct string with args and kwargs."""

    @prompt(role="system")
    def test_func(a, b=2):
        """This function takes two arguments: {{a}} and {{b}}. # noqa: DAR101"""

    assert test_func(1, b=3) == BaseMessage(
        role="system",
        content="This function takes two arguments: 1 and 3. # noqa: DAR101",
        prompt_hash=hash_template(test_func.__doc__),
    )


def test_prompt_with_missing_kwargs():
    """Test that the decorated prompt function
    raises an error when kwargs are missing."""

    @prompt(role="system")
    def test_func(a=1, b=2):
        """This function takes two arguments: {{a}} and {{b}}. # noqa: DAR101"""

    with pytest.raises(ValueError):
        test_func(a=3)
