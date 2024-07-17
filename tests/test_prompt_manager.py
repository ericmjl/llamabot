"""Tests for the prompt decorator."""

import pytest
from llamabot.prompt_manager import prompt


def test_prompt_no_args():
    """Test that the decorated prompt function returns the correct string."""

    @prompt
    def test_func():
        """This is a test function."""

    assert test_func() == "This is a test function."


def test_prompt_with_args():
    """Test that the decorated prompt function returns the correct string with args."""

    @prompt
    def test_func(a, b):
        """This function takes two arguments: {{a}} and {{b}}. # noqa: DAR101"""

    assert (
        test_func(1, 2) == "This function takes two arguments: 1 and 2. # noqa: DAR101"
    )


def test_prompt_with_kwargs():
    """Test that the decorated prompt function
    returns the correct string with kwargs."""

    @prompt
    def test_func(a=1, b=2):
        """This function takes two arguments: {{a}} and {{b}}. # noqa: DAR101"""

    assert (
        test_func(a=3, b=4)
        == "This function takes two arguments: 3 and 4. # noqa: DAR101"
    )


def test_prompt_with_args_and_kwargs():
    """Test that the decorated prompt function
    returns the correct string with args and kwargs."""

    @prompt
    def test_func(a, b=2):
        """This function takes two arguments: {{a}} and {{b}}. # noqa: DAR101"""

    assert (
        test_func(1, b=3)
        == "This function takes two arguments: 1 and 3. # noqa: DAR101"
    )


def test_prompt_with_missing_kwargs():
    """Test that the decorated prompt function
    raises an error when kwargs are missing."""

    @prompt
    def test_func(a=1, b=2):
        """This function takes two arguments: {{a}} and {{b}}. # noqa: DAR101"""

    with pytest.raises(ValueError):
        test_func(a=3)
