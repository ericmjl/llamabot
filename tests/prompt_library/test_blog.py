"""Tests for blogging prompts"""
from llamabot.prompt_library.blog import (
    compose_linkedin_post,
    compose_patreon_post,
    compose_twitter_post,
)
import jinja2
import pytest


@pytest.mark.parametrize(
    "prompt_func, args",
    [
        (compose_linkedin_post, ("This is a blog post.",)),
        (compose_patreon_post, ("This is a blog post.",)),
        (compose_twitter_post, ("This is a blog post.",)),
    ],
)
def test_prompt_library(prompt_func, args):
    """Test that the prompt library returns the expected output.

    :param prompt_func: The prompt function to test.
    :param args: The arguments to pass into the prompt function.
    """
    template = jinja2.Template(prompt_func(*args))
    assert template.render() == prompt_func(*args)
