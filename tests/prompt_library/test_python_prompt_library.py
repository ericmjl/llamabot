"""Tests for the Python prompt library."""

from hypothesis import given, settings
from hypothesis import strategies as st

from llamabot.bot.simplebot import SimpleBot
from llamabot.components.messages import BaseMessage
from llamabot.prompt_library.python import codebot, docstring, ghostwriter


@given(st.text())
@settings(deadline=None)
def test_codebot_instance(input_text: str):
    """Test that codebot returns a SimpleBot instance.

    :param input_text: The input text to pass to the codebot."""
    bot = codebot()
    assert isinstance(bot, SimpleBot)


@given(
    desired_functionality=st.text(),
    language=st.sampled_from(
        [
            "Python",
            "Java",
            "JavaScript",
            "C",
            "C++",
            "C#",
            "Ruby",
            "Go",
            "Rust",
            "Swift",
            "Kotlin",
            "TypeScript",
            "PHP",
            "Perl",
            "Objective-C",
            "Shell",
            "SQL",
            "HTML",
            "CSS",
            "R",
            "MATLAB",
            "Scala",
            "Groovy",
            "Lua",
            "Haskell",
            "Elixir",
            "Julia",
            "Dart",
            "VB.NET",
            "Assembly",
            "F#",
        ]
    ),
)
@settings(deadline=None)
def test_ghostwriter(desired_functionality: str, language: str):
    """Test the ghostwriter function with various inputs.

    :param desired_functionality: The desired functionality to pass to ghostwriter.
    :param language: The language to pass to ghostwriter.
    """
    result = ghostwriter(desired_functionality, language)
    assert isinstance(result, BaseMessage)
    assert result.content != ""


@given(code=st.text(), style=st.sampled_from(["sphinx", "numpy", "google"]))
@settings(deadline=None)
def test_docstring(code: str, style: str):
    """Test that the docstring function generates a docstring with the specified style.

    :param code: The code to pass to the docstring function.
    :param style: The style to pass to the docstring function.
    """
    docstring_prompt = docstring(code, style)
    assert isinstance(docstring_prompt, BaseMessage)
    assert docstring_prompt.content != ""
