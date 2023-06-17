"""Prompts and bots for coding.

This module provides a codebot instance and several functions for generating code, docstrings, and tests.
It also includes a function to create a panel app for codebot.

Functions:
    - ghostwriter(desired_functionality, language)
    - docstring(code)
    - module_doc(source_file)
    - tests(code, language)
    - create_panel_app()
"""

import inspect
import sys
from pathlib import Path
from typing import Union

import outlines.text as text
import panel as pn

from llamabot.bot.simplebot import SimpleBot
from llamabot.panel_utils import PanelMarkdownCallbackHandler

codebot = SimpleBot(
    """You are a programming expert.

You provide suggestions to programmers in Python by default,
but can also suggest in other programming languages when prompted.

Please write code without explaining it.
Do not explain your code, only provide code.
"""
)


@text.prompt
def ghostwriter(desired_functionality, language):
    """I would like to accomplish the following.

    {{ desired_functionality }}

    Please return code in the {{ language }} programming language.

    If writing in Python, ensure that there are type hints in the function.
    Ensure that the implementation of the function results in
    the simplest type hints possible.

    Ensure that within any errors raised, the error message is actionable
    and informs the user what they need to do to fix the error.
    Make sure the error message is prescriptive,
    possibly even more verbose than necessary,
    and includes verbiage such as "please do X" or "please do not do Y".
    # noqa: DAR101
    """


@text.prompt
def docstring(code, style="sphinx"):
    """Please help me write docstrings for the following code.

    {{ code }}

    Ensure that the docstring is written in {{ style }} style.

    Ensure that the code usage example is located before the arguments documentation.
    Ensure that the code usage example is renderable using Markdown directives.

    Do not include any typing information in the docstring
    as they should be covered by the type hints.

    Do not include the original source code in your output,
    return only the docstring starting from the triple quotes
    and ending at the triple quotes.
    Do not include the original function signature either.
    Write only the docstring and nothing else.

    # noqa: DAR101
    """


@text.prompt
def module_doc(source_file):
    """Please help me write module-level docstrings for the following code.

    {{ source_file }}

    Module-level docstrings have the following specification:

    The docstring for a module should generally list the classes,
    exceptions and functions (and any other objects)
    that are exported by the module, with a one-line summary of each.
    (These summaries generally give less detail
    than the summary line in the object's docstring.)
    The docstring for a package
    (i.e., the docstring of the package's __init__.py module)
    should also list the modules and subpackages exported by the package.

    # noqa: DAR101
    """


@text.prompt
def tests(code, language):
    """Please help me write unit tests for the following code.

    {{ code }}

    Ensure that the tests are written in the {{ language }} programming language.

    Prefer the use of property-based tests over example-based tests.
    Only suggest example-based tests
    if it is too difficult to generate property-based tests.
    For each test, please ensure that there is documentation
    that explains what the test is testing.

    If testing in Python,
    use pytest-style test functions and not Unittest-style test classes.

    # noqa: DAR101
    """


def get_function_source(file_path: Union[str, Path], function_name: str) -> str:
    """
    Get the source code of a function from a specified Python file.

    .. code-block:: python

        source_code = get_function_source("path/to/your/file.py", "function_name")

    :param file_path: The path to the Python file containing the function.
    :param function_name: The name of the function to get the source code from.
    :raises FileNotFoundError: If the provided file path is not found.
    :raises ValueError: If the provided file is not a .py file.
    :raises AttributeError: If the specified function is not found in the file.
    :raises TypeError: If the specified name is not a function.
    :return: The source code of the specified function as a string.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found. Please provide a valid file path: {file_path}"
        )

    if not file_path.suffix == ".py":
        raise ValueError(f"Invalid file type. Please provide a .py file: {file_path}")

    sys.path.insert(0, str(file_path.parent))
    module_name = file_path.stem
    module = __import__(module_name)

    function = getattr(module, function_name, None)
    if function is None:
        raise AttributeError(
            f"Function '{function_name}' not found in {file_path}. Please provide a valid function name."
        )

    if not inspect.isfunction(function):
        raise TypeError(
            f"'{function_name}' is not a function. Please provide a valid function name."
        )

    return inspect.getsource(function)


def create_panel_app() -> pn.Column:
    """Create a panel app for codebot.

    :return: The panel app.
    """
    programming_languages = [
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
    language = pn.widgets.Select(name="Select Language", options=programming_languages)

    user_specification = pn.widgets.TextAreaInput(
        name="User Specification",
        placeholder="Please write me a function that generates Fibonacci numbers.",
    )
    code_output = pn.pane.Markdown()
    code_output.object = "_Generated code will show up here._"
    test_output = pn.pane.Markdown()
    test_output.object = "_Generated unit tests will show up here._"

    def generate_code(event):
        """Callback for the code generator button.

        :param event: The button click event.
        """
        code_output.object = f"```{language.value}\n"
        markdown_handler = PanelMarkdownCallbackHandler(code_output)
        codebot.model.callbacks.set_handler(markdown_handler)
        code_text = codebot(ghostwriter(user_specification.value, language.value))
        code_output.object = f"```{language.value}\n{code_text.content}\n```"

    def generate_tests(event):
        """Callback for the test generator button.

        :param event: The button click event.
        """
        test_output.object = f"```{language.value}\n"
        markdown_handler = PanelMarkdownCallbackHandler(test_output)
        codebot.model.callbacks.set_handler(markdown_handler)
        test_text = codebot(tests(code_output.object, language.value))
        test_output.object = f"```{language.value}\n{test_text.content}\n```"

    generate_button = pn.widgets.Button(name="Generate Code")
    generate_button.on_click(generate_code)

    test_button = pn.widgets.Button(name="Generate Unit Tests")
    test_button.on_click(generate_tests)

    return pn.Column(
        language,
        user_specification,
        generate_button,
        code_output,
        test_button,
        test_output,
    )
