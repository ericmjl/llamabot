"""Prompts and bots for coding.

This module provides a codebot instance and several functions
for generating code, docstrings, and tests.
It also includes a function to create a panel app for codebot.

Functions:
    - ghostwriter(desired_functionality, language)
    - docstring(code)
    - module_doc(source_file)
    - tests(code, source_file)
    - create_panel_app()
"""

from llamabot.bot.simplebot import SimpleBot
from llamabot.prompt_manager import prompt


def codebot() -> SimpleBot:
    """Return a codebot instance.

    :return: A codebot instance.
    """
    return SimpleBot(
        """You are a programming expert.

    You provide suggestions to programmers in Python by default,
    but can also suggest in other programming languages when prompted.

    Please write code without explaining it.
    Do not explain your code, only provide code.

    In your code, prefer the use of the Pathlib module
    over the os module for handling paths.
    """
    )


@prompt(role="user")
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


@prompt(role="user")
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
    Ensure that the docstring indentation is correct for the object.

    # noqa: DAR101
    """


@prompt(role="user")
def module_doc(source_file_contents, source_file_fpath=None, file_tree=None):
    """Please help me write module-level docstrings for the following code.

    {{ source_file_contents }}

    For context, this is the source file's path:

    {{ source_file_fpath }}

    And this is the file tree of the source file's directory:

    {{ file_tree }}

    Module-level docstrings have the following specification:

    The docstring for a module should generally list the classes,
    exceptions and functions (and any other objects)
    that are exported by the module, with a one-line summary of each.
    (These summaries generally give less detail
    than the summary line in the object's docstring.)
    The docstring for a package
    (i.e., the docstring of the package's __init__.py module)
    should also list the modules and subpackages exported by the package.

    Ensure that you never spit out the original source code.

    # noqa: DAR101
    """


@prompt(role="user")
def tests(code, source_file_txt: str, dependent_source_files: str):
    """I need help writing unit tests.

    For context, here is the broader source file in which the code is defined:

    {{ source_file_txt }}

    Then, there are other source files that this function depends on,
    from which you can import objects.

    {{ dependent_source_files }}

    Prefer the use of property-based tests over example-based tests.
    Only suggest example-based tests
    if it is too difficult to generate property-based tests.
    For each test function,
    ensure that there is a docstring that explains what the test is testing.

    Use pytest-style test functions and not Unittest-style test classes.

    Here is the actual thing for which I need a test:

    {{ code }}

    Please write me tests for that code.

    # noqa: DAR101
    """
