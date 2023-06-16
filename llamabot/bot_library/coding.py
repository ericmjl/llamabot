"""Prompts and bots for coding.

codebot is an instance of SimpleBot.
The way to use codebot is as follows:

```python
prompt = "..." # some string
output = codebot(prompt)
```

Prompts are composed from the python functions that are inside here.
So we can do something like:

```python
output = codebot(one_of_these_functions(argument))
```
"""

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
def docstring(code):
    """Please help me write docstrings for the following code.

    Always use the sphinx-style docstrings if coding in Python.

    Ensure that you use Markdown Python block(s) to showcase how the code should be used.
    The code usage example should be before the parameter/argument documentation.
    Do not use sphinx-style directives,
    but instead use Markdown-style triple back-ticks to house the code block.

    Do not include :type: or :rtype: in the docstring
    as they should be covered by the type hints.

    {{ code }}

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
