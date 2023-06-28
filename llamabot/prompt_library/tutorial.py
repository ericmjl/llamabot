"""Tutorial bot and prompts."""
from outlines import text

from llamabot import SimpleBot


@text.prompt
def tutorialbot_system_prompt():
    """
    You are an expert python tutorial writer.

    Your style is engaging without being too entertaining.

    You will be provided a library of code.
    Your task is to write a tutorial for the code.

    Do not copy code from the library; instead,
    simply presume it is imported.

    Where possible, show how the code is used in relation to one another.
    For example, the output of one function being used in a downstream function.
    Do not hallucinate code that does not exist.
    """


def tutorialbot() -> SimpleBot:
    """Return a tutorial bot.

    :return: Tutorial bot
    """
    return SimpleBot(tutorialbot_system_prompt())


@text.prompt
def module_tutorial_writer(source_file):
    """Please help me write a tutorial for the following code.

    {{ source_file }}

    It should be in Markdown format.

    # noqa: DAR101
    """


@text.prompt
def typer_cli_tutorial(source_file, additional_notes):
    """I have the following CLI source file.

    {{ source_file }}

    It is a Typer CLI module, therefore, the commands are kebab-cased
    (like this: `<clitool> some-command`),
    where `<clitool>` is replaced by the actual command line tool name.
    Please help me write a tutorial about it.
    Please provide exhaustive examples about the combinations of command arguments
    that can be used.

    I have additional notes that you can use:

    {{ additional_notes }}

    [TUTORIAL BEGIN]  # noqa: DAR101
    """
