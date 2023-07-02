"""
This module provides a top-level command line interface for interacting with apps and python.

Classes:
    - None

Exceptions:
    - None

Functions:
    - None

Objects:
    - app: A Typer instance for managing command line interface.

Modules:
    - apps: A module for managing apps-related commands.
    - python: A module for managing python-related commands.
"""

from datetime import datetime

import typer
from prompt_toolkit import prompt

from llamabot import ChatBot, PromptRecorder

from . import apps, git, python, tutorial, zotero
from .utils import configure_environment_variable

app = typer.Typer()
app.add_typer(apps.app, name="apps")
app.add_typer(python.app, name="python")
app.add_typer(git.gitapp, name="git")
app.add_typer(tutorial.app, name="tutorial")
app.add_typer(zotero.app, name="zotero")


@app.command()
def configure(
    api_key: str = typer.Option(
        ..., prompt=True, hide_input=True, confirmation_prompt=True
    )
) -> None:
    """
    Configure the API key for llamabot.

    .. code-block:: python

        configure(api_key="your_api_key_here")

    :param api_key: The API key to be used for authentication.
    """
    configure_environment_variable(env_var="OPENAI_API_KEY", env_value=api_key)


@app.command()
def version():
    """
    Print the version of llamabot.
    """
    from llamabot.version import version

    typer.echo(version)


@app.command()
def chat(save: bool = typer.Option(True, "--save", "-s")):
    """Chat with LlamaBot's ChatBot.

    :param save: Whether to save the chat to a file.
    :raises Exit: If the user types "exit" or "quit".
    """
    pr = PromptRecorder()

    bot = ChatBot(
        "You are a chatbot. Respond to the user. Ensure your responses are Markdown compatible."
    )

    # Save chat to file
    save_filename = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-chat.md"

    typer.echo("Tell me something!")
    while True:
        with pr:
            input = prompt("[You]: ")
            if input in ["exit", "quit"]:
                typer.echo("It was fun chatting! Have a great day!")
                raise typer.Exit(0)
            bot(input)
            typer.echo("\n\n")

            if save:
                pr.save(save_filename)


if __name__ == "__main__":
    app()
