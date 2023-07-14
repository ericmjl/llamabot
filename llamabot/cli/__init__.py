"""Top-level command line interface for llamabot."""

import os
from datetime import datetime
from pathlib import Path

import typer

from llamabot import ChatBot, PromptRecorder

from . import apps, blog, configure, doc, git, python, tutorial, zotero
from .utils import exit_if_asked, uniform_prompt

app = typer.Typer()
app.add_typer(apps.app, name="apps")
app.add_typer(python.app, name="python")
app.add_typer(git.gitapp, name="git")
app.add_typer(tutorial.app, name="tutorial")
app.add_typer(zotero.app, name="zotero")
app.add_typer(doc.app, name="doc")
app.add_typer(blog.app, name="blog")
app.add_typer(configure.app, name="configure")


@app.command()
def version():
    """
    Print the version of llamabot.
    """
    from llamabot.version import version

    typer.echo(version)


@app.command()
def chat(save: bool = typer.Option(True, help="Whether to save the chat to a file.")):
    """Chat with LlamaBot's ChatBot.

    :param save: Whether to save the chat to a file.
    """
    pr = PromptRecorder()

    bot = ChatBot(
        "You are a chatbot. Respond to the user. Ensure your responses are Markdown compatible."
    )

    # Save chat to file
    save_filename = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-chat.md"

    while True:
        with pr:
            input = uniform_prompt()
            exit_if_asked(input)
            bot(input)
            typer.echo("\n\n")

            if save:
                pr.save(save_filename)


@app.command()
def clear_cache():
    """Clear the Llamabot cache."""
    CACHE_DIR = Path.home() / ".llamabot" / "cache"
    os.system("rm -rf {}".format(CACHE_DIR))


if __name__ == "__main__":
    app()
