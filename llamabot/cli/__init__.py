"""Top-level command line interface for llamabot."""

import os
from datetime import datetime
from pathlib import Path

import typer

from llamabot import ChatBot, PromptRecorder

from . import (
    blog,
    cache,
    configure,
    doc,
    git,
    python,
    tutorial,
    zotero,
    repo,
    serve,
    docs,
    notebook,
    logviewer,
)
from .utils import exit_if_asked, uniform_prompt

app = typer.Typer()
app.add_typer(
    python.app,
    name="python",
    help="Python bot for generating docstrings, code, and tests.",
)
app.add_typer(
    git.gitapp,
    name="git",
    help="Automatic commit message generation and pre-commit hook installation.",
)
app.add_typer(
    tutorial.app,
    name="tutorial",
    help="Automatically generate tutorials for given source files.",
)
app.add_typer(zotero.app, name="zotero", help="Chat with your Zotero library.")
app.add_typer(doc.app, name="doc", help="Chat with your documents.")
app.add_typer(
    blog.app,
    name="blog",
    help=(
        "Summarize blog posts, generate social media posts, "
        "and apply SEMBR to blog posts in a chat UI."
    ),
)
app.add_typer(configure.app, name="configure", help="Configure LlamaBot.")
app.add_typer(repo.app, name="repo", help="Chat with a code repository.")
app.add_typer(
    serve.cli, name="serve", help="Serve up a LlamaBot as a FastAPI endpoint."
)
app.add_typer(
    docs.app, name="docs", help="Create Markdown documentation from source files."
)
app.add_typer(notebook.app, name="notebook", help="Explain your notebooks.")
app.add_typer(cache.app, name="cache", help="Clear the LlamaBot cache.")
app.add_typer(logviewer.app, name="log-viewer", help="Visualize the LlamaBot logs.")


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
        "You are a chatbot. Respond to the user. "
        "Ensure your responses are Markdown compatible.",
        session_name=f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-chat",
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
                pr.save(Path(save_filename))


@app.command()
def clear_cache():
    """Clear the Llamabot cache."""
    CACHE_DIR = Path.home() / ".llamabot" / "cache"
    os.system("rm -rf {}".format(CACHE_DIR))


if __name__ == "__main__":
    app()
