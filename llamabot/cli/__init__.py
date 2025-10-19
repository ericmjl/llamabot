"""Top-level command line interface for llamabot."""

import os
from datetime import datetime
from pathlib import Path

import typer

from llamabot import SimpleBot, LanceDBDocStore

from . import (
    blog,
    configure,
    doc,
    docs,
    git,
    logviewer,
    mcp,
    notebook,
    repo,
    tutorial,
)
from .utils import exit_if_asked, uniform_prompt

app = typer.Typer()
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
    docs.app, name="docs", help="Create Markdown documentation from source files."
)
app.add_typer(notebook.app, name="notebook", help="Explain your notebooks.")
app.add_typer(logviewer.app, name="log-viewer", help="Visualize the LlamaBot logs.")
app.add_typer(
    mcp.app,
    name="mcp",
    help="MCP server for exposing LlamaBot documentation to AI agents.",
)


@app.command()
def version():
    """
    Print the version of llamabot.
    """
    from llamabot.version import version

    typer.echo(version)


@app.command()
def chat(
    model_name: str = typer.Option(..., help="The name of the model to use."),
):
    """Chat with LlamaBot's ChatBot.

    :param save: Whether to save the chat to a file.
    """

    memory = LanceDBDocStore(
        table_name=f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-chat",
        storage_path=Path.home() / ".llamabot" / "lancedb",
    )

    bot = SimpleBot(
        system_prompt="You are a chatbot. Respond to the user. "
        "Ensure your responses are Markdown compatible.",
        model_name=model_name,
        chat_memory=memory,
    )

    while True:
        input = uniform_prompt()
        exit_if_asked(input)
        bot(input)
        typer.echo("\n\n")


@app.command()
def clear_cache():
    """Clear the Llamabot cache."""
    CACHE_DIR = Path.home() / ".llamabot" / "cache"
    os.system("rm -rf {}".format(CACHE_DIR))


if __name__ == "__main__":
    app()
