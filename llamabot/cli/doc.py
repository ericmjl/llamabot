"""Document chat bots."""
from pathlib import Path

import typer

from llamabot import QueryBot

from .utils import exit_if_asked, uniform_prompt

app = typer.Typer()


@app.command()
def chat(
    doc_path: Path = typer.Argument(
        "", help="Path to the document you wish to chat with."
    )
):
    """Chat with your document.

    :param doc_path: Path to the document you wish to chat with.
    """
    bot = QueryBot(
        system_message="You are a bot that can answer questions about a document provided to you.",
        doc_path=doc_path,
    )
    typer.echo(
        "I've embedded your document. Ask me anything! Otherwise, type 'exit' or 'quit' at anytime to exit."
    )

    while True:
        query = uniform_prompt()
        exit_if_asked(query)
        bot(query)
        typer.echo("\n\n")
