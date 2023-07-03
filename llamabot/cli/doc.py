"""Document chat bots."""
from pathlib import Path

import typer
from prompt_toolkit import prompt

from llamabot import QueryBot

app = typer.Typer()


@app.command()
def chat(
    doc_path: Path = typer.Argument(
        "", help="Path to the document you wish to chat with."
    )
):
    """Chat with your document.

    :param doc_path: Path to the document you wish to chat with.
    :raises Exit: If the user quits.
    """
    bot = QueryBot(
        system_message="You are a bot that can answer questions about a document provided to you.",
        doc_path=doc_path,
    )
    typer.echo(
        "I've embedded your document. Ask me anything! Otherwise, type 'exit' or 'quit' at anytime to exit."
    )

    while True:
        query = prompt(">> ")
        if query in ["exit", "quit"]:
            typer.echo("Have a great day!")
            raise typer.Exit(0)
        bot(query)
        typer.echo("\n\n")
