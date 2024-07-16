"""Document chat bots."""

from pathlib import Path

import typer

from llamabot import QueryBot

from .utils import exit_if_asked, uniform_prompt

app = typer.Typer()


@app.command()
def chat(
    model_name: str = typer.Option(
        "mistral/mistral-medium", help="Name of the LLM to use."
    ),
    initial_message: str = typer.Option(..., help="Initial message for the bot."),
    panel: bool = typer.Option(True, help="Whether to use Panel or not."),
    doc_path: Path = typer.Argument(
        "", help="Path to the document you wish to chat with."
    ),
    address: str = typer.Option("0.0.0.0", help="Host to serve the API on."),
    port: int = typer.Option(6363, help="Port to serve the API on."),
):
    """Chat with your document.

    :param model_name: Name of the LLM to use.
    :param panel: Whether to use Panel or not. If not, we default to using CLI chat.
    :param initial_message: The initial message to send to the user.
    :param doc_path: Path to the document you wish to chat with.
    :param address: Host to serve the API on.
    :param port: Port to serve the API on.
    """
    stream_target = "stdout"
    if panel:
        stream_target = "panel"

    bot = QueryBot(
        system_prompt=(
            "You are a bot that can answer questions about a document provided to you. "
            "Based on the document, respond to the query that is given to you. "
            "As much as possible, quote from the context that is provided to you."
        ),
        collection_name=doc_path.stem.lower().replace(" ", "-"),
        document_paths=[doc_path],
        model_name=model_name,
        initial_message=initial_message,
        stream_target=stream_target,
    )
    typer.echo(
        (
            "I've embedded your document. "
            "Ask me anything! "
            "Otherwise, type 'exit' or 'quit' at anytime to exit."
        )
    )

    if panel:
        print("Serving your document in a panel...")
        bot.serve(address=address, port=port, websocket_origin=["*"])

    else:
        while True:
            query = uniform_prompt()
            exit_if_asked(query)
            bot(query)
            typer.echo("\n\n")
