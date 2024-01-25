"""Serve up LlamaBots as a FastAPI endpoint."""
from pathlib import Path
import typer
from typing import List
from fastapi import FastAPI
from llamabot.bot import QueryBot
import uvicorn

api = FastAPI()
cli = typer.Typer()


@cli.command()
def querybot(
    system_prompt: str = typer.Argument(..., help="System prompt."),
    collection_name: str = typer.Argument(..., help="Name of the collection."),
    document_paths: List[Path] = typer.Argument(..., help="Paths to the documents."),
    model_name: str = typer.Argument(
        "mistral/mistral-medium", help="Name of the model to use."
    ),
    host: str = typer.Argument("0.0.0.0", help="Host to serve the API on."),
    port: int = typer.Argument(6363, help="Port to serve the API on."),
):
    """Serve up a LlamaBot as a FastAPI endpoint."""
    bot = QueryBot(
        system_prompt=system_prompt,
        collection_name=collection_name,
        document_paths=document_paths,
        model_name=model_name,
    )

    api.add_api_route(
        "/",
        bot.create_endpoint(),
    )

    uvicorn.run(api, host=host, port=port)
