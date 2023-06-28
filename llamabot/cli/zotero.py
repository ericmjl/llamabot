"""Llamabot Zotero CLI."""
import json
import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from pyzotero import zotero
from rich.progress import Progress, SpinnerColumn, TextColumn

from .utils import configure_environment_variable

load_dotenv()

app = typer.Typer()

ZOTERO_JSON_PATH = Path.home() / ".llamabot/zotero/zotero_index.json"
ZOTERO_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)


@app.command()
def configure(
    library_id: str = typer.Option(..., prompt=True),
    api_key: str = typer.Option(..., prompt=True),
    library_type: str = typer.Option(default="user", prompt=True),
):
    """Configure Llamabot Zotero CLI environment variables.

    :param library_id: Zotero library ID
    :param api_key: Zotero API key
    :param library_type: Zotero library type
    """
    configure_environment_variable("ZOTERO_LIBRARY_ID", library_id)
    configure_environment_variable("ZOTERO_API_KEY", api_key)
    configure_environment_variable("ZOTERO_LIBRARY_TYPE", library_type)


@app.command()
def sync():
    """Sync Zotero items to a local JSON file."""
    zotero_library_id = os.environ.get("ZOTERO_LIBRARY_ID", None)
    zotero_library_type = os.environ.get("ZOTERO_LIBRARY_TYPE", None)
    zotero_api_key = os.environ.get("ZOTERO_API_KEY", None)

    zot = zotero.Zotero(
        library_id=zotero_library_id,
        library_type=zotero_library_type,
        api_key=zotero_api_key,
    )

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    )
    with progress:
        progress.add_task("Syncing Zotero items...")
        items = zot.everything(zot.items())

    with open(ZOTERO_JSON_PATH, "w+") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


# @app.command()
# def chat_paper(title: str, author: str):
#     print("Llamabot Zotero Chatbot initializing...")
#     print("Use Ctrl+C to exit.")
#     pass
