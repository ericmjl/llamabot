"""Llamabot Zotero CLI."""
import json
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.progress import Progress, SpinnerColumn, TextColumn

from llamabot import QueryBot
from llamabot.prompt_library.zotero import get_key, retrieverbot_sysprompt
from llamabot.zotero.library import ZoteroLibrary
from llamabot.zotero.utils import load_zotero

from .utils import configure_environment_variable

load_dotenv()

app = typer.Typer()

ZOTERO_JSON_PATH = Path.home() / ".llamabot/zotero/zotero_index.json"
ZOTERO_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
progress = Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    transient=True,
)


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

    zot = load_zotero()
    with progress:
        progress.add_task("Syncing Zotero items...")
        items = zot.everything(zot.items())

    with open(ZOTERO_JSON_PATH, "w+") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


@app.command()
def chat_paper(title: str = typer.Option(""), author: str = typer.Option("")):
    """Chat with a paper.

    :param title: Paper title
    :param author: Paper author
    """
    typer.echo("Llamabot Zotero Chatbot initializing...")
    typer.echo("Use Ctrl+C to exit.")

    library = ZoteroLibrary()

    with progress:
        retrieverbot = QueryBot(
            retrieverbot_sysprompt(),
            doc_paths=[ZOTERO_JSON_PATH],
        )
        response = retrieverbot(get_key(title, author))
        paper_key = json.loads(response.content)["key"]
        typer.echo(f"Retrieved key: {paper_key}")
        typer.echo(f"Paper title: {library[paper_key]['data.title']}")

    # Retrieve paper from library
    fpath = library[paper_key].download_pdf(Path("/tmp"))
    typer.echo(f"Downloaded paper to {fpath}")

    docbot = QueryBot(
        "You are an expert in answering questions about a paper.",
        doc_paths=[fpath],
        temperature=0.3,
    )

    while True:
        query = input("Ask me a question: ")
        response = docbot(query)
        print("\n\n")
