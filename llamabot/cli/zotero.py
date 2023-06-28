"""Llamabot Zotero CLI."""
import json
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.progress import Progress, SpinnerColumn, TextColumn

from llamabot import QueryBot
from llamabot.prompt_library.zotero import get_key, retrieverbot_sysprompt
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

    zot = load_zotero()

    with progress:
        retrieverbot = QueryBot(
            retrieverbot_sysprompt(),
            doc_paths=[ZOTERO_JSON_PATH],
        )
        response = retrieverbot(get_key(title, author))
        paper_key = json.loads(response.content)["key"]
        typer.echo(f"Retrieved key: {paper_key}")

    # Retrieve paper key from Zotero library.
    with open(ZOTERO_JSON_PATH) as f:
        library = [json.loads(line) for line in f.readlines()]
    item = [item for item in library if item["key"] == "GM83SMKU"][0]
    pdf_key = item["links"]["attachment"]["href"].split("/")[-1]
    progress.add_task(description="Saving PDF to disk...", total=None)
    pdf_path = ZOTERO_JSON_PATH.parent / f"{pdf_key}.pdf"
    with open(pdf_path, "wb") as f:
        f.write(zot.file(pdf_key))

    # Create new QueryBot
    # progress.add_task(description="Embedding the paper...", total=None)
    # chatbot = QueryBot(
    #     "You are an expert in answering questions about a paper.", doc_paths=[pdf_path]
    # )
