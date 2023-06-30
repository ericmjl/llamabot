"""Llamabot Zotero CLI."""
import json
from pathlib import Path

import typer
from dotenv import load_dotenv
from prompt_toolkit import prompt
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from llamabot import QueryBot
from llamabot.prompt_library.zotero import get_key, retrieverbot_sysprompt
from llamabot.zotero.library import ZoteroItem, ZoteroLibrary
from llamabot.zotero.utils import load_zotero

from .utils import configure_environment_variable

load_dotenv()

app = typer.Typer()

ZOTERO_JSON_PATH = Path.home() / ".llamabot/zotero/zotero_index.json"
ZOTERO_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
progress = Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    transient=False,
)


console = Console()


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
        task = progress.add_task("Syncing Zotero items...")
        items = zot.everything(zot.items())
        progress.stop_task(task)

    # Filter items for only things with PDFs.

    with open(ZOTERO_JSON_PATH, "w+") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


@app.command()
def chat(query: str):
    """Chat with a paper.

    :param query: A paper to search for, whether by title, author, or other metadata.
    """
    typer.echo("Llamabot Zotero Chatbot initializing...")
    typer.echo("Use Ctrl+C to exit.")

    library = ZoteroLibrary()

    with progress:
        task = progress.add_task("Embedding Zotero library...")
        retrieverbot = QueryBot(
            retrieverbot_sysprompt(),
            doc_path=ZOTERO_JSON_PATH,
            stream=True,
        )
        progress.stop_task(task)

    response = retrieverbot(get_key(query))
    paper_keys = json.loads(response.content)["key"]
    typer.echo(f"Retrieved key: {paper_keys}")

    key_title_maps = {}
    for key in paper_keys:
        entry: ZoteroItem = library[key]
        key_title_maps[key] = entry["data.title"]
        typer.echo(f"Paper title: {key_title_maps[key]}")
    # Invert mapping:
    title_key_maps = {v: k for k, v in key_title_maps.items()}

    from prompt_toolkit.completion import WordCompleter

    completer = WordCompleter(list(title_key_maps.keys()))
    while True:
        user_choice = prompt(
            "Please choose an option: ", completer=completer, complete_while_typing=True
        )
        if user_choice in title_key_maps.keys():
            break
    typer.echo(f"Awesome! You have chosen the paper: {user_choice}")
    paper_key = title_key_maps[user_choice.strip(" ")]

    # Retrieve paper from library
    with progress:
        task = progress.add_task("Downloading paper...")
        entry: ZoteroItem = library[paper_key]
        fpath = entry.download_pdf(Path("/tmp"))
        progress.stop_task(task)
    typer.echo(f"Downloaded paper to {fpath}")

    with progress:
        task = progress.add_task("Embedding paper and initializing bot...")
        docbot = QueryBot(
            "You are an expert in answering questions about a paper.",
            doc_path=fpath,
            temperature=0.3,
        )
        progress.stop_task(task)

    with progress:
        typer.echo("\n\n")
        typer.echo("Here is a summary of the paper for you to get going:")
        task = progress.add_task("Summarizing paper...")
        docbot("What is the summary of the paper?")
        progress.stop_task(task)
        typer.echo("\n\n")

    while True:
        query = prompt("Ask me a question: ")
        task = progress.add_task("Sending query...")
        response = docbot(query)
        progress.stop_task(task)
        print("\n\n")
