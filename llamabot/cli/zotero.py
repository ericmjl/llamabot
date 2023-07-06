"""Llamabot Zotero CLI."""
import json
from datetime import date
from pathlib import Path

import typer
from caseconverter import snakecase
from dotenv import load_dotenv
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from llamabot import QueryBot
from llamabot.prompt_library.zotero import get_key, retrieverbot_sysprompt
from llamabot.recorder import PromptRecorder
from llamabot.zotero.library import ZoteroItem, ZoteroLibrary

from .utils import configure_environment_variable, exit_if_asked, uniform_prompt

load_dotenv()

app = typer.Typer()

ZOTERO_JSON_DIR = Path.home() / ".llamabot/zotero/zotero_index/"
ZOTERO_JSON_DIR.mkdir(parents=True, exist_ok=True)
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
def chat(
    query: str = typer.Argument("", help="The thing you want to chat about."),
    sync: bool = typer.Option(
        True, help="Whether or not to synchronize the Zotero library."
    ),
):
    """Chat with a paper.

    :param query: A paper to search for, whether by title, author, or other metadata.
    :param sync: Whether or not to synchronize the Zotero library.
    """
    if query == "":
        while True:
            query = prompt(
                "What paper are you searching for? ",
            )
            if query:
                break
    typer.echo("Llamabot Zotero Chatbot initializing...")
    typer.echo("Use Ctrl+C to exit anytime.")

    if sync:
        library = ZoteroLibrary()
        library.to_json(ZOTERO_JSON_DIR)
    else:
        library = ZoteroLibrary(json_dir=ZOTERO_JSON_DIR)

    with progress:
        task = progress.add_task("Embedding Zotero library...", total=None)
        retrieverbot = QueryBot(
            retrieverbot_sysprompt(),
            doc_paths=list(ZOTERO_JSON_DIR.glob("*.json")),
            stream=True,
            use_cache=True,
        )
        progress.remove_task(task)

    response = retrieverbot(get_key(query))
    paper_keys = json.loads(response.content)["key"]
    typer.echo("\n\n")
    typer.echo(f"Retrieved key: {paper_keys}")

    key_title_maps = {}
    for key in paper_keys:
        entry: ZoteroItem = library[key]
        key_title_maps[key] = entry["data.title"]
        typer.echo(f"Paper title: {key_title_maps[key]}")
    # Invert mapping:
    title_key_maps = {v: k for k, v in key_title_maps.items()}
    if len(paper_keys) > 1:
        completer = WordCompleter(list(title_key_maps.keys()))
        while True:
            user_choice = prompt(
                "Please choose an option: ",
                completer=completer,
                complete_while_typing=True,
            )
            if user_choice in title_key_maps.keys():
                break
        typer.echo(f"Awesome! You have chosen the paper: {user_choice}")
    else:
        user_choice = list(title_key_maps.keys())[0]
        typer.echo("Looks like there's only one paper, so we'll choose that one!")

    paper_key = title_key_maps[user_choice.strip(" ")]

    # Retrieve paper from library
    with progress:
        task = progress.add_task("Downloading paper...")
        entry: ZoteroItem = library[paper_key]
        fpath = entry.download_pdf(Path("/tmp"))
        progress.remove_task(task)

    typer.echo(f"Downloaded paper to {fpath}")

    with progress:
        task = progress.add_task("Embedding paper and initializing bot...")
        docbot = QueryBot(
            "You are an expert in answering questions about a paper.",
            doc_paths=[fpath],
        )
        progress.remove_task(task)

    # From this point onwards, we need to record the chat.
    pr = PromptRecorder()
    date_str = date.today().strftime("%Y%m%d")
    snaked_user_choice = f"{snakecase(user_choice)}"
    save_path = Path(f"{date_str}_{snaked_user_choice}.md")
    with pr:
        typer.echo("\n\n")
        typer.echo("Here is a summary of the paper for you to get going:")
        docbot("What is the summary of the paper?")
        typer.echo("\n\n")
        pr.save(save_path)

    while True:
        with pr:
            query = uniform_prompt()
            exit_if_asked(query)
            response = docbot(query)
            typer.echo("\n\n")

            # Want to append YYYYMMDD before filename.
            pr.save(save_path)
