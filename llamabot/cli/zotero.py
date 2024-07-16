"""Llamabot Zotero CLI."""

from datetime import date
from pathlib import Path

import typer
from caseconverter import snakecase
from dotenv import load_dotenv
from prompt_toolkit import prompt
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from slugify import slugify

from llamabot import QueryBot
from llamabot.recorder import PromptRecorder
from llamabot.zotero.library import ZoteroItem, ZoteroLibrary
from llamabot.zotero.completer import PaperTitleCompleter
from llamabot.prompt_library.zotero import paper_summary, docbot_sysprompt
from llamabot.config import default_language_model
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
    model_name: str = default_language_model(),
):
    """Chat with a paper.

    :param query: A paper to search for, whether by title, author, or other metadata.
    :param sync: Whether or not to synchronize the Zotero library.
    :param model_name: The name of the model to use.
    """
    typer.echo("Llamabot Zotero Chatbot initializing...")
    typer.echo("Use Ctrl+C to exit anytime.")

    if sync:
        library = ZoteroLibrary()
        library.to_json(ZOTERO_JSON_DIR)
    else:
        library = ZoteroLibrary(json_dir=ZOTERO_JSON_DIR)

    completer = PaperTitleCompleter(library.key_title_map().values())
    while True:
        user_choice = prompt(
            "Please choose an paper: ",
            completer=completer,
            complete_while_typing=True,
        )
        if user_choice in library.key_title_map().values():
            break
    typer.echo(f"Awesome! You have chosen the paper: {user_choice}")

    paper_key = library.key_title_map(inverse=True)[user_choice.strip(" ")]

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
            docbot_sysprompt(),
            collection_name=slugify(user_choice)[:63],
            document_paths=[fpath],
            model_name=model_name,
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
        docbot(paper_summary())
        typer.echo("\n\n")
        pr.save(save_path)

    while True:
        with pr:
            query = uniform_prompt()
            exit_if_asked(query)
            docbot(query)
            typer.echo("\n\n")

            # Want to append YYYYMMDD before filename.
            pr.save(save_path)
