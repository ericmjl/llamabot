"""CLI for chatting with a code repository."""

from typing import List, Optional
import typer
import git
import tempfile
from pathlib import Path
from llamabot import QueryBot
from .utils import exit_if_asked, uniform_prompt
from slugify import slugify
from loguru import logger

app = typer.Typer()


@app.command()
def chat(
    repo_url: str,
    checkout: str = "main",
    source_file_extensions: List[str] = [
        "py",
        "jl",
        "R",
        "ipynb",
        "md",
        "tex",
        "txt",
        "lr",
        "rst",
    ],
    model_name: str = "gpt-4-0125-preview",
    initial_message: Optional[str] = None,
    panel: bool = True,
):
    """Chat with a code repository."""
    # Create a temporary directory
    logger.info("Creating temporary directory...")
    temp_dir = tempfile.TemporaryDirectory(dir="/tmp")

    # Clone the repository into the temporary directory
    logger.info("Cloning repository...")
    repo = git.Repo.clone_from(repo_url, temp_dir.name)

    # checkout the specified branch or tag (i.e. "checkout")
    logger.info(f"Checking out branch {checkout}...")
    repo.git.checkout(checkout)

    # Set the root directory to the cloned repository
    root_dir = Path(temp_dir.name)

    source_files = []
    for extension in source_file_extensions:
        files = list(root_dir.rglob(f"*.{extension}"))
        print(f"Found {len(files)} files with extension {extension}.")
        source_files.extend(files)

    stream_target = "stdout"
    if panel:
        stream_target = "panel"

    bot = QueryBot(
        system_prompt="You are a knowledgeable git repository author. Your answers come from the repository. If the answer is not in the repository, say 'I don't know'.",
        collection_name=slugify(repo_url + ":" + checkout),
        document_paths=source_files,
        model_name=model_name,
        initial_message=initial_message,
        stream_target=stream_target,
    )
    if panel:
        print("Serving your document in a panel...")
        bot.serve()

    else:
        while True:
            query = uniform_prompt()
            exit_if_asked(query)
            bot(query)
            typer.echo("\n\n")
