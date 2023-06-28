"""Git subcommand for LlamaBot CLI."""

import os
from tempfile import NamedTemporaryFile

import git
from rich.progress import Progress, SpinnerColumn, TextColumn
from typer import Typer

from llamabot.cli.utils import get_valid_input
from llamabot.code_manipulation import get_git_diff
from llamabot.prompt_library.git import commitbot, write_commit_message

gitapp = Typer()


@gitapp.command()
def commit(autocommit: bool = True):
    """Commit staged changes.

    :param autocommit: Whether to automatically commit the changes.
    """
    repo = git.Repo(search_parent_directories=True)
    bot = commitbot()
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    )

    while True:
        diff = get_git_diff()
        if not diff:
            print(
                "I don't see any staged changes to commit! Please stage some files before running me again."
            )
            return

        message = bot(write_commit_message(diff))
        print("\n\n")
        user_response = get_valid_input(
            "Do you accept this commit message?", ("y", "n", "m")
        )
        if user_response == "y":
            break
        elif user_response == "m":
            # Provide option to edit the commit message
            with NamedTemporaryFile(mode="w+", delete=False) as temp_file:
                temp_file.write(message.content)
                temp_file.flush()
                editor = os.getenv(
                    "EDITOR", "nano"
                )  # Use the system's default text editor
                os.system(f"{editor} {temp_file.name}")
                temp_file.seek(0)
                edited_message = temp_file.read().strip()
                message.content = edited_message
                break

    if autocommit:
        with progress:
            progress.add_task("Committing changes", total=None)
            repo.index.commit(message.content)
            progress.add_task("Pushing changes", total=None)
            origin = repo.remote(name="origin")
            origin.push()
