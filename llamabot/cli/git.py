"""Git subcommand for LlamaBot CLI."""
from pathlib import Path
import os
from tempfile import NamedTemporaryFile

import git
from sh import pre_commit
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
    # Run pre-commit hooks first so that we don't waste tokens if hooks fail.
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
                "I don't see any staged changes to commit! "
                "Please stage some files before running me again."
            )
            return
        pre_commit("run")

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


@gitapp.command()
def install_commit_message_hook():
    """Install a commit message hook that runs the commit message through the bot.

    :raises RuntimeError: If the current directory is not a git repository root.
    """
    # Check that we are in a repository's root. There should be a ".git" folder.
    # Use pathlib to verify.
    if not Path(".git").exists():
        raise RuntimeError(
            "You must be in a git repository root folder to use this command. "
            "Please `cd` into your git repo's root folder and try again, "
            "or use `git init` to create a new repository (if you haven't already)."
        )

    with open(".git/hooks/prepare-commit-msg", "w+") as f:
        contents = """#!/bin/sh
llamabot git autowrite-commit-message
"""
        f.write(contents)
    os.chmod(".git/hooks/prepare-commit-msg", 0o755)
    print("Commit message hook successfully installed!")


@gitapp.command()
def autowrite_commit_message():
    """Autowrite commit message based on the diff."""
    try:
        diff = get_git_diff()
        bot = commitbot()
        message = bot(write_commit_message(diff))
        with open(".git/COMMIT_EDITMSG", "w+") as commit_msg_file:
            commit_msg_file.write(message.content)
    except Exception as e:
        print(f"Error encountered: {e}")
        print("Please write your own commit message.")
