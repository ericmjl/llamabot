"""Git subcommand for LlamaBot CLI."""
from pathlib import Path
import os
from tempfile import NamedTemporaryFile

import git
from pyprojroot import here
from sh import pre_commit
from rich.progress import Progress, SpinnerColumn, TextColumn
from typer import Typer, echo

from llamabot import SimpleBot
from llamabot.cli.utils import get_valid_input
from llamabot.code_manipulation import get_git_diff
from llamabot.prompt_library.git import (
    commitbot,
    write_commit_message,
    compose_release_notes,
)

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
llamabot git compose-commit > .git/COMMIT_EDITMSG
"""
        f.write(contents)
    os.chmod(".git/hooks/prepare-commit-msg", 0o755)
    echo("Commit message hook successfully installed! 🎉")


@gitapp.command()
def compose_commit():
    """Autowrite commit message based on the diff."""
    try:
        diff = get_git_diff()
        bot = commitbot()
        msg = bot(write_commit_message(diff))
        echo(msg.content)
    except Exception as e:
        echo(f"Error encountered: {e}", err=True)
        echo("Please write your own commit message.", err=True)


@gitapp.command()
def write_release_notes(release_notes_dir: Path = Path("./docs/releases")):
    """Write release notes for the latest two tags to the release notes directory.

    :param release_notes_dir: The directory to write the release notes to.
        Defaults to "./docs/releases".
    """
    repo = git.Repo(here())
    tags = sorted(repo.tags, key=lambda t: t.commit.committed_datetime)
    if len(tags) == 0:
        # No tags, get all commit messages from the very first commit
        log_info = repo.git.log()
    elif len(tags) == 1:
        # Only one tag, get all commit messages from that tag to the current commit
        tag = tags[0]
        log_info = repo.git.log(f"{tag.commit.hexsha}..HEAD")
    else:
        # More than one tag, get all commit messages between the last two tags
        tag1, tag2 = tags[-2], tags[-1]
        log_info = repo.git.log(f"{tag1.commit.hexsha}..{tag2.commit.hexsha}")

    bot = SimpleBot(
        "You are an expert software developer "
        "who knows how to write excellent release notes based on git commit logs.",
        model_name="mistral/mistral-medium",
        api_key=os.environ["MISTRAL_API_KEY"],
        stream=False,
    )
    notes = bot(compose_release_notes(log_info))

    # Create release_notes_dir if it doesn't exist:
    release_notes_dir.mkdir(parents=True, exist_ok=True)

    # Write release notes to the file:
    with open(release_notes_dir / f"{tag2.name}.md", "w+") as f:
        f.write(notes.content)
