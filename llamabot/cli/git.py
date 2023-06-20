"""Git subcommand for LlamaBot CLI."""

import pyperclip
from typer import Typer

from llamabot.code_manipulation import get_git_diff
from llamabot.prompt_library.git import commitbot, write_commit_message

from .utils import get_valid_input

app = Typer()


@app.command
def hello():
    """Say hello."""
    print("Hello!")


@app.command
def commit_message():
    """Generate a commit message."""
    bot = commitbot()

    while True:
        message = bot(write_commit_message(get_git_diff()))
        user_response = get_valid_input("Do you accept this docstring? (y/n) ")
        if user_response == "y":
            break
    pyperclip.copy(message.content)
