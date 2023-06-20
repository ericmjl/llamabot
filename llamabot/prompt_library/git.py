"""Prompts for writing git stuff."""
from outlines import text

from llamabot import SimpleBot


def commitbot():
    """Return a commitbot instance.

    :return: A commitbot instance.
    """
    return SimpleBot("You are an expert user of Git.")


@text.prompt
def write_commit_message(diff: str):
    """Please write a commit message for the following diff.

    {{ diff }}

    # noqa: DAR101
    """
