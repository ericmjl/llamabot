"""Example Feynman bot at the CLI."""

import typer

from llamabot import SimpleBot

feynman = SimpleBot(
    "You are Richard Feynman. "
    "You will be given a difficult concept, and your task is to explain it back."
)


def ask_feynman(text: str):
    """Ask Feynman.

    :param text: Text to ask Feynman.
    """
    result = feynman(text)
    print(result)


if __name__ == "__main__":
    typer.run(ask_feynman)
