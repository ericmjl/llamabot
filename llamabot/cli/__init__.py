"""
This module provides a top-level command line interface for interacting with apps and python.

Classes:
    - None

Exceptions:
    - None

Functions:
    - None

Objects:
    - app: A Typer instance for managing command line interface.

Modules:
    - apps: A module for managing apps-related commands.
    - python: A module for managing python-related commands.
"""
import os
import re
from pathlib import Path

import typer

from . import apps, python

app = typer.Typer()
app.add_typer(apps.app, name="apps")
app.add_typer(python.app, name="python")


@app.command()
def configure(
    api_key: str = typer.Option(
        ..., prompt=True, hide_input=True, confirmation_prompt=True
    )
) -> None:
    """
    Configure the API key for llamabot.

    .. code-block:: python

        configure(api_key="your_api_key_here")

    :param api_key: The API key to be used for authentication.
    """
    config_file = Path(os.path.expanduser("~/.llamabotrc"))
    api_key_line = f'export OPENAI_API_KEY="{api_key}"'

    if config_file.exists():
        with open(config_file, "r") as file:
            content = file.readlines()

        with open(config_file, "w") as file:
            for line in content:
                if re.match(r"export OPENAI_API_KEY=.*", line):
                    file.write(api_key_line + "\n")
                else:
                    file.write(line)
    else:
        with open(config_file, "w") as file:
            file.write(api_key_line + "\n")


if __name__ == "__main__":
    app()
