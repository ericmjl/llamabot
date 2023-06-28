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

import typer

from . import apps, git, python, tutorial, zotero
from .utils import configure_environment_variable

app = typer.Typer()
app.add_typer(apps.app, name="apps")
app.add_typer(python.app, name="python")
app.add_typer(git.gitapp, name="git")
app.add_typer(tutorial.app, name="tutorial")
app.add_typer(zotero.app, name="zotero")


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
    configure_environment_variable(env_var="OPENAI_API_KEY", env_value=api_key)


@app.command()
def version():
    """
    Print the version of llamabot.
    """
    from llamabot.version import version

    typer.echo(version)


if __name__ == "__main__":
    app()
