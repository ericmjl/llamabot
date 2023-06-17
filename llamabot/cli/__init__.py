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

from . import apps, python

app = typer.Typer()
app.add_typer(apps.app, name="apps")
app.add_typer(python.app, name="python")

if __name__ == "__main__":
    app()
