"""Cache-related commands."""

import typer
from diskcache import Cache

app = typer.Typer()


@app.command()
def clear():
    """
    Clear the disk cache.

    This command will remove all items stored in the disk cache.
    """
    cache = Cache()
    cache.clear()
    typer.echo("Cache cleared successfully.")
