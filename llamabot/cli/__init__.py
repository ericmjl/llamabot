"""Top-level command line interface."""
import typer

from . import apps

app = typer.Typer()
app.add_typer(apps.app, name="apps")

if __name__ == "__main__":
    app()
