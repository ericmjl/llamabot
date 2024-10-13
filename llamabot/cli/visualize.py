"""Launch the web app to visualize and compare prompts and messages."""

import typer
import uvicorn
from pathlib import Path
from pyprojroot import here
from llamabot.web.app import create_app

app = typer.Typer()


@app.command()
def launch(db_path: Path = here() / "message_log.db"):
    """Launch the web app to visualize and compare prompts and messages.

    :param db_path: The path to the database to use.
    """

    fastapi_app = create_app(db_path)
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    app()
