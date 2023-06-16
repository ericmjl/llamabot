"""Run llamabot codebot panel apps."""
import typer

from llamabot.bot_library import coding

app = typer.Typer()


@app.command()
def codebot():
    """Run the codebot app."""

    codebot_app = coding.create_panel_app()
    codebot_app.show()


if __name__ == "__main__":
    app()
