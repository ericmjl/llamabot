"""Llamabot Zotero CLI."""
import typer

from .utils import configure_environment_variable

app = typer.Typer()


@app.command()
def configure(
    library_id: str = typer.Option(..., prompt=True),
    api_key: str = typer.Option(..., prompt=True),
    library_type: str = typer.Option(default="user", prompt=True),
):
    """Configure Llamabot Zotero CLI environment variables.

    :param library_id: Zotero library ID
    :param api_key: Zotero API key
    :param library_type: Zotero library type
    """
    configure_environment_variable("ZOTERO_LIBRARY_ID", library_id)
    configure_environment_variable("ZOTERO_API_KEY", api_key)
    configure_environment_variable("ZOTERO_LIBRARY_TYPE", library_type)


# @app.command()
# def sync():
#     pass


# @app.command()
# def chat_paper(title: str, author: str):
#     print("Llamabot Zotero Chatbot initializing...")
#     print("Use Ctrl+C to exit.")
#     pass
