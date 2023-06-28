"""Tutorial bots."""
from pathlib import Path

from typer import Typer

from llamabot.prompt_library.tutorial import (  # typer_cli_tutorial,
    module_tutorial_writer,
    tutorialbot,
)

app = Typer()


@app.command()
def writer(source_file: Path, tutorial_path: Path = None):
    """Write a tutorial for a given source file.

    :param source_file: Path to the source file to write a tutorial for.
    :param tutorial_path: Path to the tutorial file to write.
    """
    while True:
        source_code = source_file.read_text()
        bot = tutorialbot()
        tutorial = bot(module_tutorial_writer(source_code))
        print("\n\n")
        user_response = input("Do you accept this tutorial? (y/n) ")
        if user_response == "y":
            tutorial_path.write_text(tutorial.content)
            break
