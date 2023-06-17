"""
A Python bot for generating module-level and function docstrings, and code generation.

This module provides a command-line interface for generating module-level and function
docstrings, as well as generating code based on a given description. It exports the
following commands:

- module_docstrings: Generate module-level docstrings for a given module file.
- function_docstrings: Generate function docstrings for a specific function in a module file.
- code_generator: Generate code based on a given description.

Classes:
    None

Exceptions:
    None

Functions:
    module_docstrings(module_fpath: Path)
    function_docstrings(module_fpath: Path, function_name: str, style: str = "sphinx")
    code_generator(request: str)
"""

from pathlib import Path

import pyperclip
from typer import Typer

from llamabot.bot_library.coding import (
    codebot,
    docstring,
    get_function_source,
    ghostwriter,
    module_doc,
)
from llamabot.file_finder import read_file
from llamabot.utils import get_valid_input

app = Typer()


@app.command()
def module_docstrings(module_fpath: Path):
    """Generate module-level docstrings.

    :param module_fpath: Path to the module to generate docstrings for.
    """
    accepted = False
    while not accepted:
        source_code = read_file(module_fpath)
        module_docstring = codebot(module_doc(source_code))
        print("\n\n")
        user_response = get_valid_input("Do you accept this docstring? (y/n) ")
        if user_response == "y":
            accepted = True
            break
        elif user_response == "n":
            continue
    if accepted:
        pyperclip.copy(module_docstring.content)
        print("Copied to clipboard!")


@app.command()
def function_docstrings(module_fpath: Path, function_name: str, style: str = "sphinx"):
    """Generate function docstrings.

    :param module_fpath: Path to the module to generate docstrings for.
    :param function_name: Name of the function to generate docstrings for.
    :param style: Style of docstring to generate.
    """
    accepted = False
    while not accepted:
        # source_code = read_file(module_fpath)
        function_source = get_function_source(module_fpath, function_name)
        function_docstring = codebot(docstring(function_source, style=style))
        print("\n\n")
        user_response = get_valid_input("Do you accept this docstring? (y/n) ")
        if user_response == "y":
            accepted = True
            break
        elif user_response == "n":
            continue
    if accepted:
        pyperclip.copy(function_docstring.content)
        print("Copied to clipboard!")


@app.command()
def code_generator(request: str):
    """Generate code.

    :param request: A description of what the code should do.
    """
    accepted = False
    while not accepted:
        code = codebot(ghostwriter(request, "Python"))
        user_response = get_valid_input("Do you accept this code? (y/n) ")
        if user_response == "y":
            accepted = True
            break
        elif user_response == "n":
            continue
    if accepted:
        pyperclip.copy(code.content)
        print("Copied to clipboard!")
