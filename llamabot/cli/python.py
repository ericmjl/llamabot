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

from llamabot.code_manipulation import get_object_source_code, show_directory_tree
from llamabot.file_finder import read_file
from llamabot.prompt_library.coding import (
    codebot,
    docstring,
    ghostwriter,
    module_doc,
    tests,
)

from .utils import get_valid_input

app = Typer()


@app.command()
def module_docstrings(module_fpath: Path):
    """Generate module-level docstrings.

    :param module_fpath: Path to the module to generate docstrings for.
    """
    while True:
        source_code = read_file(module_fpath)
        tree_structure = show_directory_tree(module_fpath.parent)
        module_docstring = codebot(
            module_doc(source_code, module_fpath, tree_structure)
        )
        print("\n\n")
        user_response = get_valid_input("Do you accept this docstring? (y/n) ")
        if user_response == "y":
            break
    pyperclip.copy(module_docstring.content)
    print("Copied to clipboard!")


@app.command()
def generate_docstrings(module_fpath: Path, object_name: str, style: str = "sphinx"):
    """Generate function docstrings.

    :param module_fpath: Path to the module to generate docstrings for.
    :param object_name: Name of the object to generate docstrings for.
    :param style: Style of docstring to generate.
    """
    while True:
        # source_code = read_file(module_fpath)
        object_source = get_object_source_code(module_fpath, object_name)
        object_docstring = codebot(docstring(object_source, style=style))
        print("\n\n")
        user_response = get_valid_input("Do you accept this docstring? (y/n) ")
        if user_response == "y":
            break

    pyperclip.copy(object_docstring.content)
    print("Copied to clipboard!")


@app.command()
def code_generator(request: str):
    """
    Generate code.

    .. code-block:: python

        code_generator("Create a function that adds two numbers")

    :param request: A description of what the code should do.
    """
    while True:
        code = codebot(ghostwriter(request, "Python"))
        print("\n\n")
        user_response = get_valid_input("Do you accept this code? (y/n) ")
        if user_response == "y":
            break
    pyperclip.copy(code.content)
    print("Copied to clipboard!")


@app.command()
def test_writer(module_fpath: str, object_name: str):
    """Write tests for a given object.

    :param module_fpath: Path to the module to generate tests for.
    :param object_name: Name of the object to generate tests for.
    """
    while True:
        file_source = read_file(module_fpath)
        function_source = get_object_source_code(module_fpath, object_name)
        test_code = codebot(tests(function_source, file_source))
        print("\n\n")
        user_response = get_valid_input("Do you accept this test? (y/n) ")
        if user_response == "y":
            break
    pyperclip.copy(test_code.content)
    print("Copied to clipboard!")
