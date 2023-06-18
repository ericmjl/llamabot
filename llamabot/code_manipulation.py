"""Utilities for llamabot."""

import ast
import inspect
import sys
from pathlib import Path
from typing import Any, Union

import astor


def replace_object_in_file(
    source_file_path: str, object_name: str, new_object_definition: str
) -> None:
    """
    Replace an object (function or class) in a Python source file with a new object definition.

    .. code-block:: python

        source_file_path = "path/to/your/source_file.py"
        object_name = "function_or_class_name"
        new_object_definition = "def new_function():\\n    pass"
        replace_object_in_file(source_file_path, object_name, new_object_definition)

    :param source_file_path: The path to the source file containing the object to be replaced.
    :param object_name: The name of the object (function or class) to be replaced.
    :param new_object_definition: The new object definition as a string.
    :raises SyntaxError: If the source file has invalid Python syntax.
    :raises ValueError: If the object with the specified name does not exist in the source file.
    """
    with open(source_file_path, "r") as source_file:
        source_code = source_file.read()

    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        raise SyntaxError(
            "Please ensure the source file has valid Python syntax."
        ) from e

    class ObjectReplacer(ast.NodeTransformer):
        """Internal class used to replace an object in a Python source file."""

        def visit_FunctionDef(
            self, node: ast.FunctionDef
        ) -> Union[ast.FunctionDef, Any]:
            """Replace the object with the specified name.

            :param node: The node to visit.
            :return: The new node.
            """
            if node.name == object_name:
                new_node = ast.parse(new_object_definition).body[0]
                return new_node
            return node

        def visit_ClassDef(self, node: ast.ClassDef) -> Union[ast.ClassDef, Any]:
            """Replace the object with the specified name.

            :param node: The node to visit.
            :return: The new node.
            """
            if node.name == object_name:
                new_node = ast.parse(new_object_definition).body[0]
                return new_node
            return node

    replacer = ObjectReplacer()
    new_tree = replacer.visit(tree)

    if not any(
        isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == object_name
        for node in new_tree.body
    ):
        raise ValueError(
            f"Please ensure the object '{object_name}' exists in the source file."
        )

    new_source_code = astor.to_source(new_tree)

    with open(source_file_path, "w") as source_file:
        source_file.write(new_source_code)


def insert_docstring(source_file_path: str, object_name: str, new_docstring: str):
    """
    Insert a new docstring into a Python source file for a specified object.

    Usage example:

        insert_docstring("path/to/source_file.py", "function_name", "This is the new docstring.")

    :param source_file_path: The path to the source file containing the object.
    :param object_name: The name of the object (function or class) to insert the docstring for.
    :param new_docstring: The new docstring to insert.
    :raises SyntaxError: If the source code is not valid Python code.
    :raises ValueError: If the specified object does not exist in the source code.
    """
    with open(source_file_path, "r") as source_file:
        source_code = source_file.read()

    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        raise SyntaxError("Please ensure the source code is valid Python code.") from e

    class DocstringInserter(ast.NodeTransformer):
        """Insert a docstring into a Python source file."""

        def visit_FunctionDef(
            self, node: ast.FunctionDef
        ) -> Union[ast.FunctionDef, None]:
            """Visit a function definition node and insert a docstring if the function name matches the specified name.

            :param node: The function definition node.
            :return: The modified function definition node.
            """
            if node.name == object_name:
                docstring = ast.Expr(value=ast.Str(s=new_docstring))
                node.body.insert(0, docstring)
                return node
            return self.generic_visit(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> Union[ast.ClassDef, None]:
            """Visit a class definition node and insert a docstring if the class name matches the specified name.

            :param node: The class definition node.
            :return: The modified class definition node.
            """
            if node.name == object_name:
                docstring = ast.Expr(value=ast.Str(s=new_docstring))
                node.body.insert(0, docstring)
                return node
            return self.generic_visit(node)

    inserter = DocstringInserter()
    new_tree = inserter.visit(tree)

    if new_tree is None:
        raise ValueError(
            f"Please ensure the object '{object_name}' exists in the source code."
        )

    new_source_code = astor.to_source(new_tree)

    with open(source_file_path, "w") as source_file:
        source_file.write(new_source_code)


def get_object_source_code(source_file: str, object_name: str) -> Union[str, None]:
    """
    Get the source code of a specific object (function, async function, or class) from a source file.

    .. code-block:: python

        source_code = get_object_source_code("example.py", "my_function")

    :param source_file: The path to the source file containing the object.
    :param object_name: The name of the object to get the source code for.
    :return: The source code of the specified object as a string, or None if the object is not found.
    :raises SyntaxError: If there is a syntax error in the source file.
    :raises NameError: If the specified object is not found in the source file.
    """
    with open(source_file, "r+") as file:
        source_code = file.read()

    try:
        parsed_ast = ast.parse(source_code)
    except SyntaxError as e:
        raise SyntaxError(f"Please fix the syntax error in the source file: {e}")

    for node in parsed_ast.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == object_name
        ):
            return astor.to_source(node)

    raise NameError(
        f"Please ensure the object '{object_name}' exists in the source file '{source_file}'."
    )


def get_function_source(file_path: Union[str, Path], function_name: str) -> str:
    """
    Get the source code of a function from a specified Python file.

    .. code-block:: python

        source_code = get_function_source("path/to/your/file.py", "function_name")

    :param file_path: The path to the Python file containing the function.
    :param function_name: The name of the function to get the source code from.
    :raises FileNotFoundError: If the provided file path is not found.
    :raises ValueError: If the provided file is not a .py file.
    :raises AttributeError: If the specified function is not found in the file.
    :raises TypeError: If the specified name is not a function.
    :return: The source code of the specified function as a string.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found. Please provide a valid file path: {file_path}"
        )

    if not file_path.suffix == ".py":
        raise ValueError(f"Invalid file type. Please provide a .py file: {file_path}")

    sys.path.insert(0, str(file_path.parent))
    module_name = file_path.stem
    module = __import__(module_name)

    function = getattr(module, function_name, None)
    if function is None:
        raise AttributeError(
            f"Function '{function_name}' not found in {file_path}. Please provide a valid function name."
        )

    if not inspect.isfunction(function):
        raise TypeError(
            f"'{function_name}' is not a function. Please provide a valid function name."
        )

    return inspect.getsource(function)
