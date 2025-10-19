"""Utilities for llamabot."""

import inspect
import sys
from pathlib import Path
from typing import Optional, Union


from pyprojroot import here


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
            f"Function '{function_name}' not found in {file_path}. "
            "Please provide a valid function name."
        )

    if not inspect.isfunction(function):
        raise TypeError(
            f"'{function_name}' is not a function. "
            "Please provide a valid function name."
        )

    return inspect.getsource(function)


def get_git_diff(repo_path: Optional[Union[str, Path]] = None) -> str:
    """Get the git diff of a repository.

    :param repo_path: The path to the git repository.
    :raises ValueError: If the provided path is not a git repository.
    :raises ValueError: If the provided git repository
        has no staged or unstaged changes.
    :return: The git diff as a string, or None if there are no staged changes.
    """
    try:
        from git import GitCommandError, Repo
    except ImportError:
        raise ImportError(
            "git is not installed. Please install it with `pip install llamabot[cli]`."
        )

    if repo_path is None:
        repo_path = here()
    try:
        repo = Repo(repo_path)
    except GitCommandError as e:
        raise ValueError("Please provide a valid path to a git repository.") from e

    if repo.is_dirty():
        if repo.index.diff("HEAD"):
            try:
                diff = repo.git.diff("--cached")
            except GitCommandError as e:
                raise ValueError(
                    "Please ensure that the git repository has staged changes."
                ) from e
        else:
            return ""
    else:
        try:
            diff = repo.git.diff()
        except GitCommandError as e:
            raise ValueError(
                "Please ensure that the git repository has unstaged changes."
            ) from e

    return diff
