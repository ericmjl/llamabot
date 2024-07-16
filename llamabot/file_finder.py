"""
This module provides functions for file-handling.

Functions:
    - recursive_find(root_dir: Path, file_extension: str) -> List[Path]:
        Find all files in a given path with a given extension.
    - check_in_git_repo(path) -> bool:
        Check if a given path is in a git repository.
    - read_file(path: Path) -> str:
        Read a file.
"""

import subprocess
from pathlib import Path
from typing import List


def recursive_find(root_dir: Path, file_extension: str) -> List[Path]:
    """Find all files in a given path with a given extension.

    :param root_dir: Directory in which to search for files.
    :param file_extension: File extension to search for.
        As an example, this should be ".py", not "py".
    :return: List of Paths to requested documents.
    """
    return [path for path in root_dir.rglob(f"*{file_extension}") if path.is_file()]


def check_in_git_repo(path) -> bool:
    """Check if a given path is in a git repository.

    :param path: Path to check.
    :return: True if path is in a git repository, False otherwise.
    """
    git_check = subprocess.check_output(
        ["git", "rev-parse", "--is-inside-work-tree"], cwd=path
    )
    return git_check.decode("utf-8").strip() == "true"


def read_file(path: Path) -> str:
    """Read a file.

    :param path: Path to the file to be read.
    :return: Contents of the file.
    """
    with open(path, "r") as f:
        return f.read()
