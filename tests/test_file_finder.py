"""
This module contains tests for the llamabot.file_finder module.

It tests the following functions:
- recursive_find
- check_in_git_repo
- read_file

It uses the following external libraries:
- pytest
- hypothesis
"""
import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from hypothesis import given
from hypothesis import strategies as st

from llamabot.file_finder import check_in_git_repo, read_file, recursive_find


@given(st.text(min_size=1, max_size=5, alphabet="abcdefghijklmnopqrstuvwxyz"))
def test_recursive_find_empty_directory(extension):
    """Test that recursive_find returns an empty list when given an empty directory.

    :param extension: The extension to search for.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        assert recursive_find(root, extension) == []


@given(
    st.text(min_size=1, max_size=5, alphabet="abcdefghijklmnopqrstuvwxyz"),
    st.text(min_size=1, max_size=5, alphabet="abcdefghijklmnopqrstuvwxyz"),
)
def test_recursive_find_single_file(extension, file_name):
    """Test that recursive_find returns a single file when given a directory with a single file.

    :param extension: The extension to search for.
    :param file_name: The name of the file to create.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        file_path = root / f"{file_name}.{extension}"
        file_path.touch()
        assert recursive_find(root, f".{extension}") == [file_path]


@given(
    st.text(min_size=1, max_size=5, alphabet="abcdefghijklmnopqrstuvwxyz"),
    st.text(min_size=1, max_size=5, alphabet="abcdefghijklmnopqrstuvwxyz"),
)
def test_recursive_find_nested_file(extension, file_name):
    """Test that recursive_find returns a single file when given a directory with a single file.

    :param extension: The extension to search for.
    :param file_name: The name of the file to create.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        nested_dir = root / "nested"
        nested_dir.mkdir()
        file_path = nested_dir / f"{file_name}.{extension}"
        file_path.touch()
        assert recursive_find(root, f".{extension}") == [file_path]


@given(
    st.text(min_size=1, max_size=5, alphabet="abcdefghijklmnopqrstuvwxyz"),
    st.text(min_size=1, max_size=5, alphabet="abcdefghijklmnopqrstuvwxyz"),
)
def test_recursive_find_multiple_files(extension, file_name):
    """Test that recursive_find returns multiple files when given a directory with multiple files.

    :param extension: The extension to search for.
    :param file_name: The name of the file to create.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        file_path1 = root / f"{file_name}.{extension}"
        file_path1.touch()
        nested_dir = root / "nested"
        nested_dir.mkdir()
        file_path2 = nested_dir / f"{file_name}.{extension}"
        file_path2.touch()
        assert set(recursive_find(root, f".{extension}")) == {file_path1, file_path2}


def test_check_in_git_repo_true():
    """
    Test check_in_git_repo returns True when git command returns 'true'.
    """
    subprocess.check_output = MagicMock(return_value=b"true\n")
    assert check_in_git_repo("some_path") is True


def test_check_in_git_repo_false():
    """
    Test check_in_git_repo returns False when git command returns 'false'.
    """
    subprocess.check_output = MagicMock(return_value=b"false\n")
    assert check_in_git_repo("some_path") is False


def test_check_in_git_repo_error():
    """
    Test check_in_git_repo raises an exception when git command raises an exception.
    """
    subprocess.check_output = MagicMock(
        side_effect=subprocess.CalledProcessError(1, "git")
    )
    with pytest.raises(subprocess.CalledProcessError):
        check_in_git_repo("some_path")


@given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz"))
def test_read_file_contents(text: str):
    """
    Test if read_file returns the correct contents of a file.

    :param text: The text to write to the file.
    """
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(text.encode())
        temp_file_path = Path(temp_file.name)

    assert read_file(temp_file_path) == text
    os.remove(temp_file_path)


def test_read_file_nonexistent_path():
    """
    Test if read_file raises a FileNotFoundError for a nonexistent file path.
    """
    non_existent_path = Path("non_existent_file.txt")

    with pytest.raises(FileNotFoundError):
        read_file(non_existent_path)
