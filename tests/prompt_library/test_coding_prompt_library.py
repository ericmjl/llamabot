"""
This module contains tests for the get_function_source function
in the llamabot.code_manipulation module.
"""

import pytest

from llamabot.code_manipulation import get_function_source


def test_get_function_source_file_not_found():
    """Test that a FileNotFoundError is raised if the specified file is not found."""
    with pytest.raises(FileNotFoundError):
        get_function_source("non_existent_file.py", "some_function")


def test_get_function_source_invalid_file_type():
    """Test that a ValueError is raised if the specified file is not a .py file."""
    with pytest.raises(FileNotFoundError):
        get_function_source("test.txt", "some_function")


def test_get_function_source_function_not_found():
    """Test that an AttributeError is raised if the specified function is not found."""
    with pytest.raises(AttributeError):
        get_function_source(__file__, "non_existent_function")


def test_get_function_source_not_a_function():
    """Test that an AttributeError is raised if the specified name is not a function."""
    with pytest.raises(AttributeError):
        get_function_source(__file__, "Path")


def test_get_function_source_success():
    """Test that the source code of a function is returned successfully."""
    source_code = get_function_source(
        __file__, "test_get_function_source_not_a_function"
    )
    assert "def test_get_function_source_not_a_function" in source_code
