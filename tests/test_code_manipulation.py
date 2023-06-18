"""Tests for the code_manipulation module."""
import tempfile
from pathlib import Path

import pytest

from llamabot.code_manipulation import (
    get_object_source_code,
    insert_docstring,
    replace_object_in_file,
)


def test_insert_docstring_function():
    """Test that a docstring is inserted into a function successfully.

    This test tests that a docstring is inserted into a function successfully.
    """
    source_code = "def foo():\n    pass"
    object_name = "foo"
    new_docstring = "This is a test docstring."

    with tempfile.NamedTemporaryFile("w+") as temp_file:
        temp_file.write(source_code)
        temp_file.flush()

        insert_docstring(temp_file.name, object_name, new_docstring)

        with open(temp_file.name, "r") as updated_file:
            updated_code = updated_file.read()

    expected_code = f'def {object_name}():\n    """{new_docstring}"""\n    pass\n'
    assert updated_code == expected_code


def test_insert_docstring_class():
    """Test that a docstring is inserted into a class successfully."""
    source_code = "class Foo:\n    pass"
    object_name = "Foo"
    new_docstring = "This is a test docstring."

    with tempfile.NamedTemporaryFile("w+") as temp_file:
        temp_file.write(source_code)
        temp_file.flush()

        insert_docstring(temp_file.name, object_name, new_docstring)

        with open(temp_file.name, "r") as updated_file:
            updated_code = updated_file.read()

    expected_code = f'class {object_name}:\n    """{new_docstring}"""\n    pass\n'
    assert updated_code == expected_code


@pytest.mark.xfail(
    reason="18 June 2023: Fails because of quotation marks. Need to fix."
)
def test_get_object_source_code_function():
    """Test that the source code of a function is returned successfully."""
    source_code = """
def test_function():
    return "Hello, World!"
    """
    with open("test_file.py", "w") as f:
        f.write(source_code)

    result = get_object_source_code("test_file.py", "test_function")
    assert result.strip() == source_code.strip()

    Path("test_file.py").unlink()


@pytest.mark.xfail(
    reason="18 June 2023: Fails because of quotation marks. Need to fix."
)
def test_get_object_source_code_class():
    """Test that the source code of a class is returned successfully."""
    source_code = """
class TestClass:
    def __init__(self):
        self.value = "Hello, World!"
    """
    with open("test_file.py", "w") as f:
        f.write(source_code)

    result = get_object_source_code("test_file.py", "TestClass")
    assert result.strip() == source_code.strip()

    Path("test_file.py").unlink()


def test_get_object_source_code_not_found():
    """Test that a NameError is raised when the object is not found."""
    source_code = """
def test_function():
    return "Hello, World!"
    """
    with open("test_file.py", "w") as f:
        f.write(source_code)

    with pytest.raises(NameError):
        get_object_source_code("test_file.py", "non_existent_function")

    Path("test_file.py").unlink()


def test_get_object_source_code_syntax_error():
    """Test that a SyntaxError is raised when the source code has invalid Python syntax."""
    source_code = """
def test_function()
    return "Hello, World!"
    """
    with open("test_file.py", "w") as f:
        f.write(source_code)

    with pytest.raises(SyntaxError):
        get_object_source_code("test_file.py", "test_function")

    Path("test_file.py").unlink()


def test_replace_function_in_file():
    """
    Test that replace_object_in_file correctly replaces a function in a source file.
    """
    source_code = """
def foo():
    return 42
"""
    new_function_definition = """
def foo():
    return 0
"""

    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        temp_file.write(source_code)
        temp_file.flush()

        replace_object_in_file(temp_file.name, "foo", new_function_definition)

        with open(temp_file.name, "r") as updated_file:
            updated_code = updated_file.read()

    assert updated_code.strip() == new_function_definition.strip()


def test_replace_class_in_file():
    """
    Test that replace_object_in_file correctly replaces a class in a source file.
    """
    source_code = """
class Foo:
    pass
"""
    new_class_definition = """
class Foo:

    def __init__(self):
        self.x = 42
"""

    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        temp_file.write(source_code)
        temp_file.flush()

        replace_object_in_file(temp_file.name, "Foo", new_class_definition)

        with open(temp_file.name, "r") as updated_file:
            updated_code = updated_file.read()

    assert updated_code.strip() == new_class_definition.strip()


def test_replace_object_in_file_syntax_error():
    """
    Test that replace_object_in_file raises a SyntaxError when the source file has invalid Python syntax.
    """
    source_code = """
def foo():
    return 42
invalid_syntax
"""

    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        temp_file.write(source_code)
        temp_file.flush()

        try:
            replace_object_in_file(temp_file.name, "foo", "def foo():\\n    return 0")
        except SyntaxError:
            pass
        else:
            assert False, "Expected a SyntaxError to be raised."


def test_replace_object_in_file_value_error():
    """
    Test that replace_object_in_file raises a ValueError when the specified object does not exist in the source file.
    """
    source_code = """
def foo():
    return 42
"""

    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        temp_file.write(source_code)
        temp_file.flush()

        try:
            replace_object_in_file(temp_file.name, "bar", "def bar():\\n    return 0")
        except ValueError:
            pass
        else:
            assert False, "Expected a ValueError to be raised."
