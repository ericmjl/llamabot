"""Tests for the code_manipulation module."""

import os
import tempfile
from pathlib import Path

import pytest

from llamabot.code_manipulation import (
    get_dependencies,
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


def test_get_dependencies():
    """Test the get_dependencies function.

    This test tests that the get_dependencies function returns a list of dependencies.
    """
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a temporary source file
        source_code = """
import module1
from module2 import function2
from module3 import Class3

def function1():
    pass
        """
        source_file = os.path.join(temp_dir, "source_file.py")
        with open(source_file, "w") as file:
            file.write(source_code)

        # Create temporary dependency files
        module1_code = """
def function_module1():
    pass
        """
        module1_file = os.path.join(temp_dir, "module1.py")
        with open(module1_file, "w") as file:
            file.write(module1_code)

        module2_code = """
def function2():
    pass
        """
        module2_file = os.path.join(temp_dir, "module2.py")
        with open(module2_file, "w") as file:
            file.write(module2_code)

        module3_code = """
class Class3:
    pass
        """
        module3_file = os.path.join(temp_dir, "module3.py")
        with open(module3_file, "w") as file:
            file.write(module3_code)

        # Test get_dependencies function
        object_name = "function1"
        dependencies = get_dependencies(source_file, object_name)

        assert isinstance(dependencies, list)
        for dependency in dependencies:
            assert isinstance(dependency, str)
            assert os.path.isfile(dependency)
            assert os.path.splitext(dependency)[1] == ".py"

        # Additional assertions based on the specific dependencies
        assert module1_file in dependencies
        assert module2_file in dependencies
        assert module3_file in dependencies

        # Clean up temporary files and directory (optional)
        # You can comment out the following lines if you want to inspect the temporary files manually
        os.remove(source_file)
        os.remove(module1_file)
        os.remove(module2_file)
        os.remove(module3_file)
