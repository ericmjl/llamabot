"""Tests for the tools module.

This module contains tests for the tool decorator in llamabot.components.tools.
It verifies that the tool decorator properly applies JSON schemas to functions
and that the built-in tools function as intended.
"""

import pytest
from typing import Dict
from datetime import datetime

from llamabot.components.tools import (
    tool,
    add,
    today_date,
    search_internet_and_summarize,
    write_and_execute_script,
)


def test_tool_decorator():
    """Test that the tool decorator properly attaches JSON schema to functions."""

    @tool
    def example_func(a: int, b: str = "default") -> str:
        """Example function with docstring."""
        return f"{a} {b}"

    # Check that json_schema is attached to the function
    assert hasattr(example_func, "json_schema")

    # Check the basic structure of the json_schema
    schema = example_func.json_schema
    assert schema["name"] == "example_func"
    assert "Example function with docstring." in schema["description"]

    # Check parameters
    assert "parameters" in schema
    params = schema["parameters"]["properties"]
    assert "a" in params
    assert "b" in params

    # Check parameter types
    assert params["a"]["type"] == "integer"
    assert params["b"]["type"] == "string"

    # Check required parameters
    assert "required" in schema["parameters"]
    assert "a" in schema["parameters"]["required"]
    # Parameter 'b' has a default value so it shouldn't be required
    assert "b" not in schema["parameters"]["required"]

    # Call the function to make sure it still works as expected
    assert example_func(1) == "1 default"
    assert example_func(2, "test") == "2 test"


def test_add_tool():
    """Test the add tool function."""
    # Test functional behavior
    assert add(3, 4) == 7
    assert add(-1, 5) == 4
    assert add(0, 0) == 0

    # Test schema basics
    assert hasattr(add, "json_schema")
    schema = add.json_schema
    assert schema["name"] == "add"
    assert "Add two integers" in schema["description"]

    # Test parameter structure
    params = schema["parameters"]["properties"]
    assert "a" in params
    assert "b" in params
    assert params["a"]["type"] == "integer"
    assert params["b"]["type"] == "integer"


def test_today_date_tool():
    """Test the today_date tool function."""
    # Test functional behavior - returns a date string in expected format
    date_str = today_date()
    assert isinstance(date_str, str)

    # Verify date format YYYY-MM-DD
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        pytest.fail("today_date() did not return a properly formatted date string")

    # Test schema basics
    assert hasattr(today_date, "json_schema")
    schema = today_date.json_schema
    assert schema["name"] == "today_date"
    assert "Get the current date" in schema["description"]


def test_search_internet_tool_schema():
    """Test the search_internet_and_summarize tool schema."""
    # We're just testing the schema, not actual functionality to avoid internet requests
    assert hasattr(search_internet_and_summarize, "json_schema")
    schema = search_internet_and_summarize.json_schema
    assert schema["name"] == "search_internet_and_summarize"
    assert "Search internet" in schema["description"]

    # Test parameter structure
    params = schema["parameters"]["properties"]
    assert "search_term" in params
    assert "max_results" in params
    assert "backend" in params

    # Check types
    assert params["search_term"]["type"] == "string"
    assert params["max_results"]["type"] == "integer"
    assert params["backend"]["type"] == "string"


def test_write_and_execute_script_schema():
    """Test the write_and_execute_script tool schema."""
    assert hasattr(write_and_execute_script, "json_schema")
    schema = write_and_execute_script.json_schema
    assert schema["name"] == "write_and_execute_script"
    assert "Write and execute a Python script" in schema["description"]

    # Test parameter structure
    params = schema["parameters"]["properties"]
    assert "code" in params
    assert "dependencies_str" in params
    assert "python_version" in params

    # Check types
    assert params["code"]["type"] == "string"


@pytest.mark.xfail(reason="Sometimes the Docker engine may not be running.")
def test_write_and_execute_script_basic():
    """Test basic functionality of write_and_execute_script tool."""
    code = """
print("Hello, world!")
result = 5 + 7
print(f"Result: {result}")
"""
    result = write_and_execute_script(code)

    assert isinstance(result, Dict)
    assert "stdout" in result
    assert "Hello, world!" in result["stdout"]
    assert "Result: 12" in result["stdout"]
    assert result["status"] == 0  # Success
