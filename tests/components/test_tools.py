"""Tests for the tools module.

This module contains tests for the tool decorator in llamabot.components.tools.
It verifies proper handling of function schemas, JSON schema generation,
and the functionality of built-in tools.
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

    # Check the structure of the json_schema
    schema = example_func.json_schema
    assert schema["name"] == "example_func"
    assert schema["description"] == "Example function with docstring."

    # Check parameters
    assert "parameters" in schema
    params = schema["parameters"]
    assert "properties" in params
    assert "a" in params["properties"]
    assert "b" in params["properties"]

    # Check types
    assert params["properties"]["a"]["type"] == "integer"
    assert params["properties"]["b"]["type"] == "string"

    # Check default value
    assert params["properties"]["b"]["default"] == "default"

    # Check required parameters
    assert "required" in params
    assert "a" in params["required"]
    assert "b" not in params["required"]  # b has a default value


def test_add_tool():
    """Test the add tool function."""
    # Test the function itself
    assert add(3, 4) == 7
    assert add(-1, 5) == 4

    # Test the schema
    assert hasattr(add, "json_schema")
    schema = add.json_schema
    assert schema["name"] == "add"
    assert "Add two integers" in schema["description"]


def test_today_date_tool():
    """Test the today_date tool function."""
    # Test the function returns a date string in expected format
    date_str = today_date()
    assert isinstance(date_str, str)

    # Verify date format YYYY-MM-DD
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        pytest.fail("today_date() did not return a properly formatted date string.")

    # Test the schema
    assert hasattr(today_date, "json_schema")
    schema = today_date.json_schema
    assert schema["name"] == "today_date"
    assert "Get the current date" in schema["description"]


def test_search_internet_tool_schema():
    """Test the search_internet_and_summarize tool schema."""
    # Test just the schema to avoid making actual internet requests during tests
    assert hasattr(search_internet_and_summarize, "json_schema")
    schema = search_internet_and_summarize.json_schema
    assert schema["name"] == "search_internet_and_summarize"
    assert "Search internet" in schema["description"]

    # Check parameters
    params = schema["parameters"]["properties"]
    assert "search_term" in params
    assert "max_results" in params
    assert "backend" in params

    # Check default values
    assert params["backend"]["default"] == "lite"


def test_write_and_execute_script_schema():
    """Test the write_and_execute_script tool schema."""
    assert hasattr(write_and_execute_script, "json_schema")
    schema = write_and_execute_script.json_schema
    assert schema["name"] == "write_and_execute_script"
    assert "Write and execute a Python script" in schema["description"]

    # Check parameters
    params = schema["parameters"]["properties"]
    assert "code" in params
    assert "dependencies_str" in params
    assert "python_version" in params
    assert "timeout" in params

    # Check default values
    assert params["python_version"]["default"] == ">=3.11"
    assert params["timeout"]["default"] == 30


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
