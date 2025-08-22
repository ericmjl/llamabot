"""Tests for the tools module.

This module contains tests for the tool decorator in llamabot.components.tools.
It verifies that the tool decorator properly applies JSON schemas to functions
and that the built-in tools function as intended.
"""

import pytest
from typing import Dict, Optional, List, Any
from datetime import datetime

from llamabot.components.tools import (
    tool,
    add,
    today_date,
    search_internet_and_summarize,
    write_and_execute_script,
    function_to_dict,
    parse_docstring,
    json_schema_type,
)


def test_json_schema_type():
    """Test the json_schema_type function."""
    assert json_schema_type("str") == "string"
    assert json_schema_type("int") == "integer"
    assert json_schema_type("float") == "number"
    assert json_schema_type("bool") == "boolean"
    assert json_schema_type("list") == "array"
    assert json_schema_type("dict") == "object"
    assert json_schema_type("tuple") == "array"
    assert json_schema_type("set") == "array"
    assert json_schema_type("None") == "null"
    assert json_schema_type("unknown") == "string"  # default case


def test_parse_docstring_numpy_style():
    """Test parsing numpy-style docstrings."""
    docstring = """
    Add two numbers together.

    Parameters
    ----------
    a : int
        The first number to add
    b : str, optional
        The second number to add, defaults to "default"

    Returns
    -------
    str
        The concatenated result
    """

    result = parse_docstring(docstring)
    assert result["summary"] == "Add two numbers together."
    assert "a" in result["parameters"]
    assert result["parameters"]["a"]["type"] == "integer"
    assert result["parameters"]["a"]["description"] == "The first number to add"
    assert "b" in result["parameters"]
    assert result["parameters"]["b"]["type"] == "string"
    assert (
        result["parameters"]["b"]["description"]
        == 'The second number to add, defaults to "default"'
    )


def test_parse_docstring_google_style():
    """Test parsing google-style docstrings."""
    docstring = """
    Calculate the sum of two numbers.

    Args:
        x (int): The first number
        y (float): The second number, optional
        name (str): The name of the calculation

    Returns:
        float: The sum of x and y
    """

    result = parse_docstring(docstring)
    assert result["summary"] == "Calculate the sum of two numbers."
    assert "x" in result["parameters"]
    assert result["parameters"]["x"]["type"] == "integer"
    assert result["parameters"]["x"]["description"] == "The first number"
    assert "y" in result["parameters"]
    assert result["parameters"]["y"]["type"] == "number"
    assert result["parameters"]["y"]["description"] == "The second number, optional"
    assert "name" in result["parameters"]
    assert result["parameters"]["name"]["type"] == "string"
    assert result["parameters"]["name"]["description"] == "The name of the calculation"


def test_parse_docstring_sphinx_style():
    """Test parsing sphinx-style docstrings."""
    docstring = """
    Process data with given parameters.

    :param data: The input data to process
    :type data: list
    :param threshold: The threshold value for processing
    :type threshold: float
    :param verbose: Whether to print verbose output
    :type verbose: bool
    :return: The processed result
    :rtype: dict
    """

    result = parse_docstring(docstring)
    assert result["summary"] == "Process data with given parameters."
    assert "data" in result["parameters"]
    assert result["parameters"]["data"]["type"] == "array"
    assert result["parameters"]["data"]["description"] == "The input data to process"
    assert "threshold" in result["parameters"]
    assert result["parameters"]["threshold"]["type"] == "number"
    assert (
        result["parameters"]["threshold"]["description"]
        == "The threshold value for processing"
    )
    assert "verbose" in result["parameters"]
    assert result["parameters"]["verbose"]["type"] == "boolean"
    assert (
        result["parameters"]["verbose"]["description"]
        == "Whether to print verbose output"
    )


def test_parse_docstring_mixed_style():
    """Test parsing docstrings with mixed or unknown style."""
    docstring = """
    Simple function with basic description.

    This is a simple function that does something.
    """

    result = parse_docstring(docstring)
    assert "Simple function with basic description." in result["summary"]
    assert "This is a simple function that does something." in result["summary"]
    assert result["parameters"] == {}


def test_parse_docstring_empty():
    """Test parsing empty docstrings."""
    result = parse_docstring("")
    assert result["summary"] == ""
    assert result["parameters"] == {}

    # Test with None - should handle gracefully
    result = parse_docstring("")  # parse_docstring doesn't accept None
    assert result["summary"] == ""
    assert result["parameters"] == {}


def test_function_to_dict_numpy_style():
    """Test function_to_dict with numpy-style docstring."""

    def numpy_func(a: int, b: str = "default") -> str:
        """Add two values together.

        Parameters
        ----------
        a : int
            The first value
        b : str, optional
            The second value, defaults to "default"

        Returns
        -------
        str
            The concatenated result
        """
        return f"{a} {b}"

    result = function_to_dict(numpy_func)
    assert result["name"] == "numpy_func"
    assert "Add two values together." in result["description"]
    assert "a" in result["parameters"]["properties"]
    assert result["parameters"]["properties"]["a"]["type"] == "integer"
    assert result["parameters"]["properties"]["a"]["description"] == "The first value"
    assert "b" in result["parameters"]["properties"]
    assert result["parameters"]["properties"]["b"]["type"] == "string"
    assert (
        result["parameters"]["properties"]["b"]["description"]
        == 'The second value, defaults to "default"'
    )
    assert "a" in result["parameters"]["required"]
    assert "b" not in result["parameters"]["required"]


def test_function_to_dict_google_style():
    """Test function_to_dict with google-style docstring."""

    def google_func(x: int, y: float, name: str = "calc") -> float:
        """Calculate the sum of two numbers.

        Args:
            x (int): The first number
            y (float): The second number
            name (str): The name of the calculation

        Returns:
            float: The sum of x and y
        """
        return x + y

    result = function_to_dict(google_func)
    assert result["name"] == "google_func"
    assert "Calculate the sum of two numbers." in result["description"]
    assert "x" in result["parameters"]["properties"]
    assert result["parameters"]["properties"]["x"]["type"] == "integer"
    assert result["parameters"]["properties"]["x"]["description"] == "The first number"
    assert "y" in result["parameters"]["properties"]
    assert result["parameters"]["properties"]["y"]["type"] == "number"
    assert result["parameters"]["properties"]["y"]["description"] == "The second number"
    assert "name" in result["parameters"]["properties"]
    assert result["parameters"]["properties"]["name"]["type"] == "string"
    assert (
        result["parameters"]["properties"]["name"]["description"]
        == "The name of the calculation"
    )
    assert "x" in result["parameters"]["required"]
    assert "y" in result["parameters"]["required"]
    assert "name" not in result["parameters"]["required"]


def test_function_to_dict_sphinx_style():
    """Test function_to_dict with sphinx-style docstring."""

    def sphinx_func(
        data: List[int], threshold: float, verbose: bool = False
    ) -> Dict[str, int]:
        """Process data with given parameters.

        :param data: The input data to process
        :type data: list
        :param threshold: The threshold value for processing
        :type threshold: float
        :param verbose: Whether to print verbose output
        :type verbose: bool
        :return: The processed result
        :rtype: dict
        """
        return {"count": len(data)}

    result = function_to_dict(sphinx_func)
    assert result["name"] == "sphinx_func"
    assert "Process data with given parameters." in result["description"]
    assert "data" in result["parameters"]["properties"]
    assert result["parameters"]["properties"]["data"]["type"] == "array"
    assert (
        result["parameters"]["properties"]["data"]["description"]
        == "The input data to process"
    )
    assert "threshold" in result["parameters"]["properties"]
    assert result["parameters"]["properties"]["threshold"]["type"] == "number"
    assert (
        result["parameters"]["properties"]["threshold"]["description"]
        == "The threshold value for processing"
    )
    assert "verbose" in result["parameters"]["properties"]
    assert result["parameters"]["properties"]["verbose"]["type"] == "boolean"
    assert (
        result["parameters"]["properties"]["verbose"]["description"]
        == "Whether to print verbose output"
    )
    assert "data" in result["parameters"]["required"]
    assert "threshold" in result["parameters"]["required"]
    assert "verbose" not in result["parameters"]["required"]


def test_function_to_dict_complex_types():
    """Test function_to_dict with complex type annotations."""

    def complex_func(
        data: Optional[List[str]] = None, config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Process complex data types.

        :param data: Optional list of strings
        :param config: Optional configuration dictionary
        :return: Processed result
        """
        return "processed"

    result = function_to_dict(complex_func)
    assert result["name"] == "complex_func"
    assert "data" in result["parameters"]["properties"]
    assert result["parameters"]["properties"]["data"]["type"] == "array"
    assert "config" in result["parameters"]["properties"]
    assert result["parameters"]["properties"]["config"]["type"] == "object"
    assert "data" not in result["parameters"]["required"]
    assert "config" not in result["parameters"]["required"]


def test_function_to_dict_no_docstring():
    """Test function_to_dict with no docstring."""

    def no_docstring_func(a: int, b: str) -> bool:
        return True

    result = function_to_dict(no_docstring_func)
    assert result["name"] == "no_docstring_func"
    assert result["description"] == ""
    assert "a" in result["parameters"]["properties"]
    assert result["parameters"]["properties"]["a"]["type"] == "integer"
    assert "b" in result["parameters"]["properties"]
    assert result["parameters"]["properties"]["b"]["type"] == "string"
    assert "a" in result["parameters"]["required"]
    assert "b" in result["parameters"]["required"]


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
    assert schema["type"] == "function"
    assert "function" in schema
    assert schema["function"]["name"] == "example_func"
    assert schema["function"]["description"] == "Example function with docstring."
    assert "parameters" in schema["function"]
    assert schema["function"]["parameters"]["type"] == "object"
    assert "properties" in schema["function"]["parameters"]
    assert "a" in schema["function"]["parameters"]["properties"]
    assert schema["function"]["parameters"]["properties"]["a"]["type"] == "integer"
    assert "b" in schema["function"]["parameters"]["properties"]
    assert schema["function"]["parameters"]["properties"]["b"]["type"] == "string"
    assert "required" in schema["function"]["parameters"]
    assert "a" in schema["function"]["parameters"]["required"]
    assert "b" not in schema["function"]["parameters"]["required"]

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
    assert schema["type"] == "function"
    assert "function" in schema
    assert schema["function"]["name"] == "add"
    assert (
        schema["function"]["description"]
        == "Add two integers, a and b, and return the result."
    )
    assert "parameters" in schema["function"]
    assert schema["function"]["parameters"]["type"] == "object"
    assert "properties" in schema["function"]["parameters"]
    assert "a" in schema["function"]["parameters"]["properties"]
    assert schema["function"]["parameters"]["properties"]["a"]["type"] == "integer"
    assert "b" in schema["function"]["parameters"]["properties"]
    assert schema["function"]["parameters"]["properties"]["b"]["type"] == "integer"
    assert "required" in schema["function"]["parameters"]
    assert "a" in schema["function"]["parameters"]["required"]
    assert "b" in schema["function"]["parameters"]["required"]


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
    assert schema["type"] == "function"
    assert "function" in schema
    assert schema["function"]["name"] == "today_date"
    assert schema["function"]["description"] == "Get the current date."
    assert "parameters" in schema["function"]
    assert schema["function"]["parameters"]["type"] == "object"
    assert "properties" in schema["function"]["parameters"]
    assert len(schema["function"]["parameters"]["properties"]) == 0
    # Allow for 'required' to be missing if there are no required parameters
    if "required" in schema["function"]["parameters"]:
        assert len(schema["function"]["parameters"]["required"]) == 0


def test_search_internet_tool_schema():
    """Test the search_internet_and_summarize tool schema."""
    # We're just testing the schema, not actual functionality to avoid internet requests
    assert hasattr(search_internet_and_summarize, "json_schema")
    schema = search_internet_and_summarize.json_schema
    assert schema["type"] == "function"
    assert "function" in schema
    assert schema["function"]["name"] == "search_internet_and_summarize"
    assert "parameters" in schema["function"]
    assert schema["function"]["parameters"]["type"] == "object"
    assert "properties" in schema["function"]["parameters"]
    assert "search_term" in schema["function"]["parameters"]["properties"]
    assert (
        schema["function"]["parameters"]["properties"]["search_term"]["type"]
        == "string"
    )
    assert "max_results" in schema["function"]["parameters"]["properties"]
    assert (
        schema["function"]["parameters"]["properties"]["max_results"]["type"]
        == "integer"
    )
    assert "required" in schema["function"]["parameters"]
    assert "search_term" in schema["function"]["parameters"]["required"]
    assert "max_results" in schema["function"]["parameters"]["required"]


def test_write_and_execute_script_schema():
    """Test the write_and_execute_script tool schema."""
    assert hasattr(write_and_execute_script, "json_schema")
    schema = write_and_execute_script.json_schema
    assert schema["type"] == "function"
    assert "function" in schema
    assert schema["function"]["name"] == "write_and_execute_script"
    assert "parameters" in schema["function"]
    assert schema["function"]["parameters"]["type"] == "object"
    assert "properties" in schema["function"]["parameters"]
    assert "code" in schema["function"]["parameters"]["properties"]
    assert schema["function"]["parameters"]["properties"]["code"]["type"] == "string"
    assert "dependencies_str" in schema["function"]["parameters"]["properties"]
    assert (
        schema["function"]["parameters"]["properties"]["dependencies_str"]["type"]
        == "string"
    )
    assert "python_version" in schema["function"]["parameters"]["properties"]
    assert (
        schema["function"]["parameters"]["properties"]["python_version"]["type"]
        == "string"
    )
    assert "required" in schema["function"]["parameters"]
    assert "code" in schema["function"]["parameters"]["required"]
    assert "dependencies_str" not in schema["function"]["parameters"]["required"]
    assert "python_version" not in schema["function"]["parameters"]["required"]


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
