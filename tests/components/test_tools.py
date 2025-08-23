"""Tests for the tools module.

This module contains tests for the tool decorator in llamabot.components.tools.
It verifies that the tool decorator properly applies JSON schemas to functions
and that the built-in tools function as intended.
"""

import pytest
from typing import Dict, List, Any
from datetime import datetime

from llamabot.components.tools import (
    tool,
    add,
    today_date,
    search_internet_and_summarize,
    write_and_execute_script,
    function_to_dict,
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

    def numpy_func(a: int, b: str) -> str:
        """Add two numbers together.

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
        return f"{a} {b}"

    result = function_to_dict(numpy_func)
    assert result["description"] == "Add two numbers together."
    assert "a" in result["parameters"]["properties"]
    assert result["parameters"]["properties"]["a"]["type"] == "integer"
    assert (
        result["parameters"]["properties"]["a"]["description"]
        == "The first number to add"
    )
    assert "b" in result["parameters"]["properties"]
    assert result["parameters"]["properties"]["b"]["type"] == "string"
    assert (
        result["parameters"]["properties"]["b"]["description"]
        == 'The second number to add, defaults to "default"'
    )


def test_parse_docstring_google_style():
    """Test parsing google-style docstrings."""

    def google_func(x: int, y: float, name: str) -> float:
        """Calculate the sum of two numbers.

        Args:
            x (int): The first number
            y (float): The second number, optional
            name (str): The name of the calculation

        Returns:
            float: The sum of x and y
        """
        return x + y

    result = function_to_dict(google_func)
    assert result["description"] == "Calculate the sum of two numbers."
    assert "x" in result["parameters"]["properties"]
    assert result["parameters"]["properties"]["x"]["type"] == "integer"
    assert result["parameters"]["properties"]["x"]["description"] == "The first number"
    assert "y" in result["parameters"]["properties"]
    assert result["parameters"]["properties"]["y"]["type"] == "number"
    assert (
        result["parameters"]["properties"]["y"]["description"]
        == "The second number, optional"
    )
    assert "name" in result["parameters"]["properties"]
    assert result["parameters"]["properties"]["name"]["type"] == "string"
    assert (
        result["parameters"]["properties"]["name"]["description"]
        == "The name of the calculation"
    )


def test_parse_docstring_sphinx_style():
    """Test parsing sphinx-style docstrings."""

    def sphinx_func(data: list, threshold: float, verbose: bool) -> dict:
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
        return {"processed": True}

    result = function_to_dict(sphinx_func)
    assert result["description"] == "Process data with given parameters."
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


def test_parse_docstring_mixed_style():
    """Test parsing docstrings with mixed or unknown style."""

    def mixed_func() -> str:
        """Simple function with basic description.

        This is a simple function that does something.
        """
        return "hello"

    result = function_to_dict(mixed_func)
    assert "Simple function with basic description." in result["description"]
    # The long description is not included in the short description by docstring-parser
    # This is expected behavior - docstring-parser separates short and long descriptions
    assert result["parameters"]["properties"] == {}


def test_parse_docstring_empty():
    """Test parsing empty docstrings."""

    def empty_func() -> str:
        """"""
        return "hello"

    result = function_to_dict(empty_func)
    assert result["description"] == ""
    assert result["parameters"]["properties"] == {}


def test_function_to_dict_numpy_style():
    """Test function_to_dict with numpy-style docstring."""

    def numpy_func(a: int, b: str) -> str:
        """Add two values together.

        Parameters
        ----------
        a : int
            The first value
        b : str
            The second value

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
    assert result["parameters"]["properties"]["b"]["description"] == "The second value"
    assert "a" in result["parameters"]["required"]
    assert "b" in result["parameters"]["required"]


def test_function_to_dict_google_style():
    """Test function_to_dict with google-style docstring."""

    def google_func(x: int, y: float, name: str) -> float:
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
    assert "name" in result["parameters"]["required"]


def test_function_to_dict_sphinx_style():
    """Test function_to_dict with sphinx-style docstring."""

    def sphinx_func(data: List[int], threshold: float, verbose: bool) -> Dict[str, int]:
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
    assert "verbose" in result["parameters"]["required"]


def test_function_to_dict_complex_types():
    """Test function_to_dict with complex type annotations."""

    def complex_func(data: List[str], config: Dict[str, Any]) -> str:
        """Process complex data types.

        :param data: List of strings
        :param config: Configuration dictionary
        :return: Processed result
        """
        return "processed"

    result = function_to_dict(complex_func)
    assert result["name"] == "complex_func"
    assert "data" in result["parameters"]["properties"]
    assert result["parameters"]["properties"]["data"]["type"] == "array"
    assert "config" in result["parameters"]["properties"]
    assert result["parameters"]["properties"]["config"]["type"] == "object"
    assert "data" in result["parameters"]["required"]
    assert "config" in result["parameters"]["required"]


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


def test_parse_docstring_malformed():
    """Test parsing malformed docstrings."""

    # Test docstring with broken numpy format
    def broken_numpy_func() -> str:
        """Broken
        Parameters
        --
        a : int missing description
        """
        return "hello"

    result = function_to_dict(broken_numpy_func)
    # Should handle gracefully and extract what it can
    assert "Broken" in result["description"]
    # The malformed parameter section should be ignored
    assert result["parameters"]["properties"] == {}

    # Test docstring with broken google format
    def broken_google_func() -> str:
        """Broken
        Args:
            x (int: missing closing paren
        """
        return "hello"

    result = function_to_dict(broken_google_func)
    assert "Broken" in result["description"]
    assert result["parameters"]["properties"] == {}

    # Test docstring with broken sphinx format
    def broken_sphinx_func() -> str:
        """Broken
        :param x: description
        :type x: broken type
        """
        return "hello"

    result = function_to_dict(broken_sphinx_func)
    assert "Broken" in result["description"]
    # The docstring-parser might not extract malformed sphinx parameters
    # This is expected behavior - malformed docstrings may not parse correctly


def test_parse_docstring_long_descriptions():
    """Test parsing docstrings with long parameter descriptions."""

    # Test numpy style with multi-paragraph descriptions
    def long_desc_func(data: list, threshold: float) -> dict:
        """Process complex data with extensive documentation.

        Parameters
        ----------
        data : list
            The input data to process. This parameter accepts a list of items
            that will be processed according to the specified algorithm.

            The data should be pre-sorted and validated before passing to this
            function. Invalid data will be filtered out automatically.

            Examples of valid data:
            - [1, 2, 3, 4, 5]
            - ['a', 'b', 'c']
            - [{'id': 1, 'value': 100}]

        threshold : float
            The threshold value for processing. This is a critical parameter
            that determines the sensitivity of the algorithm. Values between
            0.0 and 1.0 are recommended for optimal performance.

            Lower values (0.0-0.3) will result in more aggressive filtering.
            Higher values (0.7-1.0) will be more permissive.

        Returns
        -------
        dict
            The processed result containing statistics and filtered data.
        """
        return {"processed": True}

    result = function_to_dict(long_desc_func)
    assert "Process complex data with extensive documentation." in result["description"]
    assert "data" in result["parameters"]["properties"]
    assert result["parameters"]["properties"]["data"]["type"] == "array"
    # Should capture the full multi-paragraph description
    description = result["parameters"]["properties"]["data"]["description"]
    assert "The input data to process" in description
    assert "Examples of valid data" in description
    assert "pre-sorted and validated" in description

    assert "threshold" in result["parameters"]["properties"]
    assert result["parameters"]["properties"]["threshold"]["type"] == "number"
    threshold_desc = result["parameters"]["properties"]["threshold"]["description"]
    assert "The threshold value for processing" in threshold_desc
    assert "Lower values (0.0-0.3)" in threshold_desc
    assert "Higher values (0.7-1.0)" in threshold_desc


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

    # Test that default values are included in the schema
    assert "default" in schema["function"]["parameters"]["properties"]["b"]
    assert schema["function"]["parameters"]["properties"]["b"]["default"] == "default"


def test_function_to_dict_with_default_values():
    """Test that function_to_dict properly includes default values in the schema."""

    def test_func(a: int, b: str = "hello", c: float = 3.14, d: bool = True) -> str:
        """Test function with various default values."""
        return f"{a} {b} {c} {d}"

    result = function_to_dict(test_func)

    # Check that required parameter doesn't have default
    assert "a" in result["parameters"]["properties"]
    assert "default" not in result["parameters"]["properties"]["a"]
    assert "a" in result["parameters"]["required"]

    # Check that optional parameters have defaults
    assert "b" in result["parameters"]["properties"]
    assert "default" in result["parameters"]["properties"]["b"]
    assert result["parameters"]["properties"]["b"]["default"] == "hello"
    assert "b" not in result["parameters"]["required"]

    assert "c" in result["parameters"]["properties"]
    assert "default" in result["parameters"]["properties"]["c"]
    assert result["parameters"]["properties"]["c"]["default"] == 3.14
    assert "c" not in result["parameters"]["required"]

    assert "d" in result["parameters"]["properties"]
    assert "default" in result["parameters"]["properties"]["d"]
    assert result["parameters"]["properties"]["d"]["default"] is True
    assert "d" not in result["parameters"]["required"]


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


def test_function_description_combines_short_and_long():
    """Test that function descriptions combine short and long descriptions."""

    def test_func(a: int) -> str:
        """Short description.

        This is the long description that should be included
        along with the short description.
        """
        return str(a)

    result = function_to_dict(test_func)

    # Should include both short and long descriptions
    description = result["description"]
    assert "Short description." in description
    assert "This is the long description" in description
    assert "should be included" in description

    # Should have proper formatting with double newlines
    assert "\n\n" in description
