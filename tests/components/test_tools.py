"""Tests for the tools module.

This module contains tests for the Function class and tool decorator in llamabot.components.tools.
It verifies proper handling of function schemas, source code capture, and Pydantic model generation.
"""

from llamabot.components.tools import Function, tool


def test_function_source_code():
    """Test that source code is properly captured and accessible in Function tools.

    This test verifies that:
    1. Source code is captured when creating a Function schema from a callable
    2. Source code is included in the generated Pydantic model
    3. Source code is accessible from model instances
    """

    @tool
    def example_func(a: int, b: str = "default") -> str:
        """Example function."""
        return f"{a} {b}"

    # Check that source code is captured in the Function schema
    function_schema = Function.from_callable(example_func)
    assert function_schema.source_code is not None
    assert "def example_func" in function_schema.source_code

    # Check that source code is included in the Pydantic model
    model = function_schema.to_pydantic_model()
    assert "source_code" in model.model_fields

    # Create an instance and verify the source code is accessible
    instance = model(function_name="example_func", a=1)
    assert instance.source_code is not None
    assert "def example_func" in instance.source_code
