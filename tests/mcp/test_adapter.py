"""Tests for MCP schema adaptation helpers."""

from llamabot.mcp.adapter import (
    normalize_openai_parameters,
    prefixed_tool_name,
    remap_schema_properties_for_python,
    sanitize_python_identifier,
)


def test_sanitize_python_identifier_replaces_invalid_chars() -> None:
    """Hyphens and dots become underscores."""
    assert sanitize_python_identifier("my-tool") == "my_tool"
    assert sanitize_python_identifier("1st") == "_1st"


def test_prefixed_tool_name_uses_separator() -> None:
    """Prefixed names combine server and tool."""
    assert prefixed_tool_name("srv", "add", "__") == "srv__add"


def test_normalize_openai_parameters_empty() -> None:
    """Missing schema yields empty object parameters."""
    assert normalize_openai_parameters(None) == {
        "type": "object",
        "properties": {},
        "required": [],
    }


def test_remap_schema_properties_keeps_wire_mapping() -> None:
    """Wire JSON keys map to Python-safe parameter names."""
    props = {"my-param": {"type": "string"}, "ok": {"type": "integer"}}
    new_props, wire_map, req = remap_schema_properties_for_python(
        props,
        ["my-param"],
    )
    assert "my_param" in new_props
    assert wire_map["my_param"] == "my-param"
    assert "my_param" in req
