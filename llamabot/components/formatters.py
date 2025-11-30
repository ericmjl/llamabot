"""Formatters for converting objects to displayable formats.

This module provides formatters for different display backends (Marimo, Jupyter, etc.)
that can be used with `return_object_to_user` tool.
"""

from typing import Any, Dict


def create_marimo_formatter(mo) -> callable:
    """Create a marimo formatter for return_object_to_user.

    This formatter converts result dicts/lists into mo.vstack() compatible format:
    - Strings become mo.md(...)
    - Objects (DataFrames, figures, dicts) pass through unchanged
    - Returns mo.vstack([...]) of all items

    The formatter expects the result to be a dict (key-value store) or list.
    For dicts, it uses the values directly (no labels).

    Usage in notebook:
        import marimo as mo
        from llamabot.components.formatters import create_marimo_formatter

        _globals = globals()
        _globals["_return_object_formatter"] = create_marimo_formatter(mo)

        agent = AgentBot(...)
        result = agent("query", globals_dict=_globals)

    :param mo: The marimo module
    :return: Formatter function compatible with return_object_to_user
    """

    def formatter(result: Any, globals_dict: Dict) -> Any:
        """Format result for marimo display.

        :param result: The result object (dict, list, or single value)
        :param globals_dict: Globals dictionary (unused but required by interface)
        :return: Formatted result suitable for marimo display
        """
        # Convert dict to list (just use values, no labels)
        if isinstance(result, dict):
            items = list(result.values())
        elif isinstance(result, list):
            items = result
        else:
            # Single item
            items = [result]

        # Convert strings to mo.md(), pass through everything else
        display_items = [
            mo.md(item) if isinstance(item, str) else item for item in items
        ]

        # Return vstack if multiple items, single item otherwise
        if len(display_items) == 1:
            return display_items[0]
        return mo.vstack(display_items)

    return formatter
