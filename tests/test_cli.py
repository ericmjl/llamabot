"""Tests for llamabot.cli."""

from llamabot.cli import app


def test_cli_executes():
    """Test that the CLI app initializes correctly."""
    # Test that the app is a Typer instance
    assert app is not None
    assert hasattr(app, "command")
    assert hasattr(app, "add_typer")

    # Test that subcommands are registered
    # The app should have registered commands/typers
    assert len(app.registered_groups) > 0 or len(app.registered_commands) > 0
