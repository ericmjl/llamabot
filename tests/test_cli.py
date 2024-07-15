"""Tests for llamabot.cli."""

import subprocess


def test_cli_executes():
    """Execution test for the CLI tool."""
    subprocess.run(["llamabot", "--help"])
