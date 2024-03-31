"""Tests for llamabot.cli."""

import subprocess
import time


def test_cli_tool_execution_time():
    """Test the CLI tool execution time."""
    start_time = time.time()

    # Replace 'your_cli_command' with your actual CLI command
    subprocess.run(["llamabot", "--help"])

    end_time = time.time()
    execution_time = end_time - start_time

    # Replace 'threshold_in_seconds' with your time threshold
    assert execution_time < 3.0
