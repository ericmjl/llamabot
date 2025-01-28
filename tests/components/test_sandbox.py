"""Tests for the sandbox module."""

from datetime import datetime
from pathlib import Path

import pytest

from llamabot.components.sandbox import ScriptExecutor, ScriptMetadata


@pytest.fixture
def script_metadata():
    """Create a test script metadata instance."""
    return ScriptMetadata(
        requires_python=">=3.11",
        dependencies=["requests<3"],
        auth="test-auth",
        purpose="test script",
        timestamp=datetime.now(),
    )


@pytest.fixture
def test_script():
    """Create a test script that prints JSON output."""
    return """
import json
import sys

result = {"message": "Hello from sandbox!"}
print(json.dumps(result))
sys.exit(0)
"""


def test_script_executor_initialization():
    """Test that ScriptExecutor initializes correctly."""
    executor = ScriptExecutor()
    assert executor.scripts_dir.exists()
    assert executor.results_dir.exists()


def test_write_script(script_metadata, test_script):
    """Test writing a script with metadata."""
    executor = ScriptExecutor()
    script_path = executor.write_script(test_script, script_metadata)

    assert script_path.exists()
    content = script_path.read_text()
    assert "# /// script" in content
    assert script_metadata.requires_python in content
    assert script_metadata.dependencies[0] in content
    assert script_metadata.auth in content
    assert test_script in content


def test_run_script(script_metadata, test_script):
    """Test running a script in the sandbox."""
    executor = ScriptExecutor()
    script_path = executor.write_script(test_script, script_metadata)

    result = executor.run_script(script_path)
    assert isinstance(result, dict)
    assert result["status"] == 0  # Check successful execution

    # Parse JSON from stdout
    import json

    output = json.loads(result["stdout"])
    assert output["message"] == "Hello from sandbox!"


def test_script_execution_timeout(script_metadata):
    """Test that script execution respects timeout."""
    infinite_loop = """
import time
while True:
    time.sleep(1)
"""

    executor = ScriptExecutor()
    script_path = executor.write_script(infinite_loop, script_metadata)

    with pytest.raises(Exception):
        executor.run_script(script_path, timeout=1)


def test_script_execution_error(script_metadata):
    """Test handling of script execution errors."""
    error_script = """
raise ValueError("Test error")
"""

    executor = ScriptExecutor()
    script_path = executor.write_script(error_script, script_metadata)

    with pytest.raises(RuntimeError) as exc_info:
        executor.run_script(script_path)
    assert "Test error" in str(exc_info.value)


def test_custom_temp_dir():
    """Test using a custom temporary directory."""
    temp_dir = Path("/tmp/test_sandbox")
    executor = ScriptExecutor(temp_dir=temp_dir)

    assert executor.temp_dir == temp_dir
    assert executor.scripts_dir.exists()
    assert executor.results_dir.exists()

    # Cleanup
    for path in [executor.scripts_dir, executor.results_dir, temp_dir]:
        try:
            path.rmdir()
        except Exception:
            pass
