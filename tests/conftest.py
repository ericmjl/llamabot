"""Fixtures for tests."""

import subprocess
import pytest


@pytest.fixture(scope="session", autouse=True)
def pull_gemma2_model():
    """Fixture to pull the gemma2:2b model once per session."""
    subprocess.run(["ollama", "pull", "gemma2:2b"], check=True)
