"""Fixtures for tests."""

import shutil
import subprocess

import pytest


def pytest_configure(config):
    """Register the ollama marker."""
    config.addinivalue_line(
        "markers", "requires_ollama: test requires a running ollama daemon"
    )


@pytest.fixture(scope="session", autouse=True)
def pull_gemma2_model():
    """Fixture to pull the gemma2:2b model once per session.

    If ollama is not installed or the pull fails, the error is stored
    so that tests marked ``@pytest.mark.requires_ollama`` can be skipped
    gracefully instead of crashing the entire test session.
    """
    if not shutil.which("ollama"):
        pytest.ollama_available = False
        return

    try:
        subprocess.run(
            ["ollama", "pull", "gemma2:2b"],
            check=True,
            capture_output=True,
            timeout=120,
        )
        pytest.ollama_available = True
    except (subprocess.CalledProcessError, FileNotFoundError, TimeoutError):
        pytest.ollama_available = False


@pytest.fixture(autouse=True)
def skip_without_ollama(request):
    """Skip tests marked ``requires_ollama`` when ollama is unavailable."""
    marker = request.node.get_closest_marker("requires_ollama")
    if marker and not getattr(pytest, "ollama_available", False):
        pytest.skip("ollama is not available")
