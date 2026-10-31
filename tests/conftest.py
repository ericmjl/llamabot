"""Fixtures for tests."""

import shutil
import subprocess

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_ollama: test requires a running ollama daemon"
    )
    config.addinivalue_line(
        "markers", "requires_docker: test requires a running Docker daemon"
    )


def docker_daemon_available() -> bool:
    """Return whether the Docker SDK can reach a daemon.

    Sandbox tests start containers. Without a daemon they fail in ``docker.from_env``
    instead of checking script execution.

    :return: True when ``docker.from_env().ping()`` succeeds.
    """
    try:
        import docker
        from docker.errors import DockerException
    except ImportError:
        return False

    try:
        client = docker.from_env()
        client.ping()
    except (DockerException, OSError):
        return False
    return True


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


@pytest.fixture(scope="session")
def docker_daemon_ready() -> bool:
    """Check the Docker daemon once per session.

    :return: True when a Docker daemon accepts API calls.
    """
    return docker_daemon_available()


@pytest.fixture(autouse=True)
def skip_without_docker(request, docker_daemon_ready: bool):
    """Skip tests marked ``requires_docker`` when no daemon is running.

    :param request: Pytest request for the current test.
    :param docker_daemon_ready: Session result of :func:`docker_daemon_available`.
    """
    marker = request.node.get_closest_marker("requires_docker")
    if marker and not docker_daemon_ready:
        pytest.skip("Docker daemon is not available")
