"""Tests for session markers registered in conftest."""

from unittest.mock import MagicMock

import pytest

from tests.conftest import docker_daemon_available


def test_docker_daemon_available_false_when_client_fails(monkeypatch):
    """A daemon that rejects the client is reported as unavailable."""
    docker = pytest.importorskip("docker")
    from docker.errors import DockerException

    def fail_from_env():
        """Raise the same error docker.from_env raises without a daemon."""
        raise DockerException("docker daemon is not running")

    monkeypatch.setattr(docker, "from_env", fail_from_env)
    assert docker_daemon_available() is False


def test_docker_daemon_available_true_when_ping_succeeds(monkeypatch):
    """A daemon that answers ping is reported as available."""
    docker = pytest.importorskip("docker")
    client = MagicMock()
    monkeypatch.setattr(docker, "from_env", lambda: client)

    assert docker_daemon_available() is True
    client.ping.assert_called_once()
