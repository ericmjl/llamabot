"""Module for handling secure script execution in Docker containers.

This module provides functionality for executing agent-generated scripts
in a secure Docker container environment, following best practices
for sandboxing and security.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Mapping

import docker
from docker.errors import DockerException
from loguru import logger
from pydantic import BaseModel


class ScriptMetadata(BaseModel):
    """Metadata for an agent-generated script.

    :param requires_python: Python version requirement
    :param dependencies: List of pip dependencies
    :param auth: Agent ID hash for authentication
    :param purpose: Description of script's purpose
    :param timestamp: When the script was generated
    """

    requires_python: str
    dependencies: list[str]
    auth: str
    purpose: str
    timestamp: datetime


class ScriptExecutor:
    """Handles secure execution of agent-generated scripts in Docker containers.

    :param temp_dir: Directory for storing temporary script files
    :param docker_client: Docker client instance
    """

    def __init__(self, temp_dir: Optional[Path] = None):
        self.temp_dir = temp_dir or Path(tempfile.mkdtemp())
        self.scripts_dir = self.temp_dir / "agent_scripts"
        self.results_dir = self.temp_dir / "agent_results"

        # Create directories
        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
        except DockerException as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            raise

    def write_script(
        self, code: str, metadata: ScriptMetadata, filename: Optional[str] = None
    ) -> Path:
        """Write a script with metadata to the scripts directory.

        :param code: The Python code to write
        :param metadata: Script metadata
        :param filename: Optional custom filename
        :return: Path to the written script
        """
        if filename is None:
            filename = f"script_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"

        script_path = self.scripts_dir / filename

        # Format metadata block
        metadata_block = "# /// script\n"
        metadata_block += f'# requires-python = "{metadata.requires_python}"\n'
        metadata_block += "# dependencies = [\n"
        for dep in metadata.dependencies:
            metadata_block += f'#   "{dep}",\n'
        metadata_block += "# ]\n"
        metadata_block += f'# auth = "{metadata.auth}"\n'
        metadata_block += f'# purpose = "{metadata.purpose}"\n'
        metadata_block += f'# timestamp = "{metadata.timestamp.isoformat()}"\n'
        metadata_block += "# ///\n\n"

        # Write script with metadata
        with open(script_path, "w") as f:
            f.write(metadata_block + code)

        return script_path

    def build_container(self) -> None:
        """Build the Docker container for script execution."""
        dockerfile = """
# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install the project into `/app`
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT []

# Run as non-root user for security
USER nobody

# Run the script with uv
CMD ["uv", "run", "--system-site-packages=false"]
"""

        # Write Dockerfile
        dockerfile_path = self.temp_dir / "Dockerfile"
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile)

        # Build image
        try:
            self.docker_client.images.build(
                path=str(self.temp_dir),
                tag="agent-runner",
                dockerfile="Dockerfile",
                rm=True,
            )
        except DockerException as e:
            logger.error(f"Failed to build Docker image: {e}")
            raise

    def run_script(self, script_path: Path, timeout: int = 30) -> Dict[str, Any]:
        """Run a script in the Docker container.

        :param script_path: Path to the script to run
        :param timeout: Timeout in seconds
        :return: Script execution results
        :raises: Various exceptions for execution failures
        """
        try:
            # Ensure container image exists
            self.build_container()

            # Define volume configuration using proper typing
            volumes: Mapping[str, Mapping[str, str]] = {
                str(self.scripts_dir): {"bind": "/app/scripts", "mode": "ro"},
                str(self.results_dir): {"bind": "/app/results", "mode": "rw"},
            }

            # Run container with security constraints
            container = self.docker_client.containers.run(
                "agent-runner",
                f"/app/scripts/{script_path.name}",
                volumes=volumes,
                read_only=True,
                network_mode="none",
                mem_limit="512m",
                nano_cpus=1_000_000_000,  # 1 CPU
                security_opt=["no-new-privileges"],
                cap_drop=["ALL"],
                detach=True,
            )

            try:
                # Wait for result with timeout
                result = container.wait(timeout=timeout)
                logs = container.logs().decode("utf-8")

                if result["StatusCode"] != 0:
                    raise RuntimeError(f"Script failed: {logs}")

                # Parse result from logs
                try:
                    return json.loads(logs)
                except json.JSONDecodeError:
                    return {"output": logs}

            finally:
                container.remove(force=True)

        except DockerException as e:
            logger.error(f"Docker execution error: {e}")
            raise
