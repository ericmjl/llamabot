"""Module for handling secure script execution in Docker containers.

This module provides functionality for executing agent-generated scripts
in a secure Docker container environment, following best practices
for sandboxing and security.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger
from pydantic import BaseModel


class ScriptMetadata(BaseModel):
    """Metadata for an agent-generated script.

    :param requires_python: Python version requirement
    :param dependencies: List of pip dependencies
    :param auth: Agent ID hash for authentication
    :param timestamp: When the script was generated
    """

    requires_python: str
    dependencies: list[str]
    auth: str
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
            import docker
            from docker.errors import DockerException
        except ImportError:
            raise ImportError(
                "The Python package `docker` cannot be found. Please install it using `pip install llamabot[agent]`."
            )

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
        metadata_block += f'# timestamp = "{metadata.timestamp.isoformat()}"\n'
        metadata_block += "# ///\n\n"

        # Write script with metadata

        full_script = metadata_block + code
        # print(full_script)

        with open(script_path, "w") as f:
            f.write(full_script)

        return script_path

    def build_container(self) -> None:
        """Build the Docker container for script execution."""
        try:
            from docker.errors import DockerException
        except ImportError:
            raise ImportError(
                "The Python package `docker` cannot be found. Please install it using `pip install llamabot[agent]`."
            )

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
        :return: Dictionary containing script execution results
            - stdout: Captured standard output
            - stderr: Captured standard error
            - status: Execution status code
        :raises RuntimeError: If the script execution fails
        """
        try:
            from docker.errors import DockerException
        except ImportError:
            raise ImportError(
                "The Python package `docker` cannot be found. Please install it using `pip install llamabot[agent]`."
            )

        try:
            # Ensure container image exists
            self.build_container()

            # Define volume configuration using proper typing
            volumes: dict[str, dict[str, str]] = {
                str(self.scripts_dir): {"bind": "/app/scripts", "mode": "ro"},
                str(self.results_dir): {"bind": "/app/results", "mode": "rw"},
            }

            # Add environment variables for uv
            env = {
                "UV_CACHE_DIR": "/tmp/uv-cache",
                "UV_SYSTEM_PYTHON": "false",
            }

            # Run container with security constraints
            # Create cache directory and run script in a single command
            # Note: tmpfs mounts are writable by default, so we just need to ensure the directory exists
            container = self.docker_client.containers.run(
                "agent-runner",
                [
                    "sh",
                    "-c",
                    f"mkdir -p /tmp/uv-cache && uv run /app/scripts/{script_path.name}",
                ],
                volumes=volumes,
                environment=env,
                tmpfs={
                    "/tmp": "size=2g,exec",  # General tmp space (includes uv-cache subdirectory)
                },
                read_only=True,
                mem_limit="2048m",
                nano_cpus=1_000_000_000,
                security_opt=["no-new-privileges"],
                cap_drop=["ALL"],
                detach=True,
            )

            try:
                # Wait for result with timeout
                result = container.wait(timeout=timeout)
                stdout = container.logs(stdout=True, stderr=False).decode("utf-8")
                stderr = container.logs(stdout=False, stderr=True).decode("utf-8")

                # Check if script execution failed
                if result["StatusCode"] != 0:
                    error_msg = stderr if stderr else stdout
                    raise RuntimeError(f"Script execution failed: {error_msg}")

                return {
                    "stdout": stdout,
                    "stderr": stderr,
                    "status": result["StatusCode"],
                }
            finally:
                container.remove(force=True)

        except DockerException as e:
            logger.error(f"Docker execution error: {e}")
            raise RuntimeError(f"Docker execution error: {e}")
