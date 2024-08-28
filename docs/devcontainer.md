---
intents:
- Provide reader with an overview on how the dev container is built, and what software
  is installed in dev container.
- Explain the build steps with examples of possible common failure modes in the build
  and how to fix them, such that the reader knows how to fix them.
- Understand how the devcontainer.json file influences the development container.
linked_files:
- .github/workflows/build-devcontainer.yaml
- .devcontainer/Dockerfile
- .devcontainer/devcontainer.json
---
# Development Container Overview

The development container for Llamabot is built using a Dockerfile and is influenced by the `devcontainer.json` file. This document provides an overview of how the dev container is built, the software installed within it, and the build steps with examples of possible common failure modes and how to fix them.

## Building the Development Container

The development container is built using the Dockerfile located at `.devcontainer/Dockerfile`. The Dockerfile starts with a base image `ghcr.io/prefix-dev/pixi:latest` and sets up the environment by installing necessary software and dependencies. The Dockerfile also adds a non-root user with sudo access and sets up the environment for the development of Llamabot.

### Dockerfile Contents

The Dockerfile includes the following key steps:

1. Copies necessary files and directories into the container, including the `tests` directory and the `llamabot` directory.
2. Installs `curl` and `build-essential` for C++ (needed for ChromaDB).
3. Configures apt and installs packages using `pixi` based on the `pyproject.toml` file.
4. Installs Ollama within the Docker container.
5. Sets the final command and switches back to dialog for any ad-hoc use of `apt-get`.

### Ollama Software

The 'ollama' software is used to run large language models locally within the Docker container and is installed using the command `RUN curl -fsSL https://ollama.com/install.sh | sh`. Ollama is a crucial component for running large language models within the development container.

### Tests Directory

The `tests` directory contains the software tests to get started with development.

### Llamabot Directory

The 'llamabot' directory contains the source code and documentation for the Llamabot project, highlighting its significance in the development container.

## Devcontainer.json Influence

The `devcontainer.json` file located at `.devcontainer/devcontainer.json` influences the development container by specifying the build context, customizations for Visual Studio Code, forward ports, and post-create and post-start commands.

### Devcontainer.json Contents

- Specifies the Dockerfile and build context.
- Customizes Visual Studio Code settings and extensions for the development environment.
- Forwards port 8888 for the development environment.
- Specifies post-create and post-start commands for setting up the environment and running the Llamabot server.

### Devcontainer.json Commands

The 'postCreateCommand' is used to install pre-commit and set up the Python environment, while the 'postStartCommand' is used to start the 'ollama' server.

### Purpose of postCreateCommand and postStartCommand

The 'postCreateCommand' is executed after the development container is created to set up the environment, and the 'postStartCommand' is executed after the container is started to run the Llamabot server.

## Build Process

The build process for the development container is automated using GitHub Actions. The workflow is defined in the `.github/workflows/build-devcontainer.yaml` file. The workflow is triggered on a schedule and on pushes to the main branch. It sets up QEMU, Docker Buildx, and logs in to Docker Hub. It then builds and pushes the development container to Docker Hub.

### Build Process Workflow

1. Sets up QEMU and Docker Buildx.
2. Logs in to Docker Hub using secrets.
3. Builds and pushes the development container to Docker Hub with appropriate tags and caching configurations.

## Common Failure Modes

Common failure modes in the build process may include issues with Dockerfile syntax, missing dependencies, or failed package installations. These issues can be resolved by carefully reviewing the Dockerfile, ensuring all necessary files are copied, and troubleshooting package installations, including the installation process for the 'ollama' software.

## Conclusion

This updated documentation provides an overview of the development container for Llamabot, including the build process, influence of the devcontainer.json file, and common failure modes in the build process. Developers can use this documentation to understand how the development container is built and how to troubleshoot common issues during the build process.
