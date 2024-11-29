---
diataxis_type: explanation
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

# Llamabot Development Container Guide

This guide provides a comprehensive overview of the Llamabot development container, which ensures a consistent and reproducible development environment. It covers the container's construction, the software it includes, the automated build process, and solutions to common issues.

## Table of Contents

1. [Introduction](#introduction)
2. [Building the Development Container](#building-the-development-container)
   - [Dockerfile Overview](#dockerfile-overview)
   - [Installing Ollama](#installing-ollama)
3. [Devcontainer Configuration](#devcontainer-configuration)
   - [Build Context](#build-context)
   - [Visual Studio Code Customizations](#visual-studio-code-customizations)
   - [Port Forwarding](#port-forwarding)
   - [Volume Mounting](#volume-mounting)
   - [Post-Create and Post-Start Commands](#post-create-and-post-start-commands)
4. [Automated Build Process](#automated-build-process)
   - [GitHub Actions Workflow](#github-actions-workflow)
5. [Common Issues and Solutions](#common-issues-and-solutions)
   - [Dockerfile Syntax Errors](#dockerfile-syntax-errors)
   - [Missing Dependencies](#missing-dependencies)
   - [Package Installation Failures](#package-installation-failures)
6. [Conclusion](#conclusion)

## Introduction

To provide a seamless and consistent development experience, Llamabot utilizes a development container. This container ensures that all developers operate within the same environment, eliminating discrepancies caused by differing local setups. It includes all the necessary tools and dependencies required for developing Llamabot.

## Building the Development Container

The development container is defined by a Dockerfile located at `.devcontainer/Dockerfile`. The build process starts with a base image and progresses by installing additional software and configuring the environment.

### Dockerfile Overview

The Dockerfile performs the following key steps:

1. **Base Image**: Begins with the base image `ghcr.io/prefix-dev/pixi:latest`.
2. **Environment Configuration**: Sets `DEBIAN_FRONTEND=noninteractive` to suppress interactive prompts during package installations.
3. **Software Installation**:
   - Updates package lists.
   - Installs essential packages such as `curl` and `build-essential`.
   - These are necessary for building C++ extensions required by dependencies like ChromaDB.
4. **User Configuration**:
   - Creates a non-root user with `sudo` privileges for development purposes.
5. **Ollama Installation**:
   - Installs Ollama to enable local execution of large language models within the container.

An excerpt from the Dockerfile:

```dockerfile
FROM ghcr.io/prefix-dev/pixi:latest

ENV DEBIAN_FRONTEND=noninteractive

# Update package lists and install required packages
RUN apt-get update && apt-get install -y \
    curl \
    build-essential

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Additional user configuration (if applicable)
```

### Installing Ollama

Ollama is crucial for running large language models locally inside the container. It is installed using the following command:

```dockerfile
RUN curl -fsSL https://ollama.com/install.sh | sh
```

This command downloads and executes the Ollama installation script, setting up the environment necessary for Llamabot to function properly.

## Devcontainer Configuration

The `devcontainer.json` file, located at `.devcontainer/devcontainer.json`, configures the development container, particularly for use with Visual Studio Code's Remote - Containers extension.

Key configurations include:

### Build Context

Specifies the Dockerfile and context used to build the container:

```json
"build": {
  "dockerfile": "Dockerfile",
  "context": ".."
},
```

### Visual Studio Code Customizations

Defines settings and extensions to enhance the development experience within VS Code:

```json
"settings": {
  // Add VS Code settings here
},
"extensions": [
  // List of VS Code extensions
],
```

### Port Forwarding

Forwards necessary ports (e.g., port `8888`) to access services running within the container:

```json
"forwardPorts": [8888],
```

### Volume Mounting

Mounts directories into the container to address filesystem compatibility issues, such as case insensitivity on macOS:

```json
"mounts": [
  "source=llamabot-pixi,target=/workspace/.pixi,type=volume"
],
```

### Post-Create and Post-Start Commands

Automates environment setup and starts necessary services:

- **`postCreateCommand`**: Runs after the container is created. It installs dependencies and sets up the development environment.

  ```json
  "postCreateCommand": "pixi install && \
  /workspaces/llamabot/.pixi/envs/default/bin/pre-commit install && \
  /workspaces/llamabot/.pixi/envs/default/bin/python -m ipykernel install --user --name llamabot",
  ```

- **`postStartCommand`**: Runs each time the container starts. It launches the Ollama server or other required services.

  ```json
  "postStartCommand": "ollama serve",
  ```

## Automated Build Process

To ensure the development container remains up-to-date and accessible, an automated build process is set up using GitHub Actions.

### GitHub Actions Workflow

The workflow is defined in `.github/workflows/build-devcontainer.yaml` and is triggered on a schedule or when changes are pushed to the `main` branch.

Workflow steps include:

1. **Set Up QEMU and Docker Buildx**: Configures the environment to support building images for multiple architectures.

   ```yaml
   - name: Set up QEMU
     uses: docker/setup-qemu-action@v3

   - name: Set up Docker Buildx
     uses: docker/setup-buildx-action@v3
   ```

2. **Authenticate with Docker Hub**: Logs into Docker Hub using stored secrets to allow image pushing.

   ```yaml
   - name: Login to Docker Hub
     uses: docker/login-action@v3
     with:
       username: ${{ secrets.DOCKERHUB_USERNAME }}
       password: ${{ secrets.DOCKERHUB_TOKEN }}
   ```

3. **Build and Push the Container**: Builds the development container image and pushes it to Docker Hub with appropriate tags. Uses caching to speed up subsequent builds.

   ```yaml
   - name: Build and push
     uses: docker/build-push-action@v6
     with:
       push: true
       tags: ericmjl/llamabot-devcontainer:latest, ericmjl/llamabot-devcontainer:${{ github.sha }}
       file: .devcontainer/Dockerfile
       cache-from: type=gha
       cache-to: type=gha,mode=max
   ```

## Common Issues and Solutions

When building or using the development container, you might encounter some common issues. Below are symptoms and resolutions for these problems.

### Dockerfile Syntax Errors

**Symptoms**: The build process fails with syntax error messages.

**Solutions**:

- Review the Dockerfile for syntax errors.
- Ensure all commands are correctly formatted.
- Check that all referenced files and directories exist.

### Missing Dependencies

**Symptoms**: Errors indicating missing packages or libraries during the build or at runtime.

**Solutions**:

- Verify that all required packages are listed in the Dockerfile's `apt-get install` commands.
- Check for typos in package names.
- Run `apt-get update` before installing packages to ensure the package lists are up-to-date.

### Package Installation Failures

**Symptoms**: Failure messages during `apt-get install` or other package installations.

**Solutions**:

- Check network connectivity from the build environment.
- Ensure package repositories are reachable and not experiencing downtime.
- Examine the logs for specific error messages to identify the cause.
- Retry the build; transient network issues can sometimes cause failures.

## Conclusion

By utilizing a development container, Llamabot ensures a consistent and efficient development environment for all contributors. This guide has detailed the setup and configuration of the development container, the automated build process, and troubleshooting steps for common issues. Following these guidelines will facilitate a smooth development workflow and help maintain consistency across development environments.
