FROM ghcr.io/prefix-dev/pixi:latest

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# This Dockerfile adds a non-root user with sudo access. Use the "remoteUser"
# property in devcontainer.json to use it. On Linux, the container user's GID/UIDs
# will be updated to match your local UID/GID (when using the dockerFile property).
# See https://aka.ms/vscode-remote/containers/non-root-user for details.
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Copy lockfile and config file for pixi to install env
COPY pixi.lock .
COPY pyproject.toml .
# Copy docs source, llamabot source, and software tests to get started with development
COPY docs docs
COPY llamabot llamabot
COPY tests tests

# Install curl to get ollama
# Install build-essential for C++ (needed for ChromaDB)
RUN apt-get update && apt-get install -y curl build-essential git git-lfs

# Configure apt and install packages
RUN /usr/local/bin/pixi install --manifest-path pyproject.toml

# Install Ollama within Docker container to run large language models locally
# RUN curl -fsSL https://ollama.com/install.sh | sh

# Always the final command
COPY . .

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=dialog
