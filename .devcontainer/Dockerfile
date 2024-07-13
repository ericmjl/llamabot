FROM condaforge/mambaforge

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# This Dockerfile adds a non-root user with sudo access. Use the "remoteUser"
# property in devcontainer.json to use it. On Linux, the container user's GID/UIDs
# will be updated to match your local UID/GID (when using the dockerFile property).
# See https://aka.ms/vscode-remote/containers/non-root-user for details.
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Copy environment.yml (if found) to a temp locaition so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yml .

# Configure apt and install packages
RUN mamba env create -f environment.yml -y

COPY docs docs
COPY llamabot llamabot
COPY tests tests
COPY pyproject.toml pyproject.toml
RUN /opt/conda/envs/llamabot/bin/pip install -e .
RUN /opt/conda/envs/llamabot/bin/mamba install -y pre-commit jupyter

# Always the final command
COPY . .

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=dialog
