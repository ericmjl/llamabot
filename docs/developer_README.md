# Developer Guide

This is the developer guide for LlamaBot.

LlamaBot implements a Pythonic interface to LLMs,
making it much easier to experiment with LLMs in a Jupyter notebook
and build Python apps that utilize LLMs.
All models supported by [LiteLLM](https://github.com/BerriAI/litellm) are supported by LlamaBot.

## Setting up your developer environment

Make sure that you have followed the main installation instructions before jumping to the next steps.

## Option 1: Development Container (Recommended)

For the easiest setup experience, especially on Windows, use the pre-configured development container:

### Prerequisites
- <a href="https://git-scm.com/downloads" target="_blank">Git</a> installed on your system
- <a href="https://www.docker.com/get-started" target="_blank">Docker</a> installed and running
- <a href="https://code.visualstudio.com/" target="_blank">VS Code</a> with the <a href="https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers" target="_blank">Dev Containers extension</a>

### Setup Steps
1. **Fork and Clone**: Fork the LlamaBot repository on GitHub and clone your fork locally
2. **Open in VS Code**: Open the cloned repository folder in VS Code
3. **Reopen in Container**: When prompted (or use Command Palette > "Dev Containers: Reopen in Container"), VS Code will build and start the development container
4. **Wait for Setup**: The container includes Python, all dependencies, and even Ollama pre-installed

The development container provides:
- Pre-configured Python environment with all dependencies
- Ollama for local LLM testing
- Pre-commit hooks already set up
- All development tools ready to use

## Option 2: Local Development with Pixi (Alternative)

If you prefer local development, use pixi for dependency management:

```bash
# Install pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# Set up the development environment
pixi install

# Enter the development shell
pixi shell
```

## Option 3: Creating a Conda virtual environment (Legacy)

It is recommended to have <a href="https://conda.io/projects/conda/en/latest/user-guide/install/index.html" target="_blank">Conda</a> available on your system.

```bash
conda env create -f environment.yml
```

After the conda environment creation, you may need to run the following commands.

```bash
conda activate llamabot
```

The following command will install all your pip dependencies

```bash
pip install -e .
```

## Installing pre-commit hook

It is recommended to install and use the <a href="https://pre-commit.com/" target="_blank">pre-commit</a> hook if you plan to commit this project.
This will check for any issues before committing your code. It is also one of the recommended developers' best practices.

The following command will install pre-commit in your virtual environment.

```bash
pre-commit install
```

Now before you commit your changes, running the pre-commit command below will automatically run code checks for you.
If there are any issues that pre-commit finds, such as missing line breaks, lint errors or others,
then please make the appropriate changes, rerun pre-commit to ensure that the checks pass and then commit the files.

```bash
pre-commit run
```

## Configuring your downloaded model

If you are using a local model, you may need to update the `DEFAULT_LANGUAGE_MODEL` to your local model path by
running the following command:

Replace <your_model_name> wth your locally downloaded model.

For example:

- If you are using Mistral: replace `<your_model_name>` with `mistral`.
- If you are using Llama3: replace `<your_model_name>` with `llama3`.

```bash
llamabot configure default-model --model-name "ollama/<your_model_name>"
```

OR

You could also pass the language model as an argument when extending/creating your own model from the
[SimpleBot](../llamabot/bot/simplebot.py) api.

Example:

```python
feynman = SimpleBot(
    "You are Richard Feynman. You will be given a difficult concept, and your task is to explain it back.",
    model_name="ollama/llama3"
)
```
