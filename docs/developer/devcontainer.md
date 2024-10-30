# Hack on LlamaBot with a Development Container

Development containers are one of the easiest ways to get set up with LlamaBot.
In this guide, we will show you
how to get set up with LlamaBot using development containers.

Our development container ships with Ollama installed in it,
which simplifies installation for individuals wanting to get set up.
This is particularly useful for individuals running Windows machines.

## Pre-requisites

Your system should have `git` and Docker both installed on it.

Additionally, we recommend that you configure Docker
to be able to use at least 12-14GB of RAM
to ensure that local LLMs have enough RAM to run.

## Brief Instructions

1. Fork LlamaBot to your own GitHub account.
2. Git clone the repository to your local machine using your favorite method, which may be one or more of the following:
   - HTTPS/SSH at the terminal
   - `gh repo clone` (using the GitHub CLI)
   - GitHub Desktop
3. Open the repository within VSCode.
4. Within VSCode, type `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` to bring up the command palette.
5. Type `Dev Containers: Rebuild and Reopen in Container`, and hit `Enter`. This will setup the container, install `pre-commit` and `ipykernel`, and the `pixi` project
6. The local repository is being mounted on `/workspaces/llamabot`.
7. On your fork, create a new branch.
8. Make the appropriate file changes you were thinking of and push them to your fork.
9. Make a pull request back to `ericmjl/llamabot:main` (i.e. the main branch as the target branch).

> Would you like to improve this documentation? We would love to see how you can make it better!
