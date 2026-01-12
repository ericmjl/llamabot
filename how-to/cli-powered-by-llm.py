# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "llamabot[all]",
#     "marimo>=0.17.0",
#     "pydantic",
#     "typer",
# ]
# ///

import marimo

__generated_with = "0.18.3"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## How to Build an LLM-Powered CLI

    Learn how to build a command-line interface that uses LLMs to generate structured outputs.
    This guide shows you how to create a CLI tool that automatically generates commit messages
    from git diffs using StructuredBot.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Prerequisites

    Before you begin, ensure you have:

    - **Ollama installed and running locally**: Visit [ollama.ai](https://ollama.ai) to install
    - **Required Ollama model**: Run `ollama pull gemma3n:latest` (or another model that supports structured outputs)
    - **Python 3.10+** with llamabot installed
    - **A git repository** to test the CLI with

    All llamabot models in this guide use the `ollama_chat/` prefix for local execution.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Goal

    By the end of this guide, you'll have built a CLI command that:

    - Takes a git diff as input
    - Uses StructuredBot to generate a conventional commit message
    - Returns a validated, structured commit message
    - Can be integrated into git hooks for automatic commit message generation
    """
    )
    return


@app.cell
def _():
    from enum import Enum

    from pydantic import BaseModel, Field, model_validator

    import llamabot as lmb
    from llamabot.bot.structuredbot import StructuredBot
    from llamabot.prompt_manager import prompt

    return BaseModel, Enum, Field, lmb, model_validator, prompt, StructuredBot


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Step 1: Define Your Data Schema

    First, define the Pydantic model that represents your structured output.
    For commit messages, we'll use the conventional commit format.
    """
    )
    return


@app.cell
def _(BaseModel, Enum, Field, model_validator):
    class CommitType(str, Enum):
        """Type of commit following conventional commits."""

        fix = "fix"
        feat = "feat"
        build = "build"
        chore = "chore"
        ci = "ci"
        docs = "docs"
        style = "style"
        refactor = "refactor"
        perf = "perf"
        test = "test"

    class DescriptionEntry(BaseModel):
        """A single bullet point in the commit body."""

        txt: str = Field(
            ...,
            description="A single bullet point describing one major change in the commit.",
        )

        @model_validator(mode="after")
        def validate_description(self):
            """Validate description length."""
            if len(self.txt) > 160:
                raise ValueError(
                    "Description should be less than or equal to 160 characters."
                )
            return self

    class CommitMessage(BaseModel):
        """Structured commit message following conventional commits format."""

        commit_type: CommitType = Field(
            ...,
            description="Type of change (fix, feat, docs, etc.)",
        )
        scope: str = Field(
            ...,
            description="Scope of change (e.g., 'api', 'ui', 'auth')",
        )
        description: str = Field(
            ...,
            description="Concise summary of what the commit accomplishes (present tense)",
        )
        body: list[DescriptionEntry] = Field(
            default_factory=list,
            description="Optional detailed explanation as bullet points",
        )
        breaking_change: bool = Field(
            default=False,
            description="Whether this commit introduces a breaking change",
        )

    return CommitMessage, CommitType, DescriptionEntry


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Step 2: Create the StructuredBot

    StructuredBot ensures the LLM output matches your Pydantic schema exactly.
    It automatically retries if validation fails.
    """
    )
    return


@app.cell
def _(CommitMessage, StructuredBot, prompt):
    @prompt("system")
    def commitbot_sysprompt() -> str:
        """You are an expert software developer who writes excellent and accurate commit messages.
        You will be given a git diff as input, and you will generate a structured commit message
        following the conventional commits format. Ensure your output matches the provided schema exactly.
        """

    commit_bot = StructuredBot(
        system_prompt=commitbot_sysprompt(),
        pydantic_model=CommitMessage,
        model_name="ollama_chat/gemma3n:latest",
        stream_target="none",
    )
    return commit_bot, commitbot_sysprompt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Step 3: Test the Bot

    Let's test the bot with a sample git diff to see how it generates structured commit messages.
    """
    )
    return


@app.cell
def _(commit_bot):
    # Example git diff (in practice, you'd get this from `git diff --cached`)
    sample_diff = """
    diff --git a/src/api.py b/src/api.py
    index 1234567..abcdefg 100644
    --- a/src/api.py
    +++ b/src/api.py
    @@ -10,6 +10,8 @@ def get_user(user_id: int):
             raise ValueError("User ID must be positive")
         return db.query(User).filter(User.id == user_id).first()

    +def create_user(name: str, email: str):
    +    return db.add(User(name=name, email=email))
    +
     def delete_user(user_id: int):
         db.query(User).filter(User.id == user_id).delete()
    """

    # Generate commit message
    commit_message = commit_bot(sample_diff)
    commit_message
    return commit_message, sample_diff


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Step 4: View Observability with Spans

    StructuredBot automatically creates spans for observability. Let's see what information is tracked.
    """
    )
    return


@app.cell
def _(commit_bot):
    # Display spans to see observability data
    commit_bot.spans
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    The spans show:

    - **query**: The input (git diff)
    - **model**: Which model was used
    - **validation_attempts**: How many times validation was attempted
    - **validation_success**: Whether validation succeeded
    - **schema_fields**: Fields in the Pydantic model
    - **duration_ms**: How long the operation took

    This observability helps you debug issues and understand bot behavior.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Step 5: Create the CLI Command

    Now let's wrap this in a Typer CLI command that can be used from the terminal.
    """
    )
    return


@app.cell
def _():
    import subprocess
    from pathlib import Path

    import typer

    return Path, subprocess, typer


@app.cell
def _(Path, commit_bot, subprocess, typer):
    app = typer.Typer()

    @app.command()
    def compose():
        """Generate a commit message from the current git diff."""
        # Get the git diff
        result = subprocess.run(
            ["git", "diff", "--cached"],
            capture_output=True,
            text=True,
        )

        if not result.stdout.strip():
            typer.echo(
                "No staged changes found. Stage some changes with `git add` first."
            )
            raise typer.Exit(1)

        # Generate commit message
        try:
            commit_msg = commit_bot(result.stdout)

            # Format the commit message
            formatted = f"{commit_msg.commit_type.value}({commit_msg.scope}){': ' if commit_msg.breaking_change else ': '}{commit_msg.description}\n\n"
            if commit_msg.body:
                formatted += "\n".join(f"- {entry.txt}" for entry in commit_msg.body)
            if commit_msg.breaking_change:
                formatted += (
                    "\n\nBREAKING CHANGE: This commit introduces breaking changes."
                )

            typer.echo(formatted)

            # Optionally write to .git/COMMIT_EDITMSG
            commit_editmsg = Path(".git/COMMIT_EDITMSG")
            if commit_editmsg.parent.exists():
                commit_editmsg.write_text(formatted)
                typer.echo(f"\nCommit message written to {commit_editmsg}")

        except Exception as e:
            typer.echo(f"Error generating commit message: {e}", err=True)
            raise typer.Exit(1)

    return app, compose


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Step 6: Test the CLI

    You can now use this CLI command. In a real implementation, you'd register it with your main CLI app.
    For testing, you can call the function directly.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Step 7: Integrate with Git Hooks (Optional)

    To automatically generate commit messages, you can create a git hook:

    ```bash
    # Create the hook
    cat > .git/hooks/prepare-commit-msg << 'EOF'
    #!/bin/sh
    llamabot git compose
    EOF

    chmod +x .git/hooks/prepare-commit-msg
    ```

    Now when you run `git commit` without a message, it will automatically generate one.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Summary

    You've built an LLM-powered CLI that:

    - Uses StructuredBot to ensure validated, structured outputs
    - Integrates with git to generate commit messages automatically
    - Provides observability through spans
    - Handles validation retries automatically

    **Key Takeaways:**

    - Define your Pydantic schema first
    - Use StructuredBot for guaranteed schema compliance
    - Leverage spans for debugging and observability
    - Wrap bots in CLI commands for easy terminal access
    """
    )
    return


if __name__ == "__main__":
    app.run()
