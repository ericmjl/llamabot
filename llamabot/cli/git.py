"""Git subcommand for LlamaBot CLI."""

from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from pydantic import BaseModel, Field, model_validator
from pyprojroot import here
from rich.console import Console

from llamabot import SimpleBot, prompt
from llamabot.bot.structuredbot import StructuredBot
from llamabot.code_manipulation import get_git_diff
from llamabot.config import default_language_model
from llamabot.prompt_library.git import (
    compose_git_activity_report,
    compose_release_notes,
)

gitapp = typer.Typer()


class CommitType(str, Enum):
    """Type of commit."""

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
    other = "other"


class DescriptionEntry(BaseModel):
    """Description entry."""

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
    """Commit message."""

    commit_type: CommitType = Field(
        ...,
        description=(
            "Type of change. Should usually be fix or feat. "
            "But others, based on the Angular convention, are allowed, "
            "such as build, chore, ci, docs, style, refactor, perf, test, and others."
        ),
    )
    scope: str = Field(
        ...,
        description=(
            "Scope of change. "
            "The scope in a conventional commit message represents "
            "a contextual identifier for the changes made in that particular commit. "
            "It provides a way to categorize and organize commits, "
            "making it easier to track and manage changes in a project. "
            "It can be thought of as a descriptor for the "
            "location, component, or module "
            "that is being modified or affected by the commit. "
            "For example, if a developer is working on a web application "
            "with multiple components such as the login system, "
            "user profile, and settings page, they can use scopes like "
            "login, profile or settings to indicate "
            "which part of the application the commit is related to. "
            "This helps other team members and future contributors understand "
            "the purpose of the commit "
            "without having to inspect the actual changes right away."
        ),
    )
    description: str = Field(
        ...,
        description=(
            "The description provides a concise summary of what the commit accomplishes, "
            "making it easier for team members and contributors to understand "
            "the purpose of the commit at a glance. "
            "It should be written in the present tense and be clear and informative. "
            "It should convey what the code changes do, "
            "rather than how they were implemented. "
            "It is not necessary to provide every detail in the description; "
            "instead, focus on the high-level overview of the modifications."
        ),
    )

    body: list[DescriptionEntry] = Field(
        ...,
        description=(
            "Unlike the description, "
            "which provides a concise summary of the changes made in the commit, "
            "the body allows for a more detailed explanation of the commit. "
            "The body section is used to provide additional context, reasoning, "
            "or implementation details that might be helpful "
            "for other developers, reviewers, or future contributors "
            "to understand the commit thoroughly. "
            "It is particularly useful "
            "when the changes introduced in the commit are complex "
            "or when there are specific design decisions that need to be explained. "
            "It is worth mentioning that having a body section is optional, "
            "and not every commit requires it. "
            "Simple and straightforward changes "
            "might not need a detailed explanation beyond the description."
            "If provided, this should be bullet points."
        ),
    )

    breaking_change: bool = Field(
        ..., description="Whether or not there is a breaking change in the commit. "
    )

    footer: str = Field(
        ..., description="An optional footer. Most of the time should be empty."
    )
    emoji: str = Field(..., description="An emoji that represents the commit content.")

    @model_validator(mode="after")
    def validate_body(self):
        """Validate the body length."""
        if len(self.body) > 10:
            raise ValueError("Description entries should be no more than 10 in length.")
        return self

    def format(self) -> str:
        """Format the commit message according to the provided model.

        :return: Formatted commit message as a string.
        """
        return _fmt(self)


@prompt(role="system")
def _fmt(cm) -> str:
    """{{ cm.commit_type.value }}({{ cm.scope }}){{ cm.emoji }}{%if cm.breaking_change %}!{% else %}{% endif %}: {{ cm.description }}

    {% for bullet in cm.body %}- {{ bullet.txt }}
    {% endfor %}

    {% if cm.footer %}{{ cm.footer }}{% endif %}
    """  # noqa: E501


def commitbot(model_name: str = default_language_model()) -> StructuredBot:
    """Return a structured bot for writing commit messages."""

    @prompt(role="system")
    def commitbot_sysprompt() -> str:
        """You are an expert software developer
        who writes excellent and accurate commit messages.
        You are going to be given a diff as input,
        and you will generate a structured JSON output
        based on the pydantic model provided.
        Ensure that your commit message is formatted as a conventional commit message.
        """

    bot = StructuredBot(
        system_prompt=commitbot_sysprompt(),
        pydantic_model=CommitMessage,
        model_name=model_name,
        stream_target="none",
    )
    return bot


def resolve_git_paths() -> tuple[Path, Path]:
    """Resolve the git directories of the repository enclosing the cwd.

    Works from any subdirectory and inside linked worktrees, where ``.git``
    is a file pointing at a per-worktree admin directory rather than the
    shared repository directory.

    :return: A ``(git_dir, common_dir)`` tuple. ``git_dir`` is the
        per-worktree admin directory (holds ``HEAD``, ``index``,
        ``COMMIT_EDITMSG``); ``common_dir`` is the shared repository
        directory (holds ``config``, ``refs``, ``hooks``). They are equal in
        a plain checkout.
    :raises RuntimeError: If the current directory is not inside a git
        repository.
    """
    from git import InvalidGitRepositoryError, Repo

    try:
        repo = Repo(search_parent_directories=True)
    except InvalidGitRepositoryError as e:
        raise RuntimeError(
            "You must be inside a git repository to use this command. "
            "Please `cd` into your git repository and try again, "
            "or use `git init` to create a new repository (if you haven't already)."
        ) from e
    return Path(repo.git_dir), Path(repo.common_dir)


@gitapp.command()
def hooks(model_name: str = default_language_model()):
    """Install a commit message hook that runs the commit message through the bot.

    The hook is installed into the shared hooks directory of the enclosing
    repository, so it is active for the main checkout and all linked
    worktrees. May be invoked from any subdirectory.

    :raises RuntimeError: If the current directory is not inside a git
        repository.
    """
    _, common_dir = resolve_git_paths()

    hooks_dir = common_dir / "hooks"
    hooks_dir.mkdir(exist_ok=True)
    hook_path = hooks_dir / "prepare-commit-msg"

    contents = f"""#!/bin/sh

echo "Script started with arguments: $@"

# Check if the arguments contain '-m'
commit_message_provided=false

for arg in "$@"; do
    if [ "$arg" = "message" ]; then
        commit_message_provided=true
        break
    fi
done

if $commit_message_provided; then
    echo "Commit message provided, skipping llamabot git compose."
else
    echo "No commit message found, running llamabot git compose."
    llamabot git compose --model-name {model_name}
fi
"""
    with open(hook_path, "w") as f:
        f.write(contents)
    hook_path.chmod(0o755)
    typer.echo(f"Commit message hook successfully installed at {hook_path}! 🎉")


@gitapp.command()
def compose(model_name: str = default_language_model()):
    """Autowrite commit message based on the diff.

    Writes the composed message to the ``COMMIT_EDITMSG`` of the enclosing
    repository's per-worktree git directory, so ``git commit`` picks it up
    in plain checkouts and linked worktrees alike.

    :raises RuntimeError: If the current directory is not inside a git
        repository.
    """
    try:
        git_dir, _ = resolve_git_paths()
        diff = get_git_diff()
        bot = commitbot(model_name)
        response = bot(diff, verbose=True)
        print(response.format())
        with open(git_dir / "COMMIT_EDITMSG", "w") as f:
            f.write(response.format().content)
    except Exception as e:
        typer.echo(f"Error encountered: {e}", err=True)
        typer.echo("Please write your own commit message.", err=True)
        raise e


@gitapp.command()
def write_release_notes(
    release_notes_dir: Path = typer.Option(
        Path("./docs/releases"),
        help="The directory to write the release notes to.",
    ),
    version: Optional[str] = typer.Option(
        None,
        help="Version number (e.g., '0.1.56'). If provided, used for filename and passed to LLM. If not provided, inferred from newest git tag (current behavior).",
    ),
):
    """Write release notes for the latest two tags to the release notes directory.

    :param release_notes_dir: The directory to write the release notes to.
        Defaults to "./docs/releases".
    :param version: Explicit version number. If not provided, inferred from git tags.
    """
    try:
        import git
    except ImportError:
        raise ImportError(
            "git is not installed. Please install it with `pip install llamabot[cli]`."
        )

    repo = git.Repo(here())
    tags = sorted(repo.tags, key=lambda t: t.commit.committed_datetime)

    if len(tags) == 0:
        raise ValueError(
            "No tags found. Please ensure bump2version has run and created a tag before generating release notes."
        )

    # Determine version string
    if version is not None and isinstance(version, str):
        # Explicit version provided - use as-is after stripping whitespace
        version_str = version.strip()
        # If the version is empty after stripping, fall back to inferring from tag
        if not version_str:
            newest_tag = tags[-1]
            version_str = newest_tag.name
    else:
        # Backward compatibility - infer from newest tag
        newest_tag = tags[-1]
        version_str = newest_tag.name

    # Determine commit range (same logic regardless of version source)
    if len(tags) == 1:
        # First release: get all commit messages from the very first commit
        log_info = repo.git.log()
    elif len(tags) == 2:
        # Second release: get commits from first tag to second tag
        log_info = repo.git.log(f"{tags[0].commit.hexsha}..{tags[1].commit.hexsha}")
    else:
        # Third+ release: get commits between the last two tags
        log_info = repo.git.log(f"{tags[-2].commit.hexsha}..{tags[-1].commit.hexsha}")

    # Generate release notes with explicit version
    console = Console()
    bot = SimpleBot(
        "You are an expert software developer "
        "who knows how to write excellent release notes based on git commit logs.",
        model_name=default_language_model(),
        stream_target="none",
    )

    with console.status("[bold green]Generating release notes...", spinner="dots"):
        notes = bot(compose_release_notes(log_info, version_str))

    # Write to file using explicit version
    release_notes_dir.mkdir(parents=True, exist_ok=True)
    # Ensure only one newline at the end of the file
    trimmed_notes = notes.content.rstrip() + "\n"

    # Write release notes using the version string as-is for filename
    filename = f"{version_str}.md"
    with open(release_notes_dir / filename, "w+") as f:
        f.write(trimmed_notes)


@gitapp.command()
def report(
    hours: Optional[int] = typer.Option(None, help="The number of hours to report on."),
    start_date: Optional[str] = typer.Option(
        None, help="The start date to report on. Format: YYYY-MM-DD"
    ),
    end_date: Optional[str] = typer.Option(
        None, help="The end date to report on. Format: YYYY-MM-DD"
    ),
    model_name: str = default_language_model(),
):
    """
    Write a report on the work done based on git commit logs.

    If hours is provided, it reports on the last specified hours.
    If start_date and end_date are provided, it reports on that date range.
    If neither is provided, it raises an error.

    :param hours: The number of hours to report on.
    :param start_date: The start date to report on.
    :param end_date: The end date to report on.
    :param model_name: The model name to use.
        Consult LiteLLM's documentation for options.
    """
    try:
        import pyperclip
    except ImportError:
        raise ImportError(
            "pyperclip is not installed. Please install it with `pip install llamabot[cli]`."
        )

    try:
        import git
    except ImportError:
        raise ImportError(
            "git is not installed. Please install it with `pip install llamabot[cli]`."
        )

    repo = git.Repo(here())

    if hours is not None:
        now = datetime.now()
        time_ago = now - timedelta(hours=hours)
        now_str = now.strftime("%Y-%m-%dT%H:%M:%S")
        time_ago_str = time_ago.strftime("%Y-%m-%dT%H:%M:%S")
    elif start_date and end_date:
        time_ago_str = start_date
        now_str = end_date
    else:
        raise ValueError(
            "Either 'hours' or both 'start_date' and 'end_date' must be provided."
        )

    log_info = repo.git.log(f"--since={time_ago_str}", f"--until={now_str}")
    bot = SimpleBot(
        "You are an expert software developer who writes excellent reports based on git commit logs.",
        model_name=model_name,
        stream_target="none",
    )

    console = Console()
    with console.status("[bold green]Generating report...", spinner="dots"):
        report = bot(
            compose_git_activity_report(
                log_info, str(hours) or f"from {start_date} to {end_date}"
            )
        )

    print(report.content)
    # Copy report content to clipboard
    pyperclip.copy(report.content)
    typer.echo("Report copied to clipboard. Paste it wherever you need!")
