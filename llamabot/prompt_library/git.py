"""Prompts for writing git stuff."""
from llamabot import SimpleBot
import os

try:
    from outlines import text
except ImportError:
    import warnings

    warnings.warn(
        "Please install the `outlines` package to use the llamabot prompt library."
    )


def commitbot():
    """Return a commitbot instance.

    It is hard-coded to use gpt-3.5-turbo-16k-0613 as its model.
    This model is sufficient for the quality of commit messages,
    matching gpt-4-32k but being 1/10 the cost.

    :return: A commitbot instance.
    """
    return SimpleBot(
        "You are an expert user of Git.",
        model_name=os.getenv("OPENAI_DEFAULT_MODEL", "gpt-3.5-turbo-16k-0613"),
    )


@text.prompt
def write_commit_message(diff: str):
    """Please write a commit message for the following diff.

    {{ diff }}

    # noqa: DAR101

    Use the Conventional Commits specification to write the diff.

    [COMMIT MESSAGE BEGIN]
    <type>[optional scope]: <description>

    [optional body]

    [optional footer(s)]
    [COMMIT MESSAGE END]

    The commit contains the following structural elements,
    to communicate intent to the consumers of your library:

    fix: a commit of the type fix patches a bug in your codebase
        (this correlates with PATCH in Semantic Versioning).
    feat: a commit of the type feat introduces a new feature to the codebase
        (this correlates with MINOR in Semantic Versioning).
    BREAKING CHANGE: a commit that has a footer BREAKING CHANGE:,
        or appends a ! after the type/scope,
        introduces a breaking API change
        (correlating with MAJOR in Semantic Versioning).
        A BREAKING CHANGE can be part of commits of any type.

    types other than fix: and feat: are allowed,
    for example @commitlint/config-conventional
    (based on the Angular convention) recommends
    build:, chore:, ci:, docs:, style:, refactor:, perf:, test:, and others.

    footers other than BREAKING CHANGE: <description> may be provided
    and follow a convention similar to git trailer format.

    Additional types are not mandated by the Conventional Commits specification,
    and have no implicit effect in Semantic Versioning
    (unless they include a BREAKING CHANGE).
    A scope may be provided to a commit's type,
    to provide additional contextual information and is contained within parenthesis,
    e.g., feat(parser): add ability to parse arrays.
    Within the optional body section, prefer the use of bullet points.

    Final instructions:

    1. Do not fence the commit message with back-ticks or quotation marks.
    2. Do not add any other text except the commit message itself.
    3. Only write out the commit message.

    [BEGIN COMMIT MESSAGE]
    """


@text.prompt
def compose_release_notes(commit_log):
    """Here is a commit log:

    # noqa: DAR101

    {{ commit_log }}

    Please write for me the release notes.
    The notes should contain a human-readable summary
    of each new feature that was added.

    Follow the following format:

        ## Version <version number>

        <brief summary of the new version>

        ### New Features

        - <describe in plain English> (<commit's first 6 letters>) (<commit author>)
        - <describe in plain English> (<commit's first 6 letters>) (<commit author>)

        ### Bug Fixes

        - <describe in plain English> (<commit's first 6 letters>) (<commit author>)

        ### Deprecations

        - <describe in plain English> (<commit's first 6 letters>) (<commit author>)
    """
