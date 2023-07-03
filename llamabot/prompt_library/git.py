"""Prompts for writing git stuff."""
from outlines import text

from llamabot import SimpleBot


def commitbot():
    """Return a commitbot instance.

    :return: A commitbot instance.
    """
    return SimpleBot("You are an expert user of Git.")


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

    The commit contains the following structural elements, to communicate intent to the consumers of your library:

    fix: a commit of the type fix patches a bug in your codebase
        (this correlates with PATCH in Semantic Versioning).
    feat: a commit of the type feat introduces a new feature to the codebase
        (this correlates with MINOR in Semantic Versioning).
    BREAKING CHANGE: a commit that has a footer BREAKING CHANGE:, or appends a ! after the type/scope,
        introduces a breaking API change (correlating with MAJOR in Semantic Versioning).
        A BREAKING CHANGE can be part of commits of any type.

    types other than fix: and feat: are allowed,
    for example @commitlint/config-conventional (based on the Angular convention) recommends
    build:, chore:, ci:, docs:, style:, refactor:, perf:, test:, and others.

    footers other than BREAKING CHANGE: <description> may be provided
    and follow a convention similar to git trailer format.

    Additional types are not mandated by the Conventional Commits specification,
    and have no implicit effect in Semantic Versioning (unless they include a BREAKING CHANGE).
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
