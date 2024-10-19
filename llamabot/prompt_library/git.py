"""Prompts for writing git stuff."""

from llamabot import SimpleBot
from llamabot.prompt_manager import prompt


def commitbot():
    """Return a commitbot instance.

    It is hard-coded to use mistral/mistral-medium as its model.
    This model is sufficient for the quality of commit messages,
    matching gpt-4-32k but being a fraction of the cost.

    :return: A commitbot instance.
    """
    return SimpleBot(
        "You are an expert user of Git.",
        model_name="gpt-4-0125-preview",
        stream_target="stdout",
    )


@prompt(role="user")
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


@prompt(role="user")
def compose_git_activity_report(log_info: str, hours: int) -> str:
    """Given the following git log information for the last {{ hours }} hours,
    please write a concise report summarizing the work done during this period.
    Highlight key changes, features, or fixes.
    Use markdown formatting.

    [[GIT LOG BEGIN]]
    {{ log_info }}
    [[GIT LOG END]]

    Please format your report as follows:

    1. Start with a brief summary of the overall work done in the specified time period.
    2. Use the following sections, omitting any that are not applicable:
       - New Features
       - Bug Fixes
       - Improvements
       - Documentation
       - Other Changes
    3. Under each section, use bullet points to list the relevant changes.
    4. For each bullet point, include:
       - A concise description of the change
       - The commit hash (first 7 characters)
       - The author's name (if available)
    5. If there are many similar changes, group them together and provide a summary.
    6. Highlight any significant or breaking changes.
    7. End with a brief conclusion or outlook, if appropriate.

    Example format:

    # Work Summary for the Last {{ hours }} Hours

    ## Overview
    [Brief summary of overall work]

    ## New Features
    - Implemented user authentication system (abc1234) (Jane Doe)
    - Added export functionality for reports (def5678) (John Smith)

    ## Bug Fixes
    - Fixed crash in data processing module (ghi9101) (Alice Johnson)

    ## Improvements
    - Optimized database queries for faster performance (jkl1121) (Bob Wilson)

    ## Conclusion
    [Brief conclusion or future outlook]
    """


@prompt(role="user")
def compose_release_notes(commit_log: str) -> str:
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
