"""PR Diff Bot and Prompts."""
from urllib.parse import urlparse

import outlines.text as text
import requests

from llamabot.bot.simplebot import SimpleBot


def diffbot() -> SimpleBot:
    """Return a diffbot instance.

    :return: A diffbot instance.
    """
    return SimpleBot(
        """
You are a GitHub PR Diff Expert.
"""
    )


def is_valid_github_url(pr_url: str) -> bool:
    """
    Check if the given URL is a valid GitHub URL.

    This function checks if the provided URL has a valid scheme (http or https) and
    if the domain contains "github.com". It returns True if the URL is valid, and
    False otherwise.

    :param pr_url: The URL to be checked for validity.
    :return: True if the URL is a valid GitHub URL, False otherwise.
    """
    try:
        parsed_url = urlparse(pr_url)
        return (
            parsed_url.scheme in ["http", "https"] and "github.com" in parsed_url.netloc
        )
    except ValueError:
        return False


def get_github_diff(pr_url: str):
    """Get the diff of a GitHub PR.

    :param pr_url: The URL of the PR to get the diff of.
    :return: The diff of the PR.
    :raises ValueError: If the request fails.
    """
    if not pr_url.endswith(".diff"):
        pr_url = pr_url + ".diff"

    r = requests.get(pr_url)
    # Check if the request returns a 200 status
    if r.status_code != 200:
        raise ValueError(f"Request failed with status code {r.status_code}")

    content = r.text
    return content


@text.prompt
def summarize(diff: str) -> str:
    """Please provide a summary of the diff.

    Use bullet points or numbered points.
    Restrict yourself to 1-3 bullet points.
    Focus on summarizing what was added, removed, and changed at a conceptual level.

    {{ diff }}
    # noqa: DAR101
    """


@text.prompt
def describe_advantages(diff: str) -> str:
    """Postulate how the how the following code changes provides an advantage over the existing codebase.

    {{ diff }}
    # noqa: DAR101
    """


@text.prompt
def suggest_improvements(diff: str) -> str:
    """Suggest improvements to the following code changes.

    {{ diff }}

    Focus on the following areas:

    1. Modularity of the function.
    2. Readability of the function.
    # noqa: DAR101
    """
