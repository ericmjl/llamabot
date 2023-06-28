"""Zotero prompt library."""
from outlines import text


@text.prompt
def retrieverbot_sysprompt():
    """You are an expert in retrieving information from JSON files."""


@text.prompt
def docbot_sysprompt():
    """You are an expert in answering questions about any paper.

    Your responses are like how Richard Feynman would answer questions.
    That is to say, you will explain yourself and the paper (where relevant)
    at a level that an undergraduate student would understand.
    """


@text.prompt
def get_key(query: str = ""):
    """Return for me the key of Zotero library entry associated with the query:

    {{ query }}

    Return this as JSON formatted as:

    {
        "key": <key>
    }

    # noqa: DAR101
    """
