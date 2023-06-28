"""Zotero prompt library."""
from outlines import text


@text.prompt
def retrieverbot_sysprompt():
    """You are an expert in retrieving information from JSON files."""


@text.prompt
def get_key(title: str = "", author: str = ""):
    """Return for me the key of Zotero library entry associated with {{ title }}.

    The associated author(s) is: {{ author }}

    Return this as JSON formatted as:

    {
        "key": <key>
    }

    # noqa: DAR101
    """
