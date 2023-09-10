"""Zotero prompt library."""
try:
    from outlines import text
except ImportError:
    import warnings

    warnings.warn(
        "Please install the `outlines` package to use the llamabot prompt library."
    )


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
    """Based on the context provided,
    return for me all of the keys that were provided for the query.

    Ensure that you're using the titles and abstracts.

    {{ query }}

    Return this as JSON:

    {
        "key": [<key1>, <key2>, ...]
    }

    The list should include only the keys for the papers that are relevant to the query.

    Do not return anything else except JSON.
    Ensure that you do not return any explanations of your response.
    If you cannot help with the request, return an empty list.

    # noqa: DAR101
    """
