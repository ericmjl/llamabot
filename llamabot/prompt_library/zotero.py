"""Zotero prompt library."""

from llamabot.prompt_manager import prompt


@prompt(role="system")
def retrieverbot_sysprompt():
    """You are an expert in retrieving information from JSON files."""


@prompt(role="system")
def docbot_sysprompt():
    """You are an expert in answering questions about any paper.

    Respond the way that Richard Feynman would answer questions
    to a 2nd year undergraduate student.
    """


@prompt(role="user")
def paper_summary():
    """Please synthesize a summary of this paper using the information provided to you.

    Your summary should not be a mere regurgitation of the abstract.
    Rather, your summary should highlight the key findings,
    methodology, and implications.
    """


@prompt(role="user")
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
