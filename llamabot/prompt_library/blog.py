"""Blogging prompts."""
from outlines import text

from llamabot import SimpleBot


def blogging_bot():
    """Blogging bot.

    :return: The blogging bot.
    """
    bot = SimpleBot("You are an expert blogger.")
    return bot


@text.prompt
def blog_tagger_and_summarizer(blog_post):
    """This is a blog post that I just wrote.

    {{ blog_post }}

    Please return for me up to 15 blog tags in lowercase.
    They should be at most 2 words long.

    Also, please return for me a summary of the blog post,
    written in first-person tone,
    that is at most 100 words long.
    It should be entertaining without being overly so,
    and should entice readers to read the blog post.
    Use emojis where appropriate.

    Respond with a JSON formatted as follows:

    {
        "tags": [<tag1>, <tag2>,...],
        "summary": <summary>
    }

    # noqa: DAR101
    """
