"""Blogging prompts."""
from outlines import text
from pydantic import BaseModel

from llamabot import SimpleBot


class BlogInformation(BaseModel):
    """Blog information."""

    tags: list[str]
    summary: str
    title: str


def blogging_bot():
    """Blogging bot.

    :return: The blogging bot.
    """
    bot = SimpleBot(
        "You are an expert blogger. Whenever you use hashtags, they are always lowercase.",
        temperature=0.3,
    )
    return bot


@text.prompt
def blog_title_tags_summary(blog_post, blog_info_model):
    """This is a blog post that I just wrote.

    {{ blog_post }}

    Please return for me up to 15 blog tags in lowercase.
    They should be at most 2 words long.
    Return one tag per line.
    No symbols, just the words, all lowercase.

    Also, please return for me a summary of the blog post,
    written in first-person tone,
    that is at most 100 words long.
    Use emojis where appropriate.

    Finally, please suggest a title for the blog post.
    It should be entertaining without being overly so,
    and should entice readers to read the blog post.

    Respond with a JSON formatted as follows:

    {{ blog_info_model | schema }}

    # noqa: DAR101
    """


@text.prompt
def compose_linkedin_post(blog_post):
    """This is a blog post that I just wrote:

    {{ blog_post }}

    Please compose for me a LinkedIn post
    that entices my network on LinkedIn to read it.
    Ensure that there is a call to action to interact with the post after reading
    to react with it, comment on it, or share the post with others,
    and to support my work on Patreon.
    My Patreon link is https://patreon.com/ericmjl/
    Include hashtags inline with the LinkedIn post and at the end of the post too.

    #noqa: DAR101
    """


@text.prompt
def compose_patreon_post(blog_post):
    """This is a blog post that I just wrote:

    {{ blog_post }}

    Please compose for me a Patreon post
    that entices my patrons on Patreon to read it.
    Ensure that there is a call to action to interact with the post after reading it,
    such as asking a question, or suggesting ideas that I did not write about.
    Please use emojis where appropriate as well!

    #noqa: DAR101
    """


@text.prompt
def compose_twitter_post(blog_post):
    """This is a blog post that I just wrote:

    {{ blog_post }}

    Please compose for me a Twitter post
    that entices my followers on Twitter to read it.
    Ensure that there is a call to action to interact with the post after reading it,
    such as retweeting, commenting, or sharing it with others,
    and to support my work on Patreon.
    My Patreon link is https://patreon.com/ericmjl/
    Include hashtags inline with the Twitter post.

    #noqa: DAR101
    """
