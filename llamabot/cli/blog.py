"""Blog Assistant CLI"""

import pyperclip
import typer

from llamabot.prompt_library.blog import (
    BlogInformation,
    blog_title_tags_summary,
    blogging_bot,
    compose_linkedin_post,
    compose_patreon_post,
    compose_twitter_post,
)
from llamabot.prompt_library.output_formatter import coerce_dict

from .utils import uniform_prompt

app = typer.Typer()


@app.command()
def summarize():
    """My standard blogging workflow."""
    bot = blogging_bot()

    typer.echo("Please paste your blog post below.")
    query = uniform_prompt()
    response = bot(blog_title_tags_summary(query, BlogInformation)).content
    parsed_response = coerce_dict(response)

    typer.echo("\n\n")
    typer.echo("Here is your blog title:")
    typer.echo(parsed_response["title"])

    typer.echo("\n\n")
    typer.echo("Here is your blog summary:")
    typer.echo(parsed_response["summary"])

    typer.echo("\n\n")
    typer.echo("Here are your blog tags:")
    for tag in parsed_response["tags"]:
        typer.echo(tag)


@app.command()
def social_media(platform: str):
    """Generate social media posts.

    :param platform: The social media platform to generate posts for.
        Should be one of "linkedin", "patreon", or "twitter".
    """
    bot = blogging_bot()
    typer.echo("Please paste your blog post below.")
    query = uniform_prompt()

    platform = platform.lower()

    platform_to_compose_mapping = {
        "linkedin": compose_linkedin_post,
        "patreon": compose_patreon_post,
        "twitter": compose_twitter_post,
    }

    compose_func = platform_to_compose_mapping[platform]

    patreon_post = bot(compose_func(query)).content
    pyperclip.copy(patreon_post)
    typer.echo("\n\n")
    typer.echo(f"Your {platform} post has been copied to your clipboard! 🎉")
