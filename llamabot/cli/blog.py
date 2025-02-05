"""Blog Assistant CLI"""

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
from llamabot.prompt_library.sembr import sembr as sembr_prompt
from llamabot.prompt_library.sembr import sembr_bot

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
    typer.echo("Applying SEMBR to your summary...")
    bot = sembr_bot()
    summary_sembr = bot(sembr_prompt(parsed_response["summary"]))

    typer.echo("\n\n")
    typer.echo("Here is your blog summary:")
    typer.echo(summary_sembr)

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
    try:
        import pyperclip
    except ImportError:
        raise ImportError(
            "pyperclip is not installed. Please install it with `pip install llamabot[cli]`."
        )

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

    social_media_content = bot(compose_func(query)).content
    social_media_content = coerce_dict(social_media_content)

    if platform == "patreon":
        typer.echo("\n\n")
        for k, v in social_media_content.items():
            typer.echo("\n\n")
            typer.echo(f"Here is the patreon {k}:")
            typer.echo(v)

    else:
        pyperclip.copy(social_media_content["post_text"])
        typer.echo("\n\n")
        typer.echo(f"Your {platform} post has been copied to your clipboard! 🎉")


@app.command()
def sembr():
    """Apply semantic line breaks to a blog post."""
    try:
        import pyperclip
    except ImportError:
        raise ImportError(
            "pyperclip is not installed. Please install it with `pip install llamabot[cli]`."
        )

    bot = sembr_bot()
    typer.echo("Please paste your blog post below.")
    query = uniform_prompt()

    response = bot(sembr_prompt(query)).content
    pyperclip.copy(response)
    typer.echo("\n\n")
    typer.echo("Your sembr post has been copied to your clipboard! 🎉")
