"""Blog Assistant CLI"""
import typer

from llamabot.prompt_library.blog import blog_tagger_and_summarizer, blogging_bot
from llamabot.prompt_library.output_formatter import coerce_dict

from .utils import uniform_prompt

app = typer.Typer()


@app.command()
def summarize_and_tag():
    """Summarize and tag a blog post."""
    bot = blogging_bot()

    typer.echo("Please paste your blog post below.")
    query = uniform_prompt()
    response = bot(blog_tagger_and_summarizer(query)).content
    parsed_response = coerce_dict(response)

    typer.echo("Here is your blog summary:")
    typer.echo(parsed_response["summary"])

    typer.echo("Here are your blog tags:")
    for tag in parsed_response["tags"]:
        typer.echo(tag)
