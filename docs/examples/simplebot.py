# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "marimo",
#     "llamabot>=0.17.0",
#     "python-dotenv",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # LLaMaBot's SimpleBot in under 5 minutes
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Let's say we have the text of a blog...
    """
    )
    return


@app.cell
def _():
    with open("../../data/blog_text.txt", "r+") as f:
        blog_text = f.read()
    blog_text[0:100] + "..."
    return (blog_text,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    And we'd like to create a function that takes in the text and gives us a draft LinkedIn post,
    complete with emojis,
    that is designed to entice others to read the blog post.
    LLaMaBot's `SimpleBot` lets us build that function easily.
    """
    )
    return


@app.cell
def _():
    from llamabot import SimpleBot

    system_prompt = """You are a LinkedIn post generator bot.
    A human will give you the text of a blog post that they've authored,
    and you will compose a LinkedIn post that advertises it.
    The post is intended to hook a reader into reading the blog post.
    The LinkedIn post should be written with one line per sentence.
    Each sentence should begin with an emoji appropriate to that sentence.
    The post should be written in professional English and in first-person tone for the human.
    """

    linkedin = SimpleBot(
        system_prompt=system_prompt,
        stream_target="stdout",  # this is the default!,
        model_name="gpt-4-0125-preview",
    )
    return SimpleBot, linkedin, system_prompt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Note that SimpleBot by default will always stream.
    All that you need to configure is where you want to stream to.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    With `linkedin`, we can now pass in the blog text and - voila! - get back a draft LinkedIn post.
    """
    )
    return


@app.cell
def _(blog_text, linkedin):
    linkedin_post = linkedin(blog_text)
    return (linkedin_post,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Now, you can edit it to your hearts content! :-)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Next up, we have streaming that is compatible with Panel's Chat interface,
    which expects the text to be returned in its entirety as it is being built up.
    """
    )
    return


@app.cell
def _(SimpleBot, system_prompt):
    linkedin_panel = SimpleBot(
        system_prompt=system_prompt,
        stream_target="panel",
    )
    return (linkedin_panel,)


@app.cell
def _(blog_text, linkedin_panel):
    linkedin_post_1 = linkedin_panel(blog_text)
    return (linkedin_post_1,)


@app.cell
def _(linkedin_post_1):
    for _post in linkedin_post_1:
        print(_post)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    And finally, we have streaming via the API. We return a generator that yields individual parts of text as they are being generated.
    """
    )
    return


@app.cell
def _(SimpleBot, blog_text, system_prompt):
    linkedin_api = SimpleBot(system_prompt=system_prompt, stream_target="api")
    linkedin_post_2 = linkedin_api(blog_text)
    for _post in linkedin_post_2:
        print(_post, end="")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    If you have an Ollama server running, you can hit the API using SimpleBot.
    The pre-requisite is that you have already run `ollama pull <modelname>`
    to download the model to the Ollama server.
    """
    )
    return


@app.cell
def _(system_prompt):
    print(system_prompt)
    return


@app.cell
def _(SimpleBot, blog_text, system_prompt):
    import os
    from dotenv import load_dotenv

    load_dotenv()
    linkedin_ollama = SimpleBot(
        model_name="ollama/mistral",
        system_prompt=system_prompt,
        stream_target="stdout",
        api_base=f"http://{os.getenv('OLLAMA_SERVER')}:11434",
    )
    linkedin_post_3 = linkedin_ollama(
        blog_text
    )  # Specifying Ollama via the model_name argument is necessary!s  # this is the default!
    return (linkedin_post_3,)


if __name__ == "__main__":
    app.run()
