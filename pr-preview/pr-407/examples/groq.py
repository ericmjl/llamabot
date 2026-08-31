# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "marimo",
#     "llamabot",
#     "python-dotenv",
# ]
#
# [tool.uv.sources]
# llamabot = { path = "../../", editable = true }
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
    # Using LlamaBot with Groq

    Groq is a super, super fast API for LLMs.
    """
    )
    return


@app.cell
def _():
    from llamabot import SimpleBot
    from dotenv import load_dotenv

    load_dotenv()

    groq_bot = SimpleBot(
        system_prompt="You are a cheerful llama.",
        model_name="groq/mixtral-8x7b-32768",
    )
    return SimpleBot, groq_bot


@app.cell
def _(SimpleBot):
    openai_bot = SimpleBot(
        system_prompt="You are a cheerful llama.", model_name="gpt-4-turbo"
    )
    return (openai_bot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In my own testing, Groq's mixtral implementation is ~3-4x faster than OpenAI's GPT-4 turbo model.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    As of 2024-07-20, with LiteLLM 1.35.38 (which is the current version that LlamaBot is pinned to),
    Groq does not support streaming with JSON mode via LiteLLM.
    [This GitHub issue](https://github.com/BerriAI/litellm/issues/4804) has been filed in response.
    """
    )
    return


if __name__ == "__main__":
    app.run()
