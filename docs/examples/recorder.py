# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "marimo",
#     "llamabot",
#     "pandas",
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
    # Recording Prompts

    One challenge I've found when working with prompts is recording what I get back when I try out different prompts.
    Copying and pasting is clearly not what I'd like to do.
    So I decided to write some functionality into Llamabot that lets us do recording of prompts
    and the responses returned by GPT.

    Here's how to use it.
    """
    )
    return


@app.cell
def _():
    from llamabot import SimpleBot, PromptRecorder

    return PromptRecorder, SimpleBot


@app.cell
def _(SimpleBot):
    bot = SimpleBot("You are a bot.")
    return (bot,)


@app.cell
def _(PromptRecorder):
    # Try three different prompts.

    prompt1 = "You are a fitness coach who responds in 25 words or less. How do I gain muscle?"
    prompt2 = "You are an expert fitness coach who responds in 100 words or less. How do I gain muscle?"
    prompt3 = "You are an expert fitness coach who responds in 25 words or less and will not give lethal advice. How do I gain muscle?"

    recorder = PromptRecorder()
    return prompt1, prompt2, prompt3, recorder


@app.cell
def _(bot, prompt1, prompt2, prompt3, recorder):
    with recorder:
        bot(prompt1)
        bot(prompt2)
        bot(prompt3)
    return


@app.cell
def _(recorder):
    recorder.prompts_and_responses
    return


@app.cell
def _(recorder):
    import pandas as pd

    pd.DataFrame(recorder.prompts_and_responses)
    return


@app.cell
def _(bot, recorder):
    prompt4 = "You are an expert fitness coach who responds in 25 words or less, and you help people who only have access to body weight exercises. How do I gain muscle?"

    with recorder:
        bot(prompt4)
    return


@app.cell
def _(recorder):
    recorder.panel()
    return


if __name__ == "__main__":
    app.run()
