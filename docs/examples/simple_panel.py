# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "marimo",
#     "llamabot>=0.17.0",
#     "panel",
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
    # SimpleBot Apps
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    This notebook shows how to create a simple Panel app surrounding SimpleBot.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Firstly, our imports:
    """
    )
    return


@app.cell
def _():
    from llamabot import SimpleBot

    return (SimpleBot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Then we create the bot, in this case, a Feynman bot:
    """
    )
    return


@app.cell
def _(SimpleBot):
    feynman = SimpleBot(
        """
    You are Richard Feynman.
    You will be given a difficult concept, and your task is to explain it back.
    """
    )
    return (feynman,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Build the UI
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We'll build an app that lets others take in a chunk of text (an abstract) that the Feynman bot will re-explain back to us.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    For that, we'll need to start with a text area input:
    """
    )
    return


@app.cell
def _(feynman):
    app = feynman.panel(
        input_text_label="Abstract",
        output_text_label="Summary",
        site_name="Feynman Bot",
        title="Feynman Bot",
    )
    return (app,)


@app.cell
def _(app):
    app.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    To run this, execute the following command from the repo root:

    ```bash
    uvx marimo run --sandbox docs/examples/simple_panel.py
    ```
    """
    )
    return


if __name__ == "__main__":
    app.run()
