# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "marimo",
#     "llamabot>=0.17.0",
#     "panel",
#     "python-dotenv",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    from dotenv import load_dotenv

    load_dotenv()
    return (mo,)


@app.cell
def _():
    from llamabot import SimpleBot

    return (SimpleBot,)


@app.cell
def _(SimpleBot):
    punbot = SimpleBot(
        """You are a bad pun generator.
    You will be given a root word around which to generate a pun.
    The pun should be related to the root word and contain at least one emoji.
    Generate only the pun and nothing else.
    """,
        stream_target="panel",
    )
    return (punbot,)


@app.cell
def _(mo, punbot):
    root_word = mo.ui.text(label="Root word")
    run_btn = mo.ui.run_button(label="Generate Pun")
    mo.vstack([root_word, run_btn])
    return root_word, run_btn


@app.cell
def _(mo, punbot, root_word, run_btn):
    if run_btn.value:
        response = punbot(root_word.value)
        text = "\n\n".join(f"## {r}" for r in response)
        out = mo.md(text)
    else:
        out = mo.md("_Click **Generate Pun**._")
    out
    return


if __name__ == "__main__":
    app.run()
