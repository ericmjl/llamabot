# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "llamabot[all]==0.12.7",
#     "marimo",
#     "pyprojroot==0.3.0",
#     "rerankers==0.10.0",
# ]
#
# [tool.uv.sources]
# llamabot = { path = "../", editable = true }
# ///

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import llamabot as lmb
    from pathlib import Path
    from pyprojroot import here

    return here, lmb


@app.cell
def _(here):
    docs = (here() / "docs").rglob("*.md")
    docs = [fpath.read_text() for fpath in docs]
    return (docs,)


@app.cell
def _(docs, lmb):
    llamabot_docs = lmb.LanceDBDocStore(table_name="llamabot-docs")
    llamabot_docs.extend(docs)
    return (llamabot_docs,)


@app.cell
def _(llamabot_docs):
    llamabot_docs.retrieve("How does the git hooks CLI work?")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
