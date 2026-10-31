# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "marimo",
#     "llamabot[all,cli,rag,turbovec]==0.19.0",
#     "matplotlib",
# ]
#
# [tool.uv.sources]
# llamabot = { path = "../../", editable = true }
# ///

import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
# TurboVec Document Store Demo

This notebook demonstrates how to use **TurboVecDocStore** — a vector database
backed by [turbovec](https://github.com/RyanCodrai/turbovec), which uses
Google Research's TurboQuant algorithm for compressed ANN search.

TurboVec is a good choice when **memory efficiency** and **search speed** matter
more than hybrid (vector + full-text) search. For hybrid search, use
`LanceDBDocStore` instead.
"""
    )
    return


@app.cell
def _():
    import marimo as mo
    import tempfile
    from pathlib import Path

    temp_dir = Path(tempfile.mkdtemp())
    return Path, mo, temp_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
## Creating a TurboVecDocStore

The constructor requires a `table_name` and optionally takes a `storage_path`,
`embedding_model`, and `bit_width` (2 or 4, default 4).
"""
    )
    return


@app.cell
def _(temp_dir):
    from llamabot import TurboVecDocStore

    store = TurboVecDocStore(table_name="demo-documents", storage_path=temp_dir)
    store.reset()
    return (store,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
## Adding Documents

Use `append` for single documents or `extend` for batches.
Duplicates are automatically skipped.
"""
    )
    return


@app.cell
def _(store):
    store.append(
        "Python is a programming language with clean syntax and a rich ecosystem."
    )
    store.append(
        "Rust is a systems programming language focused on safety and performance."
    )
    store.append("FastAPI is a modern web framework for building APIs with Python.")
    store.append(
        "Marimo is a reactive notebook for Python that makes data exploration interactive."
    )
    store.append(
        "LanceDB is an open-source vector database built on the Lance columnar format."
    )
    return


@app.cell
def _(store):
    documents = [
        "Vector databases store high-dimensional embeddings for similarity search.",
        "RAG (Retrieval-Augmented Generation) combines search with language models.",
        "Sentence transformers produce dense vector representations of text.",
    ]
    store.extend(documents)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
## Querying the Store

`retrieve` embeds the query and returns the most similar documents.
"""
    )
    return


@app.cell
def _(mo, store):
    query = mo.ui.text(
        value="What programming languages are available?", label="Search query"
    )
    query
    return (query,)


@app.cell
def _(query, store):
    results = store.retrieve(query.value, n_results=3)
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc}\n")
    return (results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
## Deduplication

Adding the same document twice is a no-op.
"""
    )
    return


@app.cell
def _(store):
    before = len(store.existing_records)
    store.append(
        "Python is a programming language with clean syntax and a rich ecosystem."
    )
    after = len(store.existing_records)
    before, after
    return after, before


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
## Using TurboVecDocStore with QueryBot

Because `TurboVecDocStore` implements the same `AbstractDocumentStore`
interface, you can pass it directly to `QueryBot` as the `docstore` parameter.
"""
    )
    return


@app.cell
def _(store):
    from llamabot import QueryBot

    bot = QueryBot(
        system_prompt="You are a helpful assistant that answers questions about the provided documents.",
        docstore=store,
        model_name="gpt-4o-mini",
    )
    return (bot,)


@app.cell
def _(bot):
    bot("What is Marimo?")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
## Persistence

The index and documents are automatically saved to disk.
Re-opening the same `storage_path` and `table_name` reloads everything.
"""
    )
    return


@app.cell
def _(store, temp_dir):
    print(f"Index:  {store.index_path}")
    print(f"Docs:   {store.docs_path}")
    print(f"Records in store: {len(store.existing_records)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
## Cleanup
"""
    )
    return


@app.cell
def _(store):
    store.reset()
    print("Store reset complete.")
    return


if __name__ == "__main__":
    app.run()
