# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "marimo",
#     "llamabot[rag,cli]>=0.17.0",
#     "chromadb",
#     "python-slugify",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from llamabot import QueryBot
    import git

    return QueryBot, git


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Eric Ma Q&A

    This shows how to build a blog Q&A bot using the text contents of Eric Ma's blog.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Setup: Download blog data
    """
    )
    return


@app.cell
def _(git):
    import tempfile
    from pathlib import Path

    # Create a temporary directory
    temp_dir = tempfile.TemporaryDirectory(dir="/tmp")

    repo_url = "https://github.com/duckdb/duckdb-web"
    # Clone the repository into the temporary directory
    git.Repo.clone_from(repo_url, temp_dir.name)

    # Set the root directory to the cloned repository
    root_dir = Path(temp_dir.name)
    return Path, repo_url, root_dir


@app.cell
def _(Path, repo_url):
    from slugify import slugify
    import chromadb

    client = chromadb.PersistentClient(
        path=str(Path.home() / ".llamabot" / "chroma.db")
    )
    collection = client.create_collection(slugify(repo_url), get_or_create=True)

    collection.get()
    return (slugify,)


@app.cell
def _(root_dir):
    source_file_extensions = [
        "py",
        "jl",
        "R",
        "ipynb",
        "md",
        "tex",
        "txt",
        "lr",
        "rst",
    ]

    source_files = []
    for extension in source_file_extensions:
        files = list(root_dir.rglob(f"*.{extension}"))
        print(f"Found {len(files)} files with extension {extension}.")
        source_files.extend(files)
    return (source_files,)


@app.cell
def _(QueryBot, repo_url, slugify, source_files):
    bot = QueryBot(
        system_prompt="You are an expert in the code repository given to you.",
        collection_name=slugify(repo_url),
        document_paths=source_files,
    )
    return (bot,)


@app.cell
def _(bot):
    bot("Give me an example of lambda functions in DuckDB.")
    return


@app.cell
def _(bot):
    bot("What is your view on building a digital portfolio?")
    return


@app.cell
def _(bot):
    bot("What were your experiences with the SciPy conference?")
    return


@app.cell
def _(bot):
    bot("What tutorials did you attend at the SciPy conference in 2023?")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## LlamaBot Code Query
    """
    )
    return


@app.cell
def _(QueryBot):
    import pathlib

    import llamabot
    from llamabot.file_finder import recursive_find

    pkg_root = pathlib.Path(llamabot.__file__).resolve().parent
    source_python_files = recursive_find(root_dir=pkg_root, file_extension=".py")

    codebot = QueryBot(
        "You are an expert in code Q&A.",
        collection_name="llamabot",
        document_paths=source_python_files,
        model_name="gpt-4-1106-preview",
    )
    return codebot, source_python_files


@app.cell
def _(codebot):
    codebot("How do I find all the files in a directory?")
    return


@app.cell
def _(codebot):
    codebot("Which Bot do I use to chat with my documents?")
    return


@app.cell
def _(codebot):
    codebot("Explain to me the architecture of SimpleBot.")
    return


@app.cell
def _(codebot):
    codebot("What are the CLI functions available in LlamaBot?")
    return


@app.cell
def _(source_python_files):
    from llamabot.bot.qabot import DocQABot

    codebot_1 = DocQABot(collection_name="llamabot")
    codebot_1.add_documents(document_paths=source_python_files)
    return (codebot_1,)


@app.cell
def _(codebot_1):
    codebot_1(
        "Does LlamaBot provide a function to find all files recursively in a directory?"
    )
    return


if __name__ == "__main__":
    app.run()
