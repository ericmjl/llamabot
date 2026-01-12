# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "anthropic==0.52.2",
#     "lancedb==0.23.0",
#     "llamabot[all]==0.12.6",
#     "marimo",
#     "pyprojroot==0.3.0",
# ]
#
# [tool.uv.sources]
# llamabot = { path = "../../", editable = true }
#
# ///

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium", layout_file="layouts/llamabot_docs.grid.json")


@app.cell
def _(mo):
    mo.md(
        r"""
    # Building a Chat Interface with LlamaBot and Marimo

    This notebook demonstrates how to create an interactive chat interface using LlamaBot and Marimo. We'll walk through the process of setting up a document store, creating a QueryBot, and building a reactive chat UI.
    """
    )
    return


@app.cell
def _():
    import llamabot as lmb

    return (lmb,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Document Store Setup

    We begin by creating a LanceDB document store to hold our documentation and source code. This allows our chatbot to search through relevant information when answering questions.

    """
    )
    return


@app.cell
def _(lmb):
    ds = lmb.components.docstore.LanceDBDocStore(
        table_name="llamabot-docs",
    )
    return (ds,)


@app.cell
def _(ds):
    ds.reset()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Loading Documents

    Next, we load two types of documents into our store:

    1. Markdown documentation files from the docs directory
    2. Python source code files from the llamabot package

    These documents provide the knowledge base for our chatbot to reference.
    """
    )
    return


@app.cell
def _(ds):
    from pyprojroot import here

    docs_paths = (here() / "docs").rglob("*.md")

    docs_texts = [p.read_text() for p in docs_paths]

    source_files = (here() / "llamabot").rglob("*.py")
    source_texts = [p.read_text() for p in source_files]

    ds.extend(docs_texts)
    ds.extend(source_texts)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Chat Memory Configuration

    We set up a separate document store specifically for chat history. This allows our bot to maintain context across conversation turns and provide more coherent responses.

    """
    )
    return


@app.cell
def _(lmb):
    # For simple linear memory (fast, no LLM calls)
    chat_memory = lmb.ChatMemory()

    # For intelligent threading (uses LLM for smart connections)
    # chat_memory = lmb.ChatMemory.threaded(model="gpt-4o-mini")

    return (chat_memory,)


@app.cell
def _(chat_memory):
    chat_memory.reset()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## QueryBot Creation

    The QueryBot is configured with:

    - A system prompt that defines its personality and purpose
    - The document store containing LlamaBot documentation
    - The chat memory store for conversation history

    Our system prompt instructs the bot to be helpful, kind, and cheerful when answering questions about LlamaBot.

    """
    )
    return


@app.cell
def _(chat_memory, ds, lmb):
    @lmb.prompt("system")
    def llamabot_docs_sysprompt():
        """You are a helpful assistant w.r.t. llamabot.
        Your task is to help users answer questions about LlamaBot.
        Respond kindly, be helpful and gentle, and always be cheerful."""

    qb = lmb.QueryBot(
        system_prompt=llamabot_docs_sysprompt(),
        docstore=ds,
        memory=chat_memory,
    )
    return (qb,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Marimo Chat Interface

    Finally, we create a reactive chat interface using Marimo's UI components. The `mo.ui.chat` component connects to our QueryBot through the `echo_model` function, which processes each user message and returns the bot's response.

    The interface includes sample prompts to help users get started, and the reactive nature of Marimo ensures that the UI updates automatically as the conversation progresses.

    """
    )
    return


@app.cell
def _(qb):
    import marimo as mo

    def echo_model(messages, config):
        response = qb(messages[-1].content)
        return response.content

    chat = mo.ui.chat(echo_model, prompts=["Hello", "How are you?"], max_height=600)
    chat
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## How to Use This Chat Interface

    Simply type your questions about LlamaBot in the chat interface. The QueryBot will:

    1. Search the document store for relevant information
    2. Generate a helpful response based on the documentation
    3. Maintain conversation context using the chat memory

    Try asking about LlamaBot's features, components, or usage patterns!


    """
    )
    return


if __name__ == "__main__":
    app.run()
