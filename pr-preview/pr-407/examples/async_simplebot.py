# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "marimo",
#     "llamabot",
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
    # `AsyncSimpleBot`: asynchronous streaming

    **`SimpleBot`** does a blocking chat completion: you call it, Python waits until the full answer is back.

    **`AsyncSimpleBot`** is for **async** code: you **`async for`** over token-sized **chunks** as they arrive, or **`await`** the bot for one full reply. That fits notebooks with async cells, FastAPI, and anything driven by an event loop.

    Below, the *asynchronous* part is the **`async for`** loop (and the **`async def`** cell Marimo runs with `await`). You need a running model (for example **Ollama** with the model pulled locally).
    """
    )
    return


@app.cell
def _():
    from llamabot import AsyncSimpleBot

    bot = AsyncSimpleBot(
        system_prompt="You are a concise assistant.",
        model_name="ollama/phi3",
        stream_target="none",
    )
    return (bot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### The asynchronous heart: `async for`

    The next cell is an **`async def`**. Inside it, **`async for chunk in bot.stream_async(...)`** drives an **async iterator**: each `chunk` is a text delta from the model. That is what makes this path non-blocking at the language level (your runtime can await I/O between chunks).

    Contrast: **`SimpleBot("...")`** blocks until the full string is ready; there is no `async for` over tokens.
    """
    )
    return


@app.cell
async def _(bot, mo):
    pieces: list[str] = []
    async for chunk in bot.stream_async("Say hello in exactly five words."):
        pieces.append(chunk)
    transcript = "".join(pieces)
    mo.md(f"**Streamed text:** {transcript}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### One-shot async completion: `await bot(...)`

    You can also **`await`** the bot. You get a full **`AIMessage`** when the model finishes, instead of iterating chunks yourself.
    """
    )
    return


@app.cell
async def _(bot):
    reply = await bot("Reply with a single word: async or sync?")
    reply.content
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Sync vs async (cheat sheet)

    | | **`SimpleBot`** | **`AsyncSimpleBot`** |
    |---|-----------------|----------------------|
    | Call style | `bot("hi")` — blocks | `await bot("hi")` — async |
    | Streaming | `stream_target`, sync generators | `async for` over `stream_async()` |
    | Typical use | scripts, REPL | FastAPI, WebSockets, async notebooks |

    For SSE over HTTP, see **`llamabot.sse.sse_stream`** and the [SSE example](sse_streaming.py). For the full contract, see the docs page on [async streaming](../reference/streaming_async.md).
    """
    )
    return


if __name__ == "__main__":
    app.run()
