# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "llamabot[all]==0.12.11",
#     "marimo",
# ]
#
# [tool.uv.sources]
# llamabot = { path = "../../", editable = true }
# ///

import marimo

__generated_with = "0.14.12"
app = marimo.App(width="full")


@app.cell
def _():
    import llamabot as lmb

    bot = lmb.SimpleBot(
        "You are a helpful bot.",
        # model_name="ollama_chat/phi4:latest",
        memory=lmb.ChatMemory.threaded(model="gpt-4o-mini"),
    )
    bot("Help me write a program to teach CI/CD.")
    return (bot,)


@app.cell
def _(bot):
    bot("Now help me explain the ci.yaml file.")
    return


@app.cell
def _(bot):
    bot("Now help me explain the .py file instead")
    return


@app.cell
def _(bot):
    bot("And now can you explain the overall structure of the example?")
    return


@app.cell
def _(bot):
    bot("I don't think I fully understand why a test is necessary, can you explain?")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""As we're chatting with the bot, we see that the underlying chat graph ends up reflecting the structure of the chat as well."""
    )
    return


@app.cell
def _(bot):
    import marimo as mo

    mo.mermaid(bot.memory.to_mermaid())
    return (mo,)


if __name__ == "__main__":
    app.run()
