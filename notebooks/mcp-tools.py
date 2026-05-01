# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "llamabot==0.18.1",
#     "fastmcp==3.2.4",
# ]
#
# [tool.uv.sources]
# llamabot = { path = "../", editable = true }
#
# ///

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():

    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):

    mo.md(
        r"""
    # MCP tools with LlamaBot AgentBot

    This tutorial shows **first-class MCP integration** in LlamaBot. You will:

    1. Define MCP tools on a FastMCP server.
    2. Register those tools with LlamaBot via `mcp_servers=`.
    3. Run an `AgentBot` that can use both Python tools and MCP tools.

    **Important:** always call `bot.close_mcp()` when done so sessions shut down cleanly.
    """
    )

    return


@app.cell
def _():
    import os

    from fastmcp import FastMCP

    from llamabot.bot.agentbot import AgentBot
    from llamabot.components.tools import tool
    from llamabot.mcp.manager import MCPClientManager

    @tool
    def local_uppercase(text: str) -> str:
        """Uppercase text with a local Python tool.

        :param text: Input text.
        :return: Uppercased text.
        """
        return text.upper()

    demo_mcp = FastMCP("llamabot_mcp_tutorial")

    @demo_mcp.tool()
    def greet(name: str) -> str:
        """Return a short greeting.

        :param name: Who to greet.
        :return: Greeting string.
        """
        return f"Hello, {name}! (from MCP tool greet)"

    @demo_mcp.tool()
    def add_numbers(a: int, b: int) -> int:
        """Add two integers.

        :param a: First summand.
        :param b: Second summand.
        :return: Sum.
        """
        return a + b

    return AgentBot, MCPClientManager, demo_mcp, local_uppercase, os


@app.cell
def _(MCPClientManager, demo_mcp, mo):
    mgr = MCPClientManager([demo_mcp])
    mgr.start()
    try:
        wrapped = mgr.llamabot_tools()
        names = [
            getattr(getattr(fn, "func", fn), "__name__", "<unknown>") for fn in wrapped
        ]
        bullets = "\n".join([f"- `{n}`" for n in names])
    finally:
        mgr.close()

    mo.md(
        "## Registered MCP tool names\n\n"
        "These names are what ToolBot/DecideNode routes on.\n\n"
        f"{bullets}"
    )

    return


@app.cell(hide_code=True)
def _(mo):
    model_choice = mo.ui.dropdown(
        options={
            "Anthropic Sonnet 4.5": "anthropic/claude-sonnet-4-5-20250929",
            "Anthropic Opus 4.5": "anthropic/claude-opus-4-5-20251101",
            "Anthropic Haiku 4.5 (latest)": "anthropic/claude-haiku-4-5-20251001",
        },
        value="Anthropic Sonnet 4.5",
        label="Model",
        full_width=True,
    )

    user_query = mo.ui.text_area(
        value=(
            "Use MCP tool demo__greet to greet Ada, then "
            "use local_uppercase on the phrase hello mcp."
        ),
        label="User query",
        full_width=True,
    )

    run_agent = mo.ui.run_button(label="Run AgentBot")

    mo.vstack(
        [
            mo.md("## Run an AgentBot with MCP + Python tools"),
            mo.md("Choose an Anthropic model option for this MCP tool-calling demo."),
            model_choice,
            user_query,
            run_agent,
        ]
    )

    return model_choice, run_agent, user_query


@app.cell(hide_code=True)
def _(
    AgentBot,
    demo_mcp,
    local_uppercase,
    mo,
    model_choice,
    os,
    run_agent,
    user_query,
):
    mo.stop(not run_agent.value, mo.md("Click **Run AgentBot** to execute one run."))

    selected_model = model_choice.value

    if not selected_model.startswith("anthropic/"):
        mo.stop(True, mo.md(":warning: Select an Anthropic model option."))

    if not os.environ.get("ANTHROPIC_API_KEY"):
        mo.stop(True, mo.md(":warning: Set ANTHROPIC_API_KEY for Anthropic models."))

    bot = AgentBot(
        tools=[local_uppercase],
        system_prompt=(
            "You are a careful assistant. Prefer tools when useful. "
            "MCP tools are namespaced (for example demo__greet)."
        ),
        model_name=selected_model,
        max_iterations=8,
        mcp_servers=[demo_mcp],
    )

    try:
        result = bot(user_query.value)
    finally:
        bot.close_mcp()

    mo.vstack(
        [
            mo.md("### Agent result"),
            mo.md(f"```\n{result}\n```") if isinstance(result, str) else result,
        ]
    )

    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Anthropic model IDs in this notebook (LiteLLM)

    Based on LiteLLM docs / release notes, the dropdown uses:

    - Sonnet 4.5: `anthropic/claude-sonnet-4-5-20250929`
    - Opus 4.5: `anthropic/claude-opus-4-5-20251101`
    - Haiku 4.5 (latest): `anthropic/claude-haiku-4-5-20251001`

    Set this environment variable before running:

    ```bash
    export ANTHROPIC_API_KEY=...
    ```
    """
    )

    return


if __name__ == "__main__":
    app.run()
