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


@app.cell
def _(mo):

    model_choice = mo.ui.dropdown(
        options={
            "Anthropic Sonnet 4.5": "anthropic/claude-sonnet-4-5-20250929",
            "LM Studio (OpenAI-compatible local endpoint)": "openai/qwen/qwen3.6-35b-a3b",
            "OpenAI (requires OPENAI_API_KEY)": "openai/gpt-4.1-mini",
            "Ollama local": "ollama_chat/qwen2.5:0.5b",
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
            mo.md(
                "Recommended default: **Anthropic Sonnet 4.5**. "
                "LM Studio is also available for local testing."
            ),
            model_choice,
            user_query,
            run_agent,
        ]
    )

    return model_choice, run_agent, user_query


@app.cell
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

    api_kwargs = {}
    selected_model = model_choice.value

    if selected_model == "anthropic/claude-sonnet-4-5-20250929":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            mo.stop(True, mo.md(":warning: Set ANTHROPIC_API_KEY for Sonnet 4.5."))
    elif selected_model == "openai/qwen/qwen3.6-35b-a3b":
        lmstudio_api_base = os.environ.get(
            "OPENAI_API_BASE", "http://localhost:1234/v1"
        )
        lmstudio_api_key = os.environ.get("OPENAI_API_KEY", "lm-studio")
        api_kwargs = {
            "api_base": lmstudio_api_base,
            "api_key": lmstudio_api_key,
        }
        mo.output.append(mo.md(f"Using LM Studio endpoint: `{lmstudio_api_base}`"))
    elif selected_model == "openai/gpt-4.1-mini":
        if not os.environ.get("OPENAI_API_KEY"):
            mo.stop(True, mo.md(":warning: Set OPENAI_API_KEY or switch models."))

    bot = AgentBot(
        tools=[local_uppercase],
        system_prompt=(
            "You are a careful assistant. Prefer tools when useful. "
            "MCP tools are namespaced (for example demo__greet)."
        ),
        model_name=selected_model,
        max_iterations=8,
        mcp_servers=[demo_mcp],
        **api_kwargs,
    )

    # LM Studio + qwen/qwen3.6-35b-a3b may not emit structured tool_calls when
    # ToolBot enforces tool_choice='required'. For this model, prefer auto.
    if selected_model == "openai/qwen/qwen3.6-35b-a3b":
        if hasattr(bot, "decide_node") and hasattr(bot.decide_node, "toolbot"):
            bot.decide_node.toolbot.tool_choice = "auto"

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
    ## Model and provider syntax cheatsheet

    ### Anthropic Sonnet 4.5 (recommended)

    - LiteLLM model id: `anthropic/claude-sonnet-4-5-20250929`
    - Required env var: `ANTHROPIC_API_KEY`
    - No custom `api_base` needed.

    ### LM Studio (OpenAI-compatible local)

    ```bash
    export OPENAI_API_BASE=http://localhost:1234/v1
    # Optional if your local gateway expects a key:
    export OPENAI_API_KEY=lm-studio
    ```

    Model string in this notebook: `openai/qwen/qwen3.6-35b-a3b`

    ### Stdio MCP server

    ```python
    docs_spec = MCPServerSpec(
        name="docs",
        transport="stdio",
        command="uvx",
        args=["--with", "llamabot[all]", "llamabot", "mcp", "launch"],
    )
    ```

    ### Remote HTTP / SSE MCP server

    ```python
    remote_spec = MCPServerSpec(
        name="remote",
        transport="http",  # or "sse"
        url="http://localhost:8765/mcp",
        headers={"Authorization": "Bearer TOKEN"},
    )
    ```

    Use `MCPStartupMode.BEST_EFFORT` if some servers may be offline.
    """
    )

    return


if __name__ == "__main__":
    app.run()
