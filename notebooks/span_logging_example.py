# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "llamabot",
#     "marimo>=0.17.0",
#     "pyzmq",
# ]
#
# [tool.uv.sources]
# llamabot = { path = "../", editable = true }
#
# ///

import marimo

__generated_with = "0.18.3"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    # Span-Based Logging Example

    This notebook demonstrates LlamaBot's span-based logging feature, which provides
    structured tracing of LLM operations similar to pydantic-logfire.

    ## Features:
    - Automatic span creation for bot operations
    - Nested spans for complex workflows
    - Dictionary-like attribute management
    - Query and visualization of spans
    """
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo

    from llamabot import (
        SimpleBot,
        enable_span_recording,
        get_spans,
        span,
    )

    return SimpleBot, enable_span_recording, get_spans, mo, span


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Enable Span Recording
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    First, enable span recording globally. This will automatically create spans
    for all bot operations.
    """
    )
    return


@app.cell
def _():
    # Span recording is now enabled by default for all bots
    # Spans are automatically created - no need to call enable_span_recording()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Create a Simple Bot
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    Create a SimpleBot instance. Spans will be automatically created when you call it.
    """
    )
    return


@app.cell
def _(SimpleBot):
    bot = SimpleBot(
        "You are a helpful assistant.", model_name="ollama_chat/gemma3n:latest"
    )
    return (bot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Make a Bot Call
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    When you call the bot, spans are automatically created:
    """
    )
    return


@app.cell
def _(bot):
    for i in range(3):
        response = bot(f"What is 2+{i}?")
    return (response,)


@app.cell(hide_code=True)
def _(mo, response):
    mo.md(
        f"""
    **Response:** {response.content}
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Display Spans from Bot Instance

    Each bot instance tracks its own spans. When you display the bot object
    (as the last expression in a cell), it automatically shows the spans
    visualization from the most recent call:
    """
    )
    return


@app.cell
def _(bot):
    # Display the bot object - this automatically shows spans from the last call
    bot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Query Spans
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    Retrieve all spans from the database:
    """
    )
    return


@app.cell
def _(get_spans):
    # get_spans() returns a SpanList that displays all spans together
    all_spans = get_spans()
    # Display all spans in unified visualization
    all_spans
    return (all_spans,)


@app.cell(hide_code=True)
def _(all_spans, mo):
    mo.md(
        f"""
    Found {len(all_spans)} spans
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Manual Span Creation
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    You can also create spans manually using the `span()` context manager
    or decorator. Spans support dictionary-like attribute management.
    """
    )
    return


@app.cell
def _(span):
    with span("custom_operation", user_id=123, action="test") as s:
        s["status"] = "processing"
        s.log("step_completed", step=1)

        with s.span("nested_operation") as nested:
            nested["nested_data"] = "value"
            nested.log("nested_event")

        s["status"] = "completed"
        s.log("operation_finished")
    s
    return


@app.cell
def _(bot):
    bot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Query Spans by Attributes
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    Query spans by their attributes:
    """
    )
    return


@app.cell
def _(get_spans):
    filtered_spans = get_spans(user_id=123)
    filtered_spans
    return (filtered_spans,)


@app.cell(hide_code=True)
def _(filtered_spans, mo):
    mo.md(
        f"""
    Found {len(filtered_spans)} spans with user_id=123
    """
    )
    return


@app.cell
def _(get_spans):
    gemma_spans = get_spans(model="ollama_chat/gemma3n:latest")
    gemma_spans
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Span as Decorator
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    You can also use `span()` as a decorator:
    """
    )
    return


@app.cell
def _(span):
    @span("decorated_function", category="example")
    def my_function(x: int, y: int) -> int:
        return x + y

    return (my_function,)


@app.cell
def _(my_function):
    result = my_function(5, 3)
    return (result,)


@app.cell(hide_code=True)
def _(mo, result):
    mo.md(
        f"""
    **Result:** {result}
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Query Decorator Spans
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    Now let's query for spans created by the decorator:
    """
    )
    return


@app.cell
def _(get_spans):
    decorated_spans = get_spans(operation_name="decorated_function")
    return (decorated_spans,)


@app.cell
def _(decorated_spans):
    # Display all decorated spans together
    decorated_spans
    return


@app.cell
def _(span):
    # This time without any arguments
    @span
    def explodify(s: str, i: int) -> str:
        return f"{'!' * i} {s} {'!' * i}"

    explodify("Boom!", 3)
    return


@app.cell
def _(get_spans):
    get_spans(operation_name="explodify")
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Query Spans by Category Attribute
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    We can also query spans by the attributes we set in the decorator:
    """
    )
    return


@app.cell
def _(get_spans):
    category_spans = get_spans(category="example")
    return (category_spans,)


@app.cell
def _(category_spans):
    # Display all category spans together
    category_spans
    return


if __name__ == "__main__":
    app.run()
