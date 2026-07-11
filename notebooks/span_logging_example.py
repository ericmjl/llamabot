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

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def title(mo):
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
def imports():
    import time

    import marimo as mo

    from llamabot import (
        SimpleBot,
        get_spans,
        span,
    )

    return SimpleBot, get_spans, mo, span, time


@app.cell(hide_code=True)
def create_bot_header(mo):
    mo.md(
        """
    ## Create a Simple Bot
    """
    )
    return


@app.cell(hide_code=True)
def create_bot_description(mo):
    mo.md(
        """
    Create a SimpleBot instance. Spans will be automatically created when you call it.
    """
    )
    return


@app.cell
def create_bot(SimpleBot):
    bot = SimpleBot("You are a helpful assistant.", model_name="ollama_chat/gemma4:12b")
    return (bot,)


@app.cell(hide_code=True)
def bot_call_header(mo):
    mo.md(
        """
    ## Make a Bot Call
    """
    )
    return


@app.cell(hide_code=True)
def bot_call_description(mo):
    mo.md(
        """
    When you call the bot, spans are automatically created:
    """
    )
    return


@app.cell
def make_bot_calls(bot):
    for i in range(3):
        response = bot(f"What is 2+{i}?")
    return (response,)


@app.cell(hide_code=True)
def bot_response_display(mo, response):
    mo.md(
        f"""
    **Response:** {response.content}
    """
    )
    return


@app.cell
def spans_from_bot_header(mo):
    mo.md(
        """
    ## Display Spans from Bot Instance

    Each bot instance tracks its own spans. You can access them via the `.spans` property,
    which returns a `SpanList` that displays all spans from the bot's calls.

    Note: Displaying the bot object directly shows the bot's configuration, not spans.
    Use `.spans` to view the span visualization.
    """
    )
    return


@app.cell
def display_bot_spans(bot):
    # Access spans via the .spans property (unified interface for all bots)
    bot.spans
    return


@app.cell
def display_bot_config(bot):
    # Display the bot object directly - shows configuration (not spans)
    bot
    return


@app.cell(hide_code=True)
def query_spans_header(mo):
    mo.md(
        """
    ## Query Spans
    """
    )
    return


@app.cell(hide_code=True)
def query_spans_description(mo):
    mo.md(
        """
    Retrieve all spans from the database:
    """
    )
    return


@app.cell
def query_all_spans(get_spans):
    # get_spans() returns a SpanList that displays all spans together
    all_spans = get_spans()
    # Display all spans in unified visualization
    all_spans
    return (all_spans,)


@app.cell(hide_code=True)
def all_spans_count(all_spans, mo):
    mo.md(
        f"""
    Found {len(all_spans)} spans
    """
    )
    return


@app.cell(hide_code=True)
def manual_span_header(mo):
    mo.md(
        """
    ## Manual Span Creation
    """
    )
    return


@app.cell(hide_code=True)
def manual_span_description(mo):
    mo.md(
        """
    You can also create spans manually using the `span()` context manager
    or decorator. Spans support dictionary-like attribute management.
    """
    )
    return


@app.cell
def create_manual_span(span):
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
def display_spans_after_manual(bot):
    # Display spans using the .spans property
    bot.spans
    return


@app.cell(hide_code=True)
def query_by_attr_header(mo):
    mo.md(
        """
    ## Query Spans by Attributes
    """
    )
    return


@app.cell(hide_code=True)
def query_by_attr_description(mo):
    mo.md(
        """
    Query spans by their attributes:
    """
    )
    return


@app.cell
def query_by_user_id(get_spans):
    filtered_spans = get_spans(user_id=123)
    filtered_spans
    return (filtered_spans,)


@app.cell(hide_code=True)
def filtered_spans_count(filtered_spans, mo):
    mo.md(
        f"""
    Found {len(filtered_spans)} spans with user_id=123
    """
    )
    return


@app.cell
def query_by_model(get_spans):
    gemma_spans = get_spans(model="ollama_chat/gemma4:12b")
    gemma_spans
    return


@app.cell(hide_code=True)
def decorator_header(mo):
    mo.md(
        """
    ## Span as Decorator
    """
    )
    return


@app.cell(hide_code=True)
def decorator_description(mo):
    mo.md(
        """
    You can also use `span()` as a decorator:
    """
    )
    return


@app.cell
def create_decorated_function(span):
    @span("decorated_function", category="example")
    def my_function(x: int, y: int) -> int:
        return x + y

    return (my_function,)


@app.cell
def call_decorated_function(my_function):
    result = my_function(5, 3)
    return (result,)


@app.cell(hide_code=True)
def decorated_result_display(mo, result):
    mo.md(
        f"""
    **Result:** {result}
    """
    )
    return


@app.cell(hide_code=True)
def query_decorator_header(mo):
    mo.md(
        """
    ## Query Decorator Spans
    """
    )
    return


@app.cell(hide_code=True)
def query_decorator_description(mo):
    mo.md(
        """
    Now let's query for spans created by the decorator:
    """
    )
    return


@app.cell
def query_decorated_spans(get_spans):
    decorated_spans = get_spans(operation_name="decorated_function")
    return (decorated_spans,)


@app.cell
def display_decorated_spans(decorated_spans):
    # Display all decorated spans together
    decorated_spans
    return


@app.cell
def create_explodify(span):
    # This time without any arguments
    @span
    def explodify(s: str, i: int) -> str:
        return f"{'!' * i} {s} {'!' * i}"

    explodify("Boom!", 3)
    return


@app.cell
def query_explodify_spans(get_spans):
    get_spans(operation_name="explodify")
    return


@app.cell(hide_code=True)
def category_header(mo):
    mo.md(
        """
    ## Query Spans by Category Attribute
    """
    )
    return


@app.cell(hide_code=True)
def category_description(mo):
    mo.md(
        """
    We can also query spans by the attributes we set in the decorator:
    """
    )
    return


@app.cell
def query_by_category(get_spans):
    category_spans = get_spans(category="example")
    return (category_spans,)


@app.cell
def display_category_spans(category_spans):
    # Display all category spans together
    category_spans
    return


@app.cell(hide_code=True)
def waterfall_demo(span, time, get_spans):
    # Waterfall timing demo — nested spans with simultaneous starts.
    #
    # Builds a realistic agent-trace shape using *manual* spans so the waterfall
    # can be verified without an LLM backend. The comment-diagram shows the
    # **elapsed-time** layout (what the waterfall x-axis represents):
    #
    #   agentbot_call (t=0.0 → ~1.6s)
    #   ├── decision          (t≈0 → ~0.3s)    ← starts nearly same time as root
    #   ├── llm_request       (t≈0.3 → ~1.3s)
    #   │   └── tool_call     (t≈0.4 → ~0.9s)  ← NESTED under llm_request
    #   └── decision          (t≈1.3 → ~1.5s)
    #
    # Key things to check in the Waterfall view:
    #   * Bars that start at the same time line up vertically (indent is on the
    #     label, not the bar).
    #   * The x-axis shows elapsed seconds from the trace origin.
    #   * Nesting is shown by label indentation + tree structure, NOT by
    #     shifting bars right.

    demo_tag = f"wf_demo_{int(time.time() * 1000)}"

    with span("agentbot_call", demo_tag=demo_tag) as root:
        root["prompt"] = "What is the meaning of life?"
        time.sleep(0.05)

        with root.span("decision", demo_tag=demo_tag):
            time.sleep(0.3)

        with root.span("llm_request", demo_tag=demo_tag) as llm_span:
            time.sleep(0.1)
            with llm_span.span("tool_call", demo_tag=demo_tag):
                time.sleep(0.5)
            time.sleep(0.3)

        with root.span("decision", demo_tag=demo_tag):
            time.sleep(0.2)

        time.sleep(0.05)

    demo_spans = get_spans(demo_tag=demo_tag)
    demo_spans
    return


@app.cell(hide_code=True)
def multi_turn_demo(span, time, get_spans):
    # Multi-turn waterfall demo — two sequential "user inputs".
    #
    # Each turn is a separate root span. On the waterfall they appear at
    # their correct elapsed-time position: turn 1 starts at t≈0, turn 2
    # starts after turn 1 finishes. The x-axis is elapsed seconds from
    # the trace origin (= first user input).

    multi_tag = f"multi_{int(time.time() * 1000)}"

    # --- Turn 1 ---
    with span("user_input_1", demo_tag=multi_tag) as turn1:
        turn1["query"] = "What is 2+2?"
        time.sleep(0.05)
        with turn1.span("llm_response", demo_tag=multi_tag):
            time.sleep(0.4)
        time.sleep(0.05)

    # --- Turn 2 (happens after turn 1) ---
    with span("user_input_2", demo_tag=multi_tag) as turn2:
        turn2["query"] = "Now multiply that by 3."
        time.sleep(0.05)
        with turn2.span("llm_response", demo_tag=multi_tag):
            time.sleep(0.4)
        time.sleep(0.05)

    multi_spans = get_spans(demo_tag=multi_tag)
    multi_spans
    return


if __name__ == "__main__":
    app.run()
