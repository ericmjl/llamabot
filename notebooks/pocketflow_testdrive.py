# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "anthropic==0.72.0",
#     "llamabot",
#     "pocketflow==0.0.3",
# ]
#
# [tool.uv.sources]
# llamabot = { path = "../", editable = true }
#
# ///

import marimo

__generated_with = "0.17.0"
app = marimo.App(width="columns")


@app.cell(column=0)
def _(mo):
    mo.md(r"""## PocketFlow Source""")
    return


@app.cell
def _():
    return


@app.cell(column=1)
def _():
    from pocketflow import Node, Flow
    import llamabot as lmb

    return Flow, Node, lmb


@app.cell
def _(Node, lmb):
    class SummarizeFile(Node):
        def prep(self, shared):
            return shared

        def exec(self, prep_res):
            if not prep_res:
                return "Empty file content"
            txt = prep_res["file_content"]
            prompt = f"Summarize this text in 10 words: {txt}"
            bot = lmb.SimpleBot(system_prompt="You are a helpful assistant.")
            summary = bot(prompt)  # might fail
            return summary.content

        def exec_fallback(self, prep_res, exc):
            # Provide a simple fallback instead of crashing
            return "There was an error processing your request."

        def post(self, shared, prep_res, exec_res):
            return exec_res

    return (SummarizeFile,)


@app.cell
def _(txt):
    shared_summary = dict(file_content=txt)
    return (shared_summary,)


@app.cell
def _(Flow, SummarizeFile, shared_summary):
    summarize = SummarizeFile()
    flow_summary = Flow(start=summarize)
    flow_summary.run(shared_summary)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""This isn't particularly interesting, but I am starting to settle on a pattern. In the flow of data, inputs are what are passed into `prep`, and then the 3-step cycle begins: `prep -> exec -> post`. In this simple example, I just pass strings in and out of each method. However, if a different (and consistent) data structure is passed between them, that can facilitate more complex program building (complex data structures) while still being easy to reason about (consistent)."""
    )
    return


@app.cell
def _():
    return


@app.cell
def _(Node, lmb):
    class Respond(Node):
        def exec(self, prep_res):
            bot = lmb.SimpleBot(system_prompt="You are a helpful assistant.")

            exec_res = bot(prep_res)
            return exec_res.content

        def post(shared, prep_res, exec_res):
            shared["response"] = exec_res

    return


@app.cell(hide_code=True)
def _():
    txt = """
    Pocket Flow
    A 100-line minimalist LLM framework for Agents, Task Decomposition, RAG, etc.

    Lightweight: Just the core graph abstraction in 100 lines. ZERO dependencies, and vendor lock-in.
    Expressive: Everything you love from larger frameworks—(Multi-)Agents, Workflow, RAG, and more.
    Agentic-Coding: Intuitive enough for AI agents to help humans build complex LLM applications.
    Pocket Flow – 100-line minimalist LLM framework
    Core Abstraction
    We model the LLM workflow as a Graph + Shared Store:

    Node handles simple (LLM) tasks.
    Flow connects nodes through Actions (labeled edges).
    Shared Store enables communication between nodes within flows.
    Batch nodes/flows allow for data-intensive tasks.
    Async nodes/flows allow waiting for asynchronous tasks.
    (Advanced) Parallel nodes/flows handle I/O-bound tasks.
    Pocket Flow – Core Abstraction
    Design Pattern
    From there, it’s easy to implement popular design patterns:

    Agent autonomously makes decisions.
    Workflow chains multiple tasks into pipelines.
    RAG integrates data retrieval with generation.
    Map Reduce splits data tasks into Map and Reduce steps.
    Structured Output formats outputs consistently.
    (Advanced) Multi-Agents coordinate multiple agents.
    Pocket Flow – Design Pattern
    Utility Function
    We do not provide built-in utilities. Instead, we offer examples—please implement your own:

    LLM Wrapper
    Viz and Debug
    Web Search
    Chunking
    Embedding
    Vector Databases
    Text-to-Speech
    Why not built-in?: I believe it’s a bad practice for vendor-specific APIs in a general framework:

    API Volatility: Frequent changes lead to heavy maintenance for hardcoded APIs.
    Flexibility: You may want to switch vendors, use fine-tuned models, or run them locally.
    Optimizations: Prompt caching, batching, and streaming are easier without vendor lock-in.
    Ready to build your Apps?
    Check out Agentic Coding Guidance, the fastest way to develop LLM projects with Pocket Flow!

    """
    return (txt,)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Two-Node Workflow Example

    This workflow demonstrates how data flows between two connected nodes in PocketFlow.

    **Node 1 - ExtractTopics**: Takes the input text and extracts 3-5 key topics using an LLM.

    **Node 2 - GenerateQuestions**: Receives the extracted topics and generates thought-provoking questions for each topic.

    The nodes are connected using the syntax `extract_topics - "default" >> generate_questions`, which means the output of the first node becomes the input to the second node through the "default" action.
    """
    )
    return


@app.cell
def _(Node, lmb):
    class ExtractTopics(Node):
        """First node: Extract key topics from input text"""

        def prep(self, shared):
            text_to_analyze = shared["txt"]
            return text_to_analyze

        def exec(self, prep_result):
            text_to_analyze = prep_result
            if not text_to_analyze:
                return "No content to analyze"

            prompt = f"Extract 3-5 key topics from this text. Return only the topics as a comma-separated list:\n\n{text_to_analyze}"
            bot = lmb.SimpleBot(
                system_prompt="You are a helpful assistant that extracts key topics.",
                model_name="ollama_chat/qwen3:30b",
            )
            response = bot(prompt)
            return response.content

        def post(self, shared, prep_result, exec_res):
            shared["topics"] = exec_res
            return "default"

    return (ExtractTopics,)


@app.cell
def _(Node, lmb):
    class GenerateQuestions(Node):
        """Second node: Generate questions based on topics"""

        def prep(self, shared):
            topics = shared["topics"]
            txt = shared["txt"]
            return topics, txt

        def exec(self, prep_result):
            topics, txt = prep_result

            if not topics:
                return "Cannot generate questions without valid topics"

            prompt = f"Given these topics: {topics}\n\nand the original text: {txt}\n\nGenerate 2 interesting questions for each topic."
            bot = lmb.SimpleBot(
                system_prompt="You are a helpful assistant that generates thought-provoking questions.",
                model_name="ollama_chat/qwen3:30b",
            )
            response = bot(prompt)
            return response.content

        def exec_fallback(self, prep_result, exc):
            return "Error generating questions"

        def post(self, shared, prep_result, exec_res):
            shared["questions"] = exec_res

    return (GenerateQuestions,)


@app.cell
def _(ExtractTopics, Flow, GenerateQuestions, txt):
    extract_topics = ExtractTopics()
    generate_questions = GenerateQuestions()

    extract_topics - "default" >> generate_questions

    shared_topics = dict(txt=txt)

    two_node_flow = Flow(start=extract_topics)
    result = two_node_flow.run(shared_topics)
    result
    return (two_node_flow,)


@app.cell
def _(mo, two_node_flow):
    # Visualize the two-node workflow
    mo.mermaid(flow_to_mermaid(two_node_flow))
    return


@app.cell(column=3, hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Agent Example

    Now we're going to try an agentic loop. I am going to get a ToolBot to do a decision-making step within the input.
    """
    )
    return


@app.cell
def _(lmb):
    from llamabot.components.pocketflow import flow_to_mermaid

    return (flow_to_mermaid,)


@app.cell
def _():
    import subprocess

    _result = subprocess.run("ls")
    return


@app.cell
def _(lmb):
    @lmb.tool
    def execute_shell_command(cmd: str):
        import subprocess

        result = subprocess.run(cmd)
        return result.stdout

    return


@app.cell
def _():
    return


@app.cell
def _(Node, lmb):
    from llamabot.components.tools import (
        respond_to_user,
        search_internet,
        today_date,
    )

    search_internet = lmb.tool(search_internet)

    tools = [respond_to_user, today_date]

    @lmb.prompt("system")
    def decision_bot_system_prompt(tool_names: list[str]):
        """Given the chat history, pick for me one or more tools to execute
        in order to satisfy the user's query.

        You will be given a list of tools to pick from:

        {{ tool_names }}

        Give me just the tool name to pick.
        Use the tools judiciously to help answer the user's query.
        """

    class Decide(Node):
        def prep(self, shared: dict):
            shared["memory"].append(shared["query"])
            return shared

        def exec(self, prep_result):
            bot = lmb.SimpleBot(
                system_prompt=decision_bot_system_prompt([f.__name__ for f in tools]),
                model_name="ollama_chat/qwen2.5:0.5b",
            )
            response = bot(*prep_result["memory"])

            return response.content

        def post(self, shared, prep_result, exec_result):
            shared["memory"].append(f"Chosen Tool: {exec_result}")
            return exec_result

    return Decide, today_date


@app.cell
def _(today_date):
    today_date()
    return


@app.cell
def _(Node, today_date):
    class TodayDate(Node):
        def prep(self, shared: dict):
            return shared

        def exec(self, prep_result):
            return today_date()

        def post(self, shared, prep_result, exec_result):
            shared["memory"].append(f"Today's date: {exec_result}")
            return "decide"

    return (TodayDate,)


@app.cell
def _(Node, lmb):
    class RespondToUser(Node):
        def prep(self, shared: dict):
            return shared

        def exec(self, prep_result):
            bot = lmb.SimpleBot(
                "You are a helpful assistant.",
                model_name="ollama_chat/qwen2.5:0.5b",
            )
            response = bot(*prep_result["memory"])
            return response.content

        def post(self, shared, prep_result, exec_result):
            shared["memory"].append(exec_result)
            return exec_result

    return (RespondToUser,)


@app.cell
def _(Decide, RespondToUser, TodayDate):
    # Set up the graph
    today__date = TodayDate()
    respond__to__user = RespondToUser()
    decide = Decide()

    shared = dict()
    shared["query"] = "What is the date today?"
    shared["memory"] = []

    decide - "today_date" >> today__date
    today__date - "decide" >> decide
    decide - "respond_to_user" >> respond__to__user
    return decide, shared


@app.cell
def _():
    return


@app.cell
def _(Flow, decide, shared):
    flow2 = Flow(start=decide)
    flow2.run(shared)
    return (flow2,)


@app.cell
def _(flow2, mo):
    # Visualize the agent flow
    mo.mermaid(flow_to_mermaid(flow2))
    return


@app.cell(column=4)
def _(mo):
    mo.md(r"""## Contrast to LlamaBot Agent""")
    return


@app.cell
def _(lmb):
    lmb.AgentBot.__call__
    return


@app.cell
def _(lmb):
    bot = lmb.AgentBot(
        system_prompt="You are a helpful assistant.",
        model_name="ollama_chat/qwen3:30b",
    )
    bot("What's today's date?")
    return


if __name__ == "__main__":
    app.run()
