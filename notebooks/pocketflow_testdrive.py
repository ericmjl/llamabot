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
    import asyncio, warnings, copy, time

    class BaseNode:
        def __init__(self):
            self.params, self.successors = {}, {}

        def set_params(self, params):
            self.params = params

        def next(self, node, action="default"):
            if action in self.successors:
                warnings.warn(f"Overwriting successor for action '{action}'")
            self.successors[action] = node
            return node

        def prep(self, shared):
            pass

        def exec(self, prep_res):
            pass

        def post(self, shared, prep_res, exec_res):
            pass

        def _exec(self, prep_res):
            return self.exec(prep_res)

        def _run(self, shared):
            p = self.prep(shared)
            e = self._exec(p)
            return self.post(shared, p, e)

        def run(self, shared):
            if self.successors:
                warnings.warn("Node won't run successors. Use Flow.")
            return self._run(shared)

        def __rshift__(self, other):
            return self.next(other)

        def __sub__(self, action):
            if isinstance(action, str):
                return _ConditionalTransition(self, action)
            raise TypeError("Action must be a string")

    class _ConditionalTransition:
        def __init__(self, src, action):
            self.src, self.action = src, action

        def __rshift__(self, tgt):
            return self.src.next(tgt, self.action)

    class Node(BaseNode):
        def __init__(self, max_retries=1, wait=0):
            super().__init__()
            self.max_retries, self.wait = max_retries, wait

        def exec_fallback(self, prep_res, exc):
            raise exc

        def _exec(self, prep_res):
            for self.cur_retry in range(self.max_retries):
                try:
                    return self.exec(prep_res)
                except Exception as e:
                    if self.cur_retry == self.max_retries - 1:
                        return self.exec_fallback(prep_res, e)
                    if self.wait > 0:
                        time.sleep(self.wait)

    class BatchNode(Node):
        def _exec(self, items):
            return [super(BatchNode, self)._exec(i) for i in (items or [])]

    class Flow(BaseNode):
        def __init__(self, start=None):
            super().__init__()
            self.start_node = start

        def start(self, start):
            self.start_node = start
            return start

        def get_next_node(self, curr, action):
            nxt = curr.successors.get(action or "default")
            if not nxt and curr.successors:
                warnings.warn(
                    f"Flow ends: '{action}' not found in {list(curr.successors)}"
                )
            return nxt

        def _orch(self, shared, params=None):
            curr, p, last_action = (
                copy.copy(self.start_node),
                (params or {**self.params}),
                None,
            )
            while curr:
                curr.set_params(p)
                last_action = curr._run(shared)
                curr = copy.copy(self.get_next_node(curr, last_action))
            return last_action

        def _run(self, shared):
            p = self.prep(shared)
            o = self._orch(shared)
            return self.post(shared, p, o)

        def post(self, shared, prep_res, exec_res):
            return exec_res

    class BatchFlow(Flow):
        def _run(self, shared):
            pr = self.prep(shared) or []
            for bp in pr:
                self._orch(shared, {**self.params, **bp})
            return self.post(shared, pr, None)

    class AsyncNode(Node):
        async def prep_async(self, shared):
            pass

        async def exec_async(self, prep_res):
            pass

        async def exec_fallback_async(self, prep_res, exc):
            raise exc

        async def post_async(self, shared, prep_res, exec_res):
            pass

        async def _exec(self, prep_res):
            for self.cur_retry in range(self.max_retries):
                try:
                    return await self.exec_async(prep_res)
                except Exception as e:
                    if self.cur_retry == self.max_retries - 1:
                        return await self.exec_fallback_async(prep_res, e)
                    if self.wait > 0:
                        await asyncio.sleep(self.wait)

        async def run_async(self, shared):
            if self.successors:
                warnings.warn("Node won't run successors. Use AsyncFlow.")
            return await self._run_async(shared)

        async def _run_async(self, shared):
            p = await self.prep_async(shared)
            e = await self._exec(p)
            return await self.post_async(shared, p, e)

        def _run(self, shared):
            raise RuntimeError("Use run_async.")

    class AsyncBatchNode(AsyncNode, BatchNode):
        async def _exec(self, items):
            return [await super(AsyncBatchNode, self)._exec(i) for i in items]

    class AsyncParallelBatchNode(AsyncNode, BatchNode):
        async def _exec(self, items):
            return await asyncio.gather(
                *(super(AsyncParallelBatchNode, self)._exec(i) for i in items)
            )

    class AsyncFlow(Flow, AsyncNode):
        async def _orch_async(self, shared, params=None):
            curr, p, last_action = (
                copy.copy(self.start_node),
                (params or {**self.params}),
                None,
            )
            while curr:
                curr.set_params(p)
                last_action = (
                    await curr._run_async(shared)
                    if isinstance(curr, AsyncNode)
                    else curr._run(shared)
                )
                curr = copy.copy(self.get_next_node(curr, last_action))
            return last_action

        async def _run_async(self, shared):
            p = await self.prep_async(shared)
            o = await self._orch_async(shared)
            return await self.post_async(shared, p, o)

        async def post_async(self, shared, prep_res, exec_res):
            return exec_res

    class AsyncBatchFlow(AsyncFlow, BatchFlow):
        async def _run_async(self, shared):
            pr = await self.prep_async(shared) or []
            for bp in pr:
                await self._orch_async(shared, {**self.params, **bp})
            return await self.post_async(shared, pr, None)

    class AsyncParallelBatchFlow(AsyncFlow, BatchFlow):
        async def _run_async(self, shared):
            pr = await self.prep_async(shared) or []
            await asyncio.gather(
                *(self._orch_async(shared, {**self.params, **bp}) for bp in pr)
            )
            return await self.post_async(shared, pr, None)

    return Flow, Node


@app.cell(column=1)
def _():
    # from pocketflow import Node, Flow
    import llamabot as lmb

    return (lmb,)


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

            prompt = f"Given these topics: {topics}\n\nand the original text: {txt}\m\nGenerate 2 interesting questions for each topic."
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
    return (shared_topics,)


@app.cell
def _(shared_topics):
    shared_topics
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
                model_name="gpt-4.1",
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
            pass

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
