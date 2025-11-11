# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "anthropic==0.72.0",
#     "ipython==9.7.0",
#     "llamabot",
#     "pocketflow==0.0.3",
#     "pydantic-collections==0.6.0",
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
    return


@app.cell
def _(Flow, SummarizeFile):
    summarize = SummarizeFile()
    flow_summary = Flow(start=summarize)
    # flow_summary.run(shared_summary)
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


@app.cell(column=1, hide_code=True)
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
    # result = two_node_flow.run(shared_topics)
    # result
    return (two_node_flow,)


@app.cell
def _(flow_to_mermaid, mo, two_node_flow):
    # Visualize the two-node workflow
    mo.mermaid(flow_to_mermaid(two_node_flow))
    return


@app.cell
def _(flow_to_mermaid, two_node_flow):
    print(flow_to_mermaid(two_node_flow))
    return


@app.cell(column=2, hide_code=True)
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
    from llamabot.components.pocketflow import flow_to_mermaid as _flow_to_mermaid

    def flow_to_mermaid(flow):
        """Override flow_to_mermaid to use function names for FuncNode instances."""
        lines = ["graph TD"]
        node_styles = []

        # PocketFlow uses `start_node` attribute, not `start`
        if not hasattr(flow, "start_node") or flow.start_node is None:
            return "\n".join(lines + ['A["Empty Flow"]'])

        start_node = flow.start_node
        node_id_map = {}
        next_id = [1]  # Use list to allow modification in nested function

        def collect_nodes(node):
            """Recursively collect all nodes in the graph.

            PocketFlow stores edges in node.successors dict: {action: target_node}
            """
            if node in node_id_map:
                return
            node_id_map[node] = f"N{next_id[0]}"
            next_id[0] += 1

            # PocketFlow nodes store connections in `successors` attribute
            successors = getattr(node, "successors", {})

            # Recursively collect connected nodes
            if isinstance(successors, dict):
                for action, target in successors.items():
                    if target:
                        collect_nodes(target)

        collect_nodes(start_node)

        # Generate node definitions
        for node, node_id in node_id_map.items():
            # Check if node has a 'name' property (FuncNode instances)
            if hasattr(node, "name"):
                node_name = node.name
            else:
                node_name = node.__class__.__name__
            lines.append(f'{node_id}["{node_name}"]')
            # Style nodes (light blue for visual distinction)
            style = f"style {node_id} fill:#e1f5ff,stroke:#01579b,stroke-width:2px;"
            node_styles.append(style)

        # Generate edges by traversing the graph
        visited_edges = set()

        def add_edges(node):
            """Recursively add edges to the diagram."""
            if node not in node_id_map:
                return
            node_id = node_id_map[node]

            # Get successors from PocketFlow node structure
            successors = getattr(node, "successors", {})

            # Add edges to diagram
            if isinstance(successors, dict):
                for action, target in successors.items():
                    if target and target in node_id_map:
                        target_id = node_id_map[target]
                        edge_key = (node_id, target_id, action)
                        if edge_key not in visited_edges:
                            visited_edges.add(edge_key)
                            # Add action label to edge if it's not 'default'
                            label = (
                                f'|"{action}"|'
                                if action != "default" and action
                                else ""
                            )
                            lines.append(f"{node_id} -->{label} {target_id}")
                            # Recursively add edges from target
                            add_edges(target)

        add_edges(start_node)

        lines.extend(node_styles)
        return "\n".join(lines)

    return (flow_to_mermaid,)


@app.cell
def _():
    import subprocess

    _result = subprocess.run("ls")
    return


@app.cell
def _(lmb, nodeify):
    @nodeify
    @lmb.tool
    def execute_shell_command(cmd: str):
        import subprocess

        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout

    return (execute_shell_command,)


@app.cell
def _(execute_shell_command):
    execute_shell_command("ls")
    return


@app.cell
def _(lmb):
    from llamabot.components.tools import (
        respond_to_user,
        search_internet,
        today_date,
    )
    from pydantic import BaseModel, Field
    from typing import Literal

    search_internet = lmb.tool(search_internet)

    tools = [respond_to_user, today_date]

    class ToolChoice(BaseModel):
        content: Literal[*[tool.__name__ for tool in tools]] = Field(
            ..., description="The name of the tool to use"
        )

    @lmb.prompt("system")
    def decision_bot_system_prompt():
        """Given the chat history, pick for me one or more tools to execute
        in order to satisfy the user's query.

        Give me just the tool name to pick.
        Use the tools judiciously to help answer the user's query.
        Query is always related to one of the tools.
        Use respond_to_user if you have enough information to answer the original query.
        """

    return (
        BaseModel,
        Field,
        Literal,
        ToolChoice,
        decision_bot_system_prompt,
        respond_to_user,
        today_date,
    )


@app.cell
def _(Node, ToolChoice, decision_bot_system_prompt, lmb):
    class Decide(Node):
        def prep(self, shared: dict):
            shared["memory"].append(f"Query: {shared['query']}")
            return shared

        def exec(self, prep_result):
            bot = lmb.StructuredBot(
                pydantic_model=ToolChoice,
                system_prompt=decision_bot_system_prompt(),
            )
            print(prep_result["memory"])
            response = bot(*prep_result["memory"])

            return response.content

        def post(self, shared, prep_result, exec_result):
            shared["memory"].append(f"Chosen Tool: {exec_result}")
            return exec_result

    return (Decide,)


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
def _(BaseModel, Field, Node, lmb):
    class RespondToUser(Node):
        def prep(self, shared: dict):
            return shared

        def exec(self, prep_result):
            class Response(BaseModel):
                content: str = Field(..., description="The response to the user.")

            bot = lmb.StructuredBot(
                "You are a helpful assistant.",
                model_name="ollama_chat/gemma3n:latest",
                pydantic_model=Response,
            )
            print(prep_result["memory"])
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
def _(Flow, decide, shared):
    flow2 = Flow(start=decide)
    flow2.run(shared)
    return (flow2,)


@app.cell
def _(flow2):
    flow2.run({"query": "Hey what's up?", "memory": []})
    return


@app.cell
def _(flow2, flow_to_mermaid, mo):
    # Visualize the agent flow
    mo.mermaid(flow_to_mermaid(flow2))
    return


@app.cell
def _(flow2, flow_to_mermaid):
    print(flow_to_mermaid(flow2))
    return


@app.cell(column=3, hide_code=True)
def _(mo):
    mo.md(r"""## Agent Example 2 - parameterized/shell commands""")
    return


@app.cell
def _(BaseModel, Field, Literal, Node, decision_bot_system_prompt, lmb):
    class Decide2(Node):
        def prep(self, shared: dict):
            return shared

        def exec(self, prep_result):
            sysprompt = decision_bot_system_prompt()

            print(sysprompt)

            class ToolChoice(BaseModel):
                content: Literal[*[tool.__name__ for tool in prep_result["tools"]]] = (
                    Field(..., description="The name of the tool to use")
                )
                justification: str = Field(..., description="Why this tool was chosen.")

            bot = lmb.StructuredBot(
                pydantic_model=ToolChoice,
                system_prompt=decision_bot_system_prompt(),
            )

            if prep_result["memory"]:
                response = bot(*prep_result["memory"])
            else:
                response = bot(prep_result["query"])

            return response.content

        def post(self, shared, prep_result, exec_result):
            shared["memory"].append(f"Query: {shared['query']}")
            shared["memory"].append(f"Chosen Tool: {exec_result}")
            return exec_result

    return (Decide2,)


@app.cell
def _(BaseModel, Field, Node, execute_shell_command, lmb):
    class ShellCommand(Node):
        def prep(self, shared: dict):
            return shared

        def exec(self, prep_result):
            class Cmd(BaseModel):
                content: str = Field(..., description="The shell command to execute")

            bot = lmb.StructuredBot(
                system_prompt="You are an expert at writing shell commands. For the chat trace that you will be given, write a shell command that accomplishes the user's request. Only output the command, nothing else.",
                pydantic_model=Cmd,
                model_name="ollama_chat/gemma3n:latest",
            )

            response = bot(*prep_result["memory"])
            print(response.content)
            result = execute_shell_command(response.content)
            return result

        def post(self, shared, prep_result, exec_result):
            shared["memory"].append(f"Output: {exec_result}")
            print(shared["memory"])
            return "decide"

    return (ShellCommand,)


@app.cell
def _(Decide2, Flow, RespondToUser, ShellCommand, TodayDate):
    def _():
        # Set up the graph
        today_date = TodayDate()
        respond_to_user = RespondToUser()
        decide = Decide2()
        shell_command = ShellCommand()

        decide - "today_date" >> today_date
        today_date - "decide" >> decide
        decide - "execute_shell_command" >> shell_command
        shell_command - "decide" >> decide
        decide - "respond_to_user" >> respond_to_user

        flow = Flow(start=decide)
        return flow

    flow3 = _()
    return (flow3,)


@app.cell
def _():
    return


@app.cell
def _(flow3, flow_to_mermaid, mo):
    mo.mermaid(flow_to_mermaid(flow3))
    return


@app.cell
def _(flow3, flow_to_mermaid):
    print(flow_to_mermaid(flow3))
    return


@app.cell
def _(execute_shell_command, respond_to_user, today_date):
    shared3 = dict()
    shared3["query"] = "Hey, what files have been modified today?"
    shared3["memory"] = []
    shared3["tools"] = [respond_to_user, today_date, execute_shell_command]
    # out3 = flow3.run(shared3)
    # print(out3)
    return


@app.cell
def _():
    return


@app.cell(column=4)
def _(mo):
    mo.md(r"""## AgentBot Design""")
    return


@app.cell
def _(Flow, Node, flow_to_mermaid, mo):
    from typing import List, Callable
    from llamabot.components.chat_memory import ChatMemory

    class AgentBot:
        def __init__(self, tools: List[Callable], decide_node: Node):
            self.tools = tools
            self.shared = dict(memory=[])
            self.decide_node = decide_node

            for tool in self.tools:
                self.decide_node - tool.func.__name__ >> tool
                if tool.loopback_name is not None:
                    tool - tool.loopback_name >> self.decide_node

            self.flow = Flow(start=self.decide_node)

        def __call__(self, query):
            self.shared["memory"].append(query)

            result = self.flow.run(self.shared)

            return result

        def _display_(self):
            return mo.mermaid(flow_to_mermaid(self.flow))

    return (AgentBot,)


@app.cell
def _(Node):
    def nodeify(func=None, *, loopback_name: str = "decide"):
        """Decorator to turn a function into a Pocket Flow Node.

        Works with both regular functions and llamabot @tool decorated functions.
        """

        def decorator(func):
            class FuncNode(Node):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.loopback_name = loopback_name
                    # Store reference to the original function
                    self.func = func

                @property
                def name(self):
                    """Return the name of the wrapped function."""
                    return self.func.__name__

                def prep(self, shared):
                    return shared

                def exec(self, prep_result):
                    # Extract function call arguments from shared
                    func_call = prep_result.get("func_call", {})

                    # Call the function with the arguments
                    return self.func(**func_call)

                def post(self, shared, prep_result, exec_res):
                    shared["memory"].append(exec_res)

                    if self.loopback_name is None:
                        return exec_res
                    return loopback_name

                def __getattr__(self, name):
                    """Proxy attribute access to the original function.

                    This allows ToolBot to access json_schema and __name__
                    from the original function.
                    """
                    if name == "func":
                        # Avoid infinite recursion when accessing self.func
                        raise AttributeError(
                            f"'{self.__class__.__name__}' object has no attribute '{name}'"
                        )
                    # Proxy to the original function
                    return getattr(self.func, name)

                def __call__(self, *args, **kwargs):
                    """Make FuncNode callable, proxying to the original function."""
                    return self.func(*args, **kwargs)

            return FuncNode()

        # If called without parentheses
        if func is not None:
            return decorator(func)

        # If called with parentheses
        return decorator

    return (nodeify,)


@app.cell
def _(lmb, nodeify):
    @nodeify(loopback_name="decide")
    @lmb.tool
    def some_tool(some_arg: str, some_other_arg: int) -> str:
        """Dummy tool to call on when instructed to do so.

        :param some_arg: A string argument.
        :param some_other_arg: An integer argument.
        :return: A string confirming receipt of the arguments.
        """
        return f"Received {some_arg} and {some_other_arg}"

    return (some_tool,)


@app.cell
def _(lmb, nodeify):
    @nodeify(loopback_name=None)
    @lmb.tool
    def respond_to_user2(message: str) -> str:
        """Generate a response to the user message.

        :param message: The response back to the user.
        :return: The response string.
        """
        return message

    return (respond_to_user2,)


@app.cell
def _(Node, decision_bot_system_prompt, json, lmb):
    class Decide3(Node):
        def __init__(self, tools, model_name: str = "gpt-4.1", *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.tools = tools
            self.model_name = model_name

        def prep(self, shared):
            return shared

        def exec(self, prep_res):
            """Protocol: Decide which tool to use based on query."""
            bot = lmb.ToolBot(
                model_name=self.model_name,
                tools=self.tools,
                system_prompt=decision_bot_system_prompt(),
            )

            # Get tool calls from ToolBot
            tool_calls = bot(prep_res["memory"])

            # Handle case where no tool calls are returned
            if not tool_calls:
                raise ValueError("No tool calls returned from ToolBot")

            # Get the first tool call (ToolBot typically returns one)
            tool_call = tool_calls[0]

            # Extract function name and arguments
            func_name = tool_call.function.name
            func_args_json = tool_call.function.arguments

            # Parse JSON arguments string to dict
            try:
                func_args = json.loads(func_args_json)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse tool call arguments: {e}")

            # Store in shared for the next node
            prep_res["func_call"] = func_args

            # Return the function name as the action/path
            return func_name

        def post(self, shared, prep_res, exec_res):
            shared["memory"].append(f"Chosen Tool: {exec_res}")
            return exec_res

    return (Decide3,)


@app.cell
def _(
    AgentBot,
    Decide3,
    execute_shell_command,
    nodeify,
    respond_to_user2,
    some_tool,
    today_date,
):
    today_date2 = nodeify(today_date, loopback_name="decide")
    execute_shell_command2 = nodeify(execute_shell_command, loopback_name="decide")
    tools_3 = [some_tool, respond_to_user2, today_date2, execute_shell_command2]

    decide3 = Decide3(tools=tools_3)

    agent_bot = AgentBot(tools=tools_3, decide_node=decide3)

    result = agent_bot(
        "Please use some_tool with some_arg='hello' and some_other_arg=42, and then call on respond_to_user."
    )
    return agent_bot, result


@app.cell
def _(result):
    result
    return


@app.cell
def _(agent_bot):
    agent_bot
    return


@app.cell
def _(agent_bot):
    agent_bot("What files have recently been modified?")
    return


@app.cell
def _(agent_bot):
    agent_bot("What's the holiday today?")
    return


@app.cell
def _():
    import json

    return (json,)


@app.cell(column=5)
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
    # bot("What's today's date?")
    return


@app.cell
def _():
    # bot("What files have been modified today?")
    return


@app.cell(column=6, hide_code=True)
def _(mo):
    mo.md(r"""## Flow for blog summary, tags, and banner image generation""")
    return


@app.cell
def _():
    import litellm

    litellm.drop_params = True
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""I've used to generate tags, summary, and banner image for my blog before over 3 clicks, but now I'd like to turn it into a flow that also allows me to generate them with one command, and to do it in situ and directly in the directory that my blog post is being written in."""
    )
    return


@app.cell(hide_code=True)
def _():
    txt_blog = """
    #blog-draft

    How does one let an LLM agent work continuously with as few interruptions? There are great wins to be had here if we can make agents good enough to work autonomously.

    One key to making this work is to allow for certain command line commands to be run without approval. Things like `grep`/`ripgrep`, `find`/`fd`, `pixi run pytest...`, and a few other command line command prefixes (or exact commands) can enable an LLM agent to autonomously gain context over what it's working with. For an example of CLI tools that interact with external services, I have `gh pr view` as well, which allow the LLM to autonomously cook in the background while it gains context. One key thing you want is to make sure you **only auto-accept commands that are non-destructive**, i.e. you should not auto-accept `git commit`, `git push`, `rm`, and other filesystem/git/state-modifying changes.

    For Cursor and Claude Code, being able to search the web automatically without needing approval is another superpower. I have it auto-approved on my machine, but I also monitor the outputs for prompt poisoning coming from the internet; after all, internet-based prompt poisoning is a known attack vector.

    Know that most agents have a keyboard shortcut to cancel actions; as of 3 Nov 2025, the ones that I know of are:

    - Cursor: `Ctrl+C`
    - VSCode + Github Copilot: `Cmd+Esc`
    - Claude Code: `Esc`

    If you happen to be observing the thread of what it's doing, knowing those keyboard shortcuts can help you catch an agent looping incessantly, or going down a rabbit hole that you know is bad.

    Plan mode (in Cursor and Claude) is an absolute gift from the creators of these coding agents. A common observation from users of AI-assisted coding tools is that "plan mode" (and similarly-branded modes) gives the coding agent the necessary grounding to stay on-course with a task, compared to an agent that doesn't have a plan to reference back. This behaviour is reminiscent of how humans can stay on-track with a plan if the plan is explicitly written out. I have also experienced multiple sessions where the coding agent one-shot completed a task after a few iterations on a plan; granted, these were well-defined and scoped.

    Others have written their own tips, here's a curated collection of stuff I have found useful:

    - [Simon Willison's coding agent tips](https://simonwillison.net/2025/Oct/25/coding-agent-tips/)
    - [Geoffrey Litt suggests coding like a surgeon](https://www.geoffreylitt.com/2025/10/24/code-like-a-surgeon)
    - [`@omarsar0` on Twitter loves plan mode on Claude Code](https://x.com/omarsar0/status/1984641893519839271)
    - [`@mattpocockuk` has awesome tips on how to use AI for coding](https://x.com/mattpocockuk), including [this tip](https://x.com/mattpocockuk/status/1985056806893211915)
    """
    return (txt_blog,)


@app.cell
def _(BaseModel, Field, Node, lmb):
    class BlogTags(BaseModel):
        tags: list[str] = Field(
            ...,
            description="List of lowercase blog post tags. Ideally one word.",
        )

    blog_tagger = lmb.StructuredBot(
        system_prompt="Given the following blog post, give me a list of appropriate tags for it.",
        pydantic_model=BlogTags,
        model_name="gpt-4.1",
    )

    class BlogTagger(Node):
        def prep(self, shared):
            return shared

        def exec(self, prep_res):
            txt = prep_res["blog_txt"]

            response = blog_tagger(txt)
            return response.tags

        def post(self, shared, prep_res, exec_res):
            shared["blog_tags"] = exec_res
            return "next"

    return (BlogTagger,)


@app.cell
def _(BaseModel, Field, Node, lmb):
    class BlogSummarizer(Node):
        def prep(self, shared):
            return shared

        def exec(self, prep_res):
            txt = prep_res["blog_txt"]

            class BlogSummary(BaseModel):
                summary: str = Field(
                    ...,
                    description="Concise summary of the blog post. Should begin with, 'In this blog post,...'. Written in first-person tone.",
                )

            bot = lmb.StructuredBot(
                system_prompt="You are a helpful assistant that summarizes blog posts.",
                pydantic_model=BlogSummary,
                model_name="gpt-4.1",
            )

            response = bot(txt)
            return response.summary

        def post(self, shared, prep_res, exec_res):
            shared["blog_summary"] = exec_res
            return "next"

    return (BlogSummarizer,)


@app.cell
def _(BaseModel, Field, Node, lmb):
    @lmb.prompt(role="system")
    def bannerbot_dalle_prompter_sysprompt():
        """
        **As 'Prompt Designer',**
        your role is to create highly detailed and imaginative prompts for DALL-E,
        designed to generate banner images for blog posts in a watercolor style,
        with a 16:4 aspect ratio.

        You will be given a chunk of text or a summary that comes from the blog post.
        Your task is to translate the key concepts, ideas,
        and themes from the text into an image prompt.

        **Guidelines for creating the prompt:**
        - Use vivid and descriptive language to specify the image's mood, colors,
          composition, and style.
        - Vary your approach significantly between prompts - avoid repetitive patterns,
          elements, or compositions that could make images look similar.
        - Explore diverse watercolor techniques: washes, wet-on-wet, dry brush,
          salt effects, splattering, or layered glazes.
        - Consider different artistic styles within watercolor: impressionistic,
          expressionistic, minimalist, detailed botanical, atmospheric, or abstract.
        - Vary the color palettes: warm vs cool tones, monochromatic vs complementary,
          muted vs vibrant, or seasonal color schemes.
        - Mix different compositional approaches: centered focal points, rule of thirds,
          diagonal compositions, or asymmetrical balance.
        - Incorporate varied symbolic elements: natural objects, architectural forms,
          organic shapes, geometric patterns, or conceptual representations.
        - Focus on maximizing the use of imagery and symbols to represent ideas,
          avoiding any inclusion of text or character symbols in the image.
        - If the text is vague or lacks detail, make thoughtful and creative assumptions
          to create a compelling visual representation.

        The prompt should be suitable for a variety of blog topics,
        evoking an emotional or intellectual connection to the content.
        Ensure the description specifies the watercolor art style,
        the wide 16:4 banner aspect ratio,
        and your chosen artistic approach.

        **Example Output Prompts (showing variety):**

        Example 1 (Minimalist): "A minimalist watercolor composition in 16:4 aspect ratio,
        featuring a single elegant tree branch with delicate cherry blossoms against a soft,
        pale background. The painting uses a limited palette of soft pinks and creams,
        with subtle watercolor washes creating gentle atmospheric depth."

        Example 2 (Expressionistic): "A dynamic watercolor painting in 16:4 aspect ratio,
        with bold, gestural brushstrokes in deep blues and purples creating an energetic
        abstract composition. The paint flows freely across the surface, suggesting movement
        and creativity through organic, flowing forms and vibrant color interactions."

        Example 3 (Detailed): "A detailed watercolor botanical study in 16:4 aspect ratio,
        featuring intricate leaves and flowers rendered with precise brushwork and layered
        glazes. The composition uses a rich, earthy palette with careful attention to
        light and shadow, creating depth through multiple transparent washes."

        Do **NOT** include any text or character symbols in the image description.
        """

    class BannerImagePromptGenerator(Node):
        def prep(self, shared):
            return shared

        def exec(self, prep_res):
            txt = prep_res["blog_txt"]

            class BannerImagePrompt(BaseModel):
                prompt: str = Field(
                    ...,
                    description="A concise but descriptive prompt for generating a banner image for the blog post.",
                )

            bot = lmb.StructuredBot(
                system_prompt=bannerbot_dalle_prompter_sysprompt(),
                pydantic_model=BannerImagePrompt,
                model_name="gpt-4.1",
            )

            response = bot(txt)
            return response.prompt

        def post(self, shared, prep_res, exec_res):
            shared["banner_image_prompt"] = exec_res
            return "next"

    return (BannerImagePromptGenerator,)


@app.cell
def _(Node, lmb):
    class BannerImageGenerator(Node):
        def prep(self, shared):
            return shared

        def exec(self, prep_res):
            prompt = prep_res["banner_image_prompt"]

            bot = lmb.ImageBot(size="1792x1024")

            img_url = bot(prompt)
            return img_url

        def post(self, shared, prep_res, exec_res):
            shared["banner_image_url"] = exec_res
            return "next"

    return (BannerImageGenerator,)


@app.cell
def _(
    BannerImageGenerator,
    BannerImagePromptGenerator,
    BlogSummarizer,
    BlogTagger,
    Flow,
    txt_blog,
):
    blog_tagger_node = BlogTagger()
    blog_summarizer_node = BlogSummarizer()
    banner_image_prompt_node = BannerImagePromptGenerator()
    banner_image_generator_node = BannerImageGenerator()

    blog_tagger_node - "next" >> blog_summarizer_node
    blog_summarizer_node - "next" >> banner_image_prompt_node
    banner_image_prompt_node - "next" >> banner_image_generator_node

    shared_blogging_tool = dict()
    shared_blogging_tool["blog_txt"] = txt_blog

    flow_blog_tool = Flow(start=blog_tagger_node)
    # flow_blog_tool.run(shared=shared_blogging_tool)
    return (flow_blog_tool,)


@app.cell
def _():
    # mo.image(shared_blogging_tool["banner_image_url"])
    return


@app.cell
def _(flow_blog_tool, flow_to_mermaid, mo):
    mo.mermaid(flow_to_mermaid(flow_blog_tool))
    return


@app.cell(column=7)
def _():
    return


@app.cell(column=8)
def _():
    # blog_post_flow = """graph TD
    # A[load_blog] -->|grammar_and_spell_check| B[grammar_and_spell_check]
    # B -.->|store| F
    # B -->|next| C[coherence_check]
    # C -.->|store| F
    # C -->|next| E[seo_optimizer]
    # E -.->|store| F
    # E -->|rewrite| G[rewrite]
    # F[shared] -.-> G

    # classDef sharedStyle fill:#87CEEB,stroke:#4169E1,stroke-width:2px,stroke-dasharray: 5 5
    # class F sharedStyle
    # """

    # mo.mermaid(blog_post_flow)
    return


@app.cell
def _():
    # class GrammarChecker(Node):
    #     def prep(self, shared):
    #         return shared

    #     def exec(self, prep_res):
    #         txt = prep_res["blog_post"]

    #         class GrammarIssue(BaseModel):
    #             issue: str = Field(..., description="Grammar issue discovered.")
    #             suggestion: str = Field(
    #                 ..., description="Suggested correction for grammar issue."
    #             )

    #         class GrammarIssues(BaseModel):
    #             issues: list[GrammarIssue] = Field(
    #                 ..., description="List of grammar issues found."
    #             )

    #         bot = lmb.StructuredBot(
    #             system_prompt="You are a helpful writing assistant.",
    #             pydantic_model=GrammarIssues,
    #             # model_name="ollama_chat/gemma3n:latest",
    #         )

    #         response = bot(txt)
    #         return response

    #     def post(self, shared, prep_res, exec_res):
    #         shared["issues"].extend(exec_res.issues)

    #         return "next"
    return


@app.cell
def _():
    # class IncompleteSentenceChecker(Node):
    #     def prep(self, shared):
    #         return shared

    #     def exec(self, prep_res):
    #         txt = prep_res["blog_post"]

    #         class IncompleteSentenceIssue(BaseModel):
    #             issue: str = Field(
    #                 ..., description="Incomplete sentence issue discovered."
    #             )
    #             suggestion: str = Field(
    #                 ...,
    #                 description="Suggested correction for incomplete sentence issue.",
    #             )

    #             def __str__(self) -> str:
    #                 """Return the issue + suggestion as a string."""
    #                 return f"Issue: {self.issue}\nSuggestion: {self.suggestion}"

    #         class IncompleteSentenceIssues(BaseModel):
    #             issues: list[IncompleteSentenceIssue] = Field(
    #                 ..., description="List of incomplete sentence issues found."
    #             )

    #         bot = lmb.StructuredBot(
    #             system_prompt="You are a helpful writing assistant that identifies incomplete sentences, sentence fragments, and sentences that are left hanging without proper completion.",
    #             pydantic_model=IncompleteSentenceIssues,
    #             model_name="anthropic/claude-sonnet-4-5-20250929",
    #         )

    #         response = bot(txt)
    #         return response

    #     def post(self, shared, prep_res, exec_res):
    #         shared["issues"].extend(exec_res.issues)

    #         return "next"
    return


@app.cell
def _():
    # class SelfContradictionChecker(Node):
    #     def prep(self, shared):
    #         return shared

    #     def exec(self, prep_res):
    #         txt = prep_res["blog_post"]

    #         class SelfContradictionIssue(BaseModel):
    #             issue: str = Field(
    #                 ...,
    #                 description="Specific self-contradiction issue discovered. This is where one point made in the text directly contradicts another point made in the text, or has the potential to be misunderstood as such.",
    #             )
    #             suggestion: str = Field(
    #                 ...,
    #                 description="Suggested correction for self-contradiction issue.",
    #             )

    #             def __str__(self) -> str:
    #                 """Return the issue + suggestion as a string."""
    #                 return f"Issue: {self.issue}\nSuggestion: {self.suggestion}"

    #         class SelfContradictionIssues(BaseModel):
    #             issues: list[SelfContradictionIssue] = Field(
    #                 ..., description="List of self-contradiction issues found."
    #             )

    #         bot = lmb.StructuredBot(
    #             system_prompt="You are a helpful writing assistant.",
    #             pydantic_model=SelfContradictionIssues,
    #             model_name="anthropic/claude-sonnet-4-5-20250929",
    #         )

    #         response = bot(txt)
    #         return response

    #     def post(self, shared, prep_res, exec_res):
    #         shared["issues"].extend(exec_res.issues)

    #         return "next"
    return


@app.cell
def _():
    # class SEOOptimizer(Node):
    #     def prep(self, shared):
    #         return shared

    #     def exec(self, prep_res):
    #         txt = prep_res["blog_post"]

    #         class SEOOptimizerIssue(BaseModel):
    #             issue: str = Field(
    #                 ..., description="SEO problem with the blog post."
    #             )
    #             suggestion: str = Field(
    #                 ...,
    #                 description="Specific suggestion search-engine optimize the blog post.",
    #             )

    #             def __str__(self) -> str:
    #                 """Return the issue + suggestion as a string."""
    #                 return f"Issue: {self.issue}\nSuggestion: {self.suggestion}"

    #         class SEOOptimizerIssues(BaseModel):
    #             issues: list[SEOOptimizerIssue] = Field(
    #                 ..., description="List of SEO suggestions."
    #             )

    #         bot = lmb.StructuredBot(
    #             system_prompt="You are a helpful writing assistant.",
    #             pydantic_model=SelfContradictionIssues,
    #             model_name="anthropic/claude-sonnet-4-5-20250929",
    #         )

    #         response = bot(txt)
    #         return response

    #     def post(self, shared, prep_res, exec_res):
    #         shared["self_contradiction_issues"] = exec_res

    #         return "next"
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    # class RewrittenBlogPost(BaseModel):
    #     content: str = Field(..., description="The rewritten blog post.")

    # class Rewrite(Node):
    #     def prep(self, shared):
    #         return shared

    #     def exec(self, prep_res):
    #         bot = lmb.StructuredBot(
    #             system_prompt="You are an expert blog post writer. Rewrite the blog post to address all issues raised, improving grammar, coherence, and eliminating self-contradictions, while preserving the tone, voice, style, and general feel of the original. Ensure the final output is polished and ready for publication.",
    #             pydantic_model=RewrittenBlogPost,
    #             model_name="anthropic/claude-sonnet-4-5-20250929",
    #         )

    #         all_issues = [str(i) for i in prep_res["issues"]]

    #         response = bot(*all_issues)
    #         prep_res["rewritten_blog"] = response.content

    #     def post(self, shared, prep_res, exec_res):
    #         shared["rewritten_blog"] = prep_res["rewritten_blog"]
    #         return "done"
    return


@app.cell
def _():
    # shared_blog["grammar_issues"]
    return


@app.cell
def _():
    return


@app.cell
def _():
    # # Node instantiation
    # grammar_checker = GrammarChecker()
    # incomplete_sentence_checker = IncompleteSentenceChecker()
    # self_contradiction_checker = SelfContradictionChecker()
    # rewrite = Rewrite()

    # # Edge (flow) construction
    # grammar_checker - "next" >> incomplete_sentence_checker
    # incomplete_sentence_checker - "next" >> self_contradiction_checker
    # self_contradiction_checker - "next" >> rewrite

    # # Create shared
    # shared_blog = dict()
    # shared_blog["blog_post"] = txt_blog
    # shared_blog["issues"] = []

    # # Execute flow
    # flow_blog = Flow(start=grammar_checker)
    # flow_blog.run(shared_blog)
    return


@app.cell
def _():
    return


@app.cell
def _():
    # shared_blog["issues"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Interactive Issue Viewer""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Approach 1: Tabs with slider navigation**

    Each issue type gets its own tab, and within each tab you can navigate through issues one-by-one.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    # grammar_issues = shared_blog["grammar_issues"].issues
    # coherence_issues = shared_blog["coherence_issues"].issues
    # contradiction_issues = shared_blog["self_contradiction_issues"].issues

    # grammar_slider = (
    #     mo.ui.slider(
    #         0,
    #         max(0, len(grammar_issues) - 1),
    #         value=0,
    #         label=f"Grammar Issue (1 of {len(grammar_issues)})",
    #         step=1,
    #         full_width=True,
    #     )
    #     if grammar_issues
    #     else None
    # )

    # coherence_slider = (
    #     mo.ui.slider(
    #         0,
    #         max(0, len(coherence_issues) - 1),
    #         value=0,
    #         label=f"Coherence Issue (1 of {len(coherence_issues)})",
    #         step=1,
    #         full_width=True,
    #     )
    #     if coherence_issues
    #     else None
    # )

    # contradiction_slider = (
    #     mo.ui.slider(
    #         0,
    #         max(0, len(contradiction_issues) - 1),
    #         value=0,
    #         label=f"Self-Contradiction Issue (1 of {len(contradiction_issues)})",
    #         step=1,
    #         full_width=True,
    #     )
    #     if contradiction_issues
    #     else None
    # )
    return


@app.cell(hide_code=True)
def _():
    # def render_issue_card(issue, issue_type):
    #     return mo.md(
    #         f"""
    #         <div style="border: 2px solid #01579b; border-radius: 8px; padding: 20px; background-color: #e1f5ff; margin: 10px 0;">
    #             <h3 style="color: #01579b; margin-top: 0;">🔍 {issue_type}</h3>
    #             <div style="background-color: white; padding: 15px; border-radius: 5px; margin: 10px 0;">
    #                 <strong>Issue:</strong>
    #                 <p style="margin: 5px 0;">{issue.issue}</p>
    #             </div>
    #             <div style="background-color: #c8e6c9; padding: 15px; border-radius: 5px; margin: 10px 0;">
    #                 <strong>✅ Suggestion:</strong>
    #                 <p style="margin: 5px 0;">{issue.suggestion}</p>
    #             </div>
    #         </div>
    #         """
    #     )

    # grammar_tab = (
    #     mo.vstack(
    #         [
    #             grammar_slider,
    #             render_issue_card(
    #                 grammar_issues[grammar_slider.value], "Grammar Issue"
    #             )
    #             if grammar_issues
    #             else mo.md("No grammar issues found!"),
    #         ]
    #     )
    #     if grammar_slider
    #     else mo.md("No grammar issues found!")
    # )

    # coherence_tab = (
    #     mo.vstack(
    #         [
    #             coherence_slider,
    #             render_issue_card(
    #                 coherence_issues[coherence_slider.value], "Coherence Issue"
    #             )
    #             if coherence_issues
    #             else mo.md("No coherence issues found!"),
    #         ]
    #     )
    #     if coherence_slider
    #     else mo.md("No coherence issues found!")
    # )

    # contradiction_tab = (
    #     mo.vstack(
    #         [
    #             contradiction_slider,
    #             render_issue_card(
    #                 contradiction_issues[contradiction_slider.value],
    #                 "Self-Contradiction Issue",
    #             )
    #             if contradiction_issues
    #             else mo.md("No self-contradiction issues found!"),
    #         ]
    #     )
    #     if contradiction_slider
    #     else mo.md("No self-contradiction issues found!")
    # )

    # tabs_viewer = mo.ui.tabs(
    #     {
    #         "📝 Grammar": grammar_tab,
    #         "🧩 Coherence": coherence_tab,
    #         "⚠️ Self-Contradiction": contradiction_tab,
    #     }
    # )

    # tabs_viewer
    return


@app.cell
def _():
    # mo.md(shared_blog["rewritten_blog"])
    return


if __name__ == "__main__":
    app.run()
