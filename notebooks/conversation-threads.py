# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "anthropic==0.57.1",
#     "llamabot[all]==0.12.11",
#     "marimo",
#     "matplotlib==3.10.3",
#     "networkx==3.5",
#     "pydantic==2.11.7",
# ]
#
# [tool.uv.sources]
# llamabot = { path = "../", editable = true }
# ///

import marimo

__generated_with = "0.14.10"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import networkx as nx
    import llamabot as lmb

    return lmb, nx


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Conversation Knowledge Graph

    The conversation knowledge graph combines hybrid search on all conversation turns with knowledge tree generation from a conversation. It serves as a memory module for chats.
    """
    )
    return


@app.cell
def _(lmb, nx):
    from datetime import datetime

    class ConversationKG:
        def __init__(self, table_name: str):
            self.graph = nx.DiGraph()
            self.docstore = lmb.LanceDBDocStore(table_name=table_name)
            self.docstore.reset()

        def append(self):
            raise NotImplementedError()

        def extend(self, conversation_turn: list[str]):
            """Append a single conversation turn to the conversation knowledge graph."""
            # Add to conversation graph.
            for msg in conversation_turn:
                self.graph.add_node(msg, time=datetime.now())

            for msg1, msg2 in zip(conversation_turn[:-1], conversation_turn[1:]):
                self.graph.add_edge(msg1, msg2)

            # Also sync up and add to docstore.
            self.docstore.extend(conversation_turn)

        def __str__(self) -> str:
            edgelist = []
            for n1, n2 in self.graph.edges():
                edgelist.append(f"[[{n1}]] -> [[{n2}\n]]")
            return "".join(edgelist)

    kg = ConversationKG(table_name="test-convo-kg")
    str(kg)
    return (kg,)


@app.cell
def _(kg, lmb):
    bot = lmb.SimpleBot("You are a helpful assistant.")

    message1 = lmb.user("I'm looking to generate some code to help me make a coffee.")
    response1 = bot(message1)
    kg.extend([message1.content, response1.content])
    return (bot,)


@app.cell
def _(kg):
    print(kg)
    return


@app.cell
def _(kg, message2):
    kg.docstore.retrieve(message2.content, 1)
    return


@app.cell
def _(kg, lmb):
    # First step, we need to automatically retrieve the node that is most likely to be the one that we want.
    # For simplicity, I am going to just use hybrid search on the kg docstore.
    message2 = lmb.user("I'd like to try doing the checklist please.")
    retrieved2 = kg.docstore.retrieve(message2.content, 1)
    retrieved2
    return message2, retrieved2


@app.cell
def _(bot, kg, message2, retrieved2):
    # This one should append automatically to the coffee response by the bot.

    response2 = bot(retrieved2[0], message2)
    kg.extend([retrieved2[0], message2.content, response2.content])
    return (response2,)


@app.cell
def _(kg):
    len(kg.graph.nodes)
    return


@app.cell
def _(kg, nx):
    import matplotlib.pyplot as plt

    nx.draw(kg.graph)
    plt.show()
    return (plt,)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(kg, lmb):
    message3 = lmb.user("I'd like to try doing the simulation now instead.")
    retrieved3 = kg.docstore.retrieve(message3.content, 1)
    retrieved3
    return


@app.cell
def _():
    # This tells me that we cannot naïvely rely on cosine similarity search to do "appropriate previous node" search
    # I need to do better, probably by doing two things:
    # 1. Try using one StructuredBot to generate a message ID that should be used as the parent node, and
    # 2. Use another StructuredBot to generate a one-sentence summary of the generated message, and use that as the node id.
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell(column=1)
def _(lmb, response2):
    from pydantic import BaseModel, Field
    from hashlib import sha256
    from llamabot.components.messages import BaseMessage

    # This one is for StructuredBot usage
    class MessageSummary(BaseModel):
        title: str = Field(..., description="Title of the message.")
        summary: str = Field(
            ..., description="Summary of the message. Two sentences max."
        )

    # This one is not for structuredbot usage, but as a datamodel.
    # I want this present so that we have automatic validation.
    # Don't want to have message accidentally be a `str`.
    class Node(BaseModel):
        message_id: int
        message: BaseMessage
        message_summary: MessageSummary

        def __hash__(self):
            # node_text = f"{self.message_summary.title}: {self.message_summary.summary}\n{self.message.role}: {self.message.content}"
            return self.message_id

        def __eq__(self, other):
            if isinstance(other, Node):
                return self.message_id == other.message_id
            elif isinstance(other, int):
                return self.message_id == other
            return False

    message_summarizer_bot = lmb.StructuredBot(
        lmb.system(
            "You are an expert at summarizing messages. You will be given a message and your mission is to succinctly summarize the message."
        ),
        pydantic_model=MessageSummary,
        model_name="gpt-4.1-mini",
    )

    response_summary = message_summarizer_bot(response2)
    response_summary
    return BaseModel, Field, MessageSummary, Node, message_summarizer_bot


@app.cell
def _(nx):
    # node = Node(message_id=1, message_summary=response_summary, message=response2)

    G = nx.DiGraph()
    len(G)
    return (G,)


@app.cell
def _(nx, plt):
    def draw(G):
        from networkx.drawing.nx_agraph import (
            graphviz_layout,
        )  # Requires pygraphviz

        pos = graphviz_layout(G, prog="dot")  # 'dot' is the hierarchical layout

        nx.draw(G, pos, with_labels=True)
        plt.show()

    return


@app.function
def to_mermaid(
    G,
    node_abbr=None,
    node_labels=None,
    directed=True,
    graph_name="G",
    indent="    ",
):
    """
    Convert a NetworkX graph to Mermaid diagram text with abbreviated node symbols and labels.
    Colors nodes based on message role.

    Parameters:
        G: networkx.Graph or networkx.DiGraph
        node_abbr: dict mapping node to abbreviation (e.g., {"Node1": "A"})
        node_labels: dict mapping node to label (e.g., {"Node1": "Start Node"})
        directed: bool, True for directed graph (TD), False for undirected (LR)
        graph_name: str, optional, name for the graph
        indent: str, indentation for lines

    Returns:
        str: Mermaid diagram as a string
    """
    if node_abbr is None:
        # Default: use numbers/letters as abbreviations
        node_abbr = {n: f"N{i}" for i, n in enumerate(G.nodes())}
    if node_labels is None:
        node_labels = {n: str(n) for n in G.nodes()}

    # Mermaid header
    direction = "TD" if directed else "LR"
    lines = [f"%% {graph_name}", f"graph {direction}"]

    # Node definitions with labels
    user_nodes = []
    assistant_nodes = []

    for n in G.nodes():
        abbr = node_abbr[n]
        label = node_labels[n]
        # Mermaid node with label: abbr["label"]
        lines.append(f'{indent}{abbr}["{label}"]')

        # Categorize nodes by role for styling
        if "node" in G.nodes[n]:
            role = G.nodes[n]["node"].message.role
            if role == "user":
                user_nodes.append(abbr)
            elif role == "assistant":
                assistant_nodes.append(abbr)

    # Edge definitions
    edge_op = "-->" if directed else "---"
    for u, v in G.edges():
        u_abbr = node_abbr[u]
        v_abbr = node_abbr[v]
        lines.append(f"{indent}{u_abbr} {edge_op} {v_abbr}")

    # Add styling for different roles
    if user_nodes or assistant_nodes:
        lines.append("")  # Empty line for readability

        # Define classes for different roles
        lines.append(
            f"{indent}classDef userClass fill:#e1f5fe,stroke:#0277bd,stroke-width:2px"
        )
        lines.append(
            f"{indent}classDef assistantClass fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px"
        )

        # Apply classes to nodes
        if user_nodes:
            user_nodes_str = ",".join(user_nodes)
            lines.append(f"{indent}class {user_nodes_str} userClass")

        if assistant_nodes:
            assistant_nodes_str = ",".join(assistant_nodes)
            lines.append(f"{indent}class {assistant_nodes_str} assistantClass")

    return "\n".join(lines)


@app.cell
def _():
    # Note: we do not distinguish between retrieval for the purposes of appending to knowledge graph and retrieval for purposes of
    return


@app.cell
def _(BaseModel, Field, lmb):
    class ChosenNode(BaseModel):
        node: str = Field(
            ...,
            description="A node that is chosen to connect to the message that is provided.",
        )

    node_chooser = lmb.StructuredBot(
        system_prompt="You are an expert that specializes in identifying a single node from a collection of nodes is most logical to be connected to the user message that is being sent into the system. Choose the node to be connected to based on the following criteria: (a) it is a direct response to an Assistant Message, and (b) out of the options available, it is the most semantically relevant to the User Message. You cannot return nothing (empty string), you must always return something.",
        pydantic_model=ChosenNode,
        model_name="gpt-4.1-mini",
    )
    return ChosenNode, node_chooser


@app.cell
def _(lmb):
    # Now we try doing the full conversation turn twice.
    linear_history = []

    @lmb.prompt("user")
    def node_chooser_user_prompt(
        candidate_nodes: list[str], last_message: str, user_message: str
    ):
        """Here are the candidate nodes to choose from:

        {% for node in candidate_nodes %}
        {{ node }}
        ---
        {% endfor %}

        This is the last message that was AI-generated:

        {{ last_message }}

        ---

        And finally, here is the user message:

        {{ user_message }}

        Now choose the candidate node to link the user message to.
        """

    return linear_history, node_chooser_user_prompt


@app.cell
def _(
    G,
    Node,
    bot,
    linear_history,
    lmb,
    message_summarizer_bot,
    node_chooser,
    node_chooser_user_prompt,
):
    def conversation_turn(message):
        # Perform retrieval
        # Step 1: linear history using last message.
        messages = []
        chosen_node = None

        if len(G):
            # Use in-memory BM25 docstore, generated on the fly.
            ds = lmb.BM25DocStore()
            # Cast to list for debugging; in reality, don't need to cast to list.
            nodes_to_add = list(
                n
                for n, d in G.nodes(data=True)
                if d["node"].message.role == "assistant"
            )
            print("Nodes to add: ", nodes_to_add)
            nodes_to_add = [
                f"summary: {G.nodes[nta]['node'].message_summary.summary} \nnode: {nta}"
                for nta in nodes_to_add
            ]
            ds.extend(nodes_to_add)
            retrieved_messages = ds.retrieve(message.content, n_results=5)

            chosen_node = node_chooser(
                node_chooser_user_prompt(retrieved_messages, message)
            )

        # Generate response
        if chosen_node is not None:
            response = bot(
                G.nodes[chosen_node.node]["node"].message,
                messages[-1] if messages else "",
                message,
            )
        else:
            response = bot(messages[-1] if messages else "", message)

        # Update memory
        message_summary = message_summarizer_bot(message)
        response_summary = message_summarizer_bot(response)

        message_node = Node(
            message_id=len(linear_history),
            message=message,
            message_summary=message_summary,
        )
        linear_history.append(message)
        G.add_node(message_node.message_summary.title, node=message_node)

        response_node = Node(
            message_id=len(linear_history),
            message=response,
            message_summary=response_summary,
        )
        linear_history.append(response)
        G.add_node(response_node.message_summary.title, node=response_node)

        G.add_edge(
            message_node.message_summary.title, response_node.message_summary.title
        )

        if chosen_node:
            G.add_edge(
                G.nodes[chosen_node.node]["node"].message_summary.title,
                message_node.message_summary.title,
            )

        return response

    return


@app.cell
def _():
    # G
    # msg1 = lmb.user("I'm looking to generate some code to help me make a coffee.")

    # resp1 = conversation_turn(msg1)
    # print(resp1)
    # mo.mermaid(to_mermaid(G))
    return


@app.cell
def _():
    # msg2 = lmb.user("I'd like to try doing the checklist please.")

    # resp2 = conversation_turn(msg2)
    # print(resp2)
    # mo.mermaid(to_mermaid(G))
    return


@app.cell
def _():
    # msg3 = lmb.user("Let's do all three steps.")

    # resp3 = conversation_turn(msg3)
    # print(resp3)
    # mo.mermaid(to_mermaid(G))
    return


@app.cell
def _():
    # msg4 = lmb.user("Can we try a simulation instead")

    # resp4 = conversation_turn(msg4)
    # print(resp4)
    # mo.mermaid(to_mermaid(G))
    return


@app.cell
def _():
    # resp4

    # node_selector = mo.ui.dropdown(options=list(G.nodes()))
    # node_selector
    return


@app.cell
def _():
    # G.nodes[node_selector.value]["node"].__dict__["message"].__dict__
    return


@app.cell
def _():
    # # Find bidirectional edges
    # for n1, n2 in G.edges():
    #     if G.has_edge(n2, n1):
    #         print(n1, "<-->", n2)
    return


@app.cell
def _():
    # msg5 = lmb.user("No good.")

    # resp5 = conversation_turn(msg5)
    # print(resp5)
    # mo.mermaid(to_mermaid(G))
    return


@app.cell
def _():
    # msg6 = lmb.user(
    #     "Let's go back to thinking about the simulation. Can you write it more concisely?"
    # )

    # resp6 = conversation_turn(msg6)
    # print(resp6)
    # mo.mermaid(to_mermaid(G))
    return


@app.cell
def _():
    return


@app.cell(column=2)
def _(ChosenNode, MessageSummary, Node, lmb, node_chooser_user_prompt, nx):
    class GraphMemoryBot(lmb.SimpleBot):
        def __init__(self):
            super().__init__("You are a helpful assistant.")
            self.G = nx.DiGraph()
            self.linear_history = []

            # Initialize the helper bots
            self.message_summarizer_bot = lmb.StructuredBot(
                lmb.system(
                    "You are an expert at summarizing messages. You will be given a message and your mission is to succinctly summarize the message."
                ),
                pydantic_model=MessageSummary,
                model_name="gpt-4.1",
            )

            self.node_chooser = lmb.StructuredBot(
                system_prompt="You are an expert that specializes in identifying a single node from a collection of nodes is most logical to be connected to the user message that is being sent into the system. Choose the node to be connected to based on the following criteria: (a) it is a direct response to an Assistant Message, and (b) out of the options available, it is the most semantically relevant to the User Message, or (c) it is in response to the last message provided (you'll know this through vague references in the user message). You cannot return nothing (empty string), you must always return something.",
                pydantic_model=ChosenNode,
                model_name="gpt-4.1-mini",
            )

        def __call__(self, message):
            # Perform retrieval
            chosen_node = None

            if len(self.G):
                # Use in-memory BM25 docstore, generated on the fly.
                ds = lmb.BM25DocStore()
                # Cast to list for debugging; in reality, don't need to cast to list.
                nodes_to_add = list(
                    n
                    for n, d in self.G.nodes(data=True)
                    if d["node"].message.role == "assistant"
                )
                print("\nNodes to add: ", nodes_to_add, "\n")
                nodes_to_add = [
                    f"summary: {self.G.nodes[nta]['node'].message_summary.summary} \ncontent: {self.G.nodes[nta]['node'].message.content} \nnode: {nta}"
                    for nta in nodes_to_add
                ]
                ds.extend(nodes_to_add)
                retrieved_messages = ds.retrieve(message.content, n_results=5)

                chosen_node = self.node_chooser(
                    node_chooser_user_prompt(
                        retrieved_messages,
                        self.linear_history[-1].content,
                        message,
                    )
                )

            # Generate response
            if chosen_node is not None:
                response = super().__call__(
                    self.G.nodes[chosen_node.node]["node"].message,
                    self.linear_history[-1] if self.linear_history else "",
                    message,
                )
            else:
                response = super().__call__(
                    self.linear_history[-1] if self.linear_history else "", message
                )

            # Update memory
            message_summary = self.message_summarizer_bot(message)
            response_summary = self.message_summarizer_bot(response)

            message_node = Node(
                message_id=len(self.linear_history),
                message=message,
                message_summary=message_summary,
            )
            self.linear_history.append(message)
            self.G.add_node(message_node.message_summary.title, node=message_node)

            response_node = Node(
                message_id=len(self.linear_history),
                message=response,
                message_summary=response_summary,
            )
            self.linear_history.append(response)
            self.G.add_node(response_node.message_summary.title, node=response_node)

            self.G.add_edge(
                message_node.message_summary.title,
                response_node.message_summary.title,
            )

            if chosen_node:
                self.G.add_edge(
                    self.G.nodes[chosen_node.node]["node"].message_summary.title,
                    message_node.message_summary.title,
                )

            return response

    return (GraphMemoryBot,)


@app.cell
def _(GraphMemoryBot, lmb, mo):
    gmbot = GraphMemoryBot()
    r1 = gmbot(
        lmb.user("I'd like for you to write me some code to make coffee please.")
    )
    mo.mermaid(to_mermaid(gmbot.G))
    return (gmbot,)


@app.cell
def _(gmbot, lmb, mo):
    r2 = gmbot(lmb.user("Let's make that API actually."))
    mo.mermaid(to_mermaid(gmbot.G))
    return


@app.cell
def _(gmbot, lmb, mo):
    r3 = gmbot(lmb.user("Let's expand the API to handle those cases."))
    mo.mermaid(to_mermaid(gmbot.G))
    return


@app.cell
def _(gmbot, lmb, mo):
    r4 = gmbot(
        lmb.user(
            "Let's rewind and try a different thing. Make me a coffee-making checklist program."
        )
    )
    mo.mermaid(to_mermaid(gmbot.G))
    return


@app.cell
def _(gmbot, lmb, mo):
    r5 = gmbot(
        lmb.user("Can you turn that checklist program into an API instead please?")
    )
    mo.mermaid(to_mermaid(gmbot.G))
    return


@app.cell
def _(gmbot, lmb, mo):
    r6 = gmbot(
        lmb.user(
            "OK, let's go back to the coffee-making checklist program. What did we talk about last?"
        )
    )
    mo.mermaid(to_mermaid(gmbot.G))
    return


@app.cell
def _(gmbot, lmb, mo):
    msg7 = lmb.user(
        "Let's make improvements to the interactive coffee-making checklist program. What did we last talk about?"
    )
    r7 = gmbot(msg7)
    mo.mermaid(to_mermaid(gmbot.G))
    return


@app.cell
def _():
    return


@app.cell
def _(gmbot, lmb, mo):
    msg8 = lmb.user("let's add the feature where we can save progress between runs.")
    r8 = gmbot(msg8)
    mo.mermaid(to_mermaid(gmbot.G))
    return


@app.cell
def _(gmbot, lmb, mo):
    gmbot(lmb.user("Let's go back to talking about interactive runs please."))
    mo.mermaid(to_mermaid(gmbot.G))
    return


@app.cell
def _(gmbot, lmb, mo):
    gmbot(lmb.user("Let's go back to talking about interactive runs please."))
    mo.mermaid(to_mermaid(gmbot.G))
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell(column=3)
def _(lmb):
    memory = lmb.GraphChatMemory()
    return


if __name__ == "__main__":
    app.run()
