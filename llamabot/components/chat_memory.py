"""Chat memory components for LlamaBot.

This module provides memory systems that can store and retrieve conversation history
in more sophisticated ways than simple linear storage.
"""

from abc import ABC, abstractmethod
from typing import List

from pydantic import BaseModel, Field
from llamabot.components.messages import (
    BaseMessage,
    SystemMessage,
)
from llamabot import prompt


class MessageSummary(BaseModel):
    """Summary of a message."""

    title: str = Field(..., description="Title of the message.")
    summary: str = Field(..., description="Summary of the message. Two sentences max.")


class ChosenNode(BaseModel):
    """A node that is chosen to connect to the message that is provided."""

    node: str = Field(
        ...,
        description="A node that is chosen to connect to the message that is provided.",
    )


class Node(BaseModel):
    """A node in the graph."""

    message_id: int
    message: BaseMessage
    message_summary: MessageSummary

    def __hash__(self):
        """Hash the node by its message_id."""
        return self.message_id

    def __eq__(self, other):
        """Check if the node is equal to another node."""
        if isinstance(other, Node):
            return self.message_id == other.message_id
        elif isinstance(other, int):
            return self.message_id == other
        return False


@prompt("user")
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


@prompt("user")
def node_chooser_feedback_prompt(
    candidate_nodes: list[str],
    last_message: str,
    user_message: str,
    invalid_choice: str,
    valid_node_names: list[str],
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

    ---

    IMPORTANT: You previously chose "{{ invalid_choice }}" but that is not a valid assistant node name.
    Human messages can ONLY connect to assistant (AI) messages, never to other human messages.

    The valid assistant node names available are:
    {% for valid_name in valid_node_names %}
    - {{ valid_name }}
    {% endfor %}

    Please choose one of the valid ASSISTANT node names from the list above to link the user message to.
    """


class AbstractChatMemory(ABC):
    """Abstract base class for chat memory systems."""

    @abstractmethod
    def append(self, human_message: BaseMessage, assistant_message: BaseMessage):
        """Append a conversation turn (human message + assistant response).

        :param human_message: The human message
        :param assistant_message: The assistant's response
        """
        raise NotImplementedError()

    @abstractmethod
    def retrieve(self, query: str, n_results: int = 10) -> List[BaseMessage]:
        """Retrieve relevant messages from memory.

        :param query: The query to search for
        :param n_results: Maximum number of messages to return
        :return: List of relevant messages
        """
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        """Reset the memory system."""
        raise NotImplementedError()


class GraphChatMemory(AbstractChatMemory):
    """Graph-based chat memory with conversation threading and retrieval."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "networkx is required for GraphChatMemory. "
                "Please install it with `pip install networkx`"
            )

        from llamabot.bot.structuredbot import StructuredBot

        # Core attributes
        self.graph = nx.DiGraph()
        self.linear_history = []
        self.model_name = model_name

        # Helper bots
        self.message_summarizer = StructuredBot(
            system_prompt=SystemMessage(
                content="You are an expert at summarizing messages. You will be given a message and your mission is to succinctly summarize the message."  # noqa: E501
            ),
            pydantic_model=MessageSummary,
            model_name=self.model_name,
        )

        self.node_chooser = StructuredBot(
            system_prompt=SystemMessage(
                content="You are an expert that specializes in identifying a single node from a collection of nodes is most logical to be connected to the user message that is being sent into the system. Choose the node to be connected to based on the following criteria: (a) it is a direct response to an Assistant Message, and (b) out of the options available, it is the most semantically relevant to the User Message, or (c) it is in response to the last message provided (you'll know this through vague references in the user message). You cannot return nothing (empty string), you must always return something."  # noqa: E501
            ),
            pydantic_model=ChosenNode,
            model_name=self.model_name,
        )

    def append(self, human_message: BaseMessage, assistant_message: BaseMessage):
        """Append a conversation turn to the graph memory.

        :param human_message: The human message
        :param assistant_message: The assistant's response
        """
        from llamabot.components.docstore import BM25DocStore

        # Create user node
        user_summary = self.message_summarizer(human_message)
        user_node = Node(
            message_id=len(self.linear_history),
            message=human_message,
            message_summary=user_summary,
        )
        self.linear_history.append(human_message)
        self.graph.add_node(user_node.message_summary.title, node=user_node)

        # Create assistant node
        assistant_summary = self.message_summarizer(assistant_message)
        assistant_node = Node(
            message_id=len(self.linear_history),
            message=assistant_message,
            message_summary=assistant_summary,
        )
        self.linear_history.append(assistant_message)
        self.graph.add_node(assistant_node.message_summary.title, node=assistant_node)

        # Add edge from user to assistant
        self.graph.add_edge(
            user_node.message_summary.title, assistant_node.message_summary.title
        )

        # Find best parent node to connect to (integrated parent finding logic)
        if len(self.graph) > 2:  # More than just this conversation turn
            # Get all assistant nodes for retrieval
            assistant_nodes = [
                n
                for n, d in self.graph.nodes(data=True)
                if d.get("node")
                and d["node"].message.role == "assistant"
                and n
                != assistant_node.message_summary.title  # Exclude current assistant node
            ]

            if assistant_nodes:
                # Use BM25 for initial retrieval
                bm25_store = BM25DocStore()
                node_docs = []
                for node_id in assistant_nodes:
                    node_data = self.graph.nodes[node_id]
                    doc = f"summary: {node_data['node'].message_summary.summary}\ncontent: {node_data['node'].message.content}\nnode: {node_id}"
                    node_docs.append(doc)

                bm25_store.extend(node_docs)
                retrieved = bm25_store.retrieve(human_message.content, n_results=5)

                # Use node chooser to select best connection with validation and retry
                last_message = (
                    self.linear_history[-3].content
                    if len(self.linear_history) >= 3
                    else ""
                )

                chosen_node = None
                max_retries = 3
                previous_invalid_choice = None

                for attempt in range(max_retries):
                    try:
                        if attempt == 0:
                            # First attempt - use regular prompt
                            chosen = self.node_chooser(
                                node_chooser_user_prompt(
                                    retrieved, last_message, human_message.content
                                )
                            )
                        else:
                            # Retry attempts - use feedback prompt with only assistant nodes
                            valid_node_names = (
                                assistant_nodes  # Only assistant nodes are valid
                            )
                            chosen = self.node_chooser(
                                node_chooser_feedback_prompt(
                                    retrieved,
                                    last_message,
                                    human_message.content,
                                    previous_invalid_choice,
                                    valid_node_names,
                                )
                            )

                        # Validate that chosen node exists in graph AND is an assistant node
                        if (
                            chosen.node
                            and chosen.node in self.graph.nodes()
                            and chosen.node != user_node.message_summary.title
                            and chosen.node
                            in assistant_nodes  # Must be an assistant node
                        ):
                            chosen_node = chosen.node
                            break
                        else:
                            # Invalid choice - store for feedback
                            previous_invalid_choice = chosen.node or "None"

                    except Exception:
                        # Error in node chooser - continue to next attempt
                        previous_invalid_choice = "Error occurred"
                        continue

                # Create edge if we found a valid node
                if chosen_node:
                    self.graph.add_edge(chosen_node, user_node.message_summary.title)
                else:
                    # Final fallback: connect to most recent assistant node
                    if assistant_nodes:
                        self.graph.add_edge(
                            assistant_nodes[-1], user_node.message_summary.title
                        )

    def retrieve(
        self,
        query: str,
        n_results: int = 10,
        use_graph_context: bool = True,
        context_depth: int = 5,
    ) -> List[BaseMessage]:
        """Retrieve relevant messages from the graph.

        :param query: The query to search for
        :param n_results: Maximum number of messages to return
        :param use_graph_context: Whether to use graph structure for context
        :param context_depth: How many messages to walk up the chain
        :return: List of relevant messages
        """
        from llamabot.components.docstore import BM25DocStore

        if not self.graph.nodes():
            return []

        if not use_graph_context:
            # Fallback to original flat BM25 search
            all_nodes = []
            node_map = {}

            for node_id, node_data in self.graph.nodes(data=True):
                if "node" in node_data:
                    node_obj = node_data["node"]
                    content = f"Role: {node_obj.message.role}\nSummary: {node_obj.message_summary.summary}\nContent: {node_obj.message.content}"
                    all_nodes.append(content)
                    node_map[content] = node_obj.message

            bm25_store = BM25DocStore()
            bm25_store.extend(all_nodes)
            retrieved_docs = bm25_store.retrieve(query, n_results)

            results = []
            for doc in retrieved_docs:
                if doc in node_map:
                    results.append(node_map[doc])
            return results

        # Graph-aware retrieval
        # 1. Find best attachment point using BM25 on assistant nodes only
        assistant_nodes = [
            n
            for n, d in self.graph.nodes(data=True)
            if d.get("node") and d["node"].message.role == "assistant"
        ]

        if not assistant_nodes:
            return []

        # Use BM25 to find most relevant assistant node
        bm25_store = BM25DocStore()
        node_docs = []
        node_map = {}

        for node_id in assistant_nodes:
            node_data = self.graph.nodes[node_id]
            node_obj = node_data["node"]
            doc = f"summary: {node_obj.message_summary.summary}\ncontent: {node_obj.message.content}"
            node_docs.append(doc)
            node_map[doc] = node_id

        bm25_store.extend(node_docs)
        retrieved_docs = bm25_store.retrieve(query, n_results=1)  # Get best match

        if not retrieved_docs or retrieved_docs[0] not in node_map:
            return []

        # 2. Walk up the conversation chain from the best attachment point
        start_node = node_map[retrieved_docs[0]]

        # Walk up the conversation chain
        messages = []
        current_node = start_node
        visited = set()

        for _ in range(context_depth):
            if current_node in visited or current_node not in self.graph.nodes():
                break

            visited.add(current_node)

            # Add current node's message
            node_data = self.graph.nodes[current_node]
            if "node" in node_data:
                messages.append(node_data["node"].message)

            # Find parent node (predecessor)
            predecessors = list(self.graph.predecessors(current_node))
            if predecessors:
                # Take the first predecessor (could be improved with heuristics)
                current_node = predecessors[0]
            else:
                break

        # Reverse to get chronological order (oldest first)
        return messages[::-1]

    def reset(self):
        """Reset the graph memory store."""
        self.graph.clear()
        self.linear_history.clear()

    def to_mermaid(
        self,
        node_labels=None,
        directed=True,
        graph_name="ConversationGraph",
        indent="    ",
    ) -> str:
        """Convert the conversation graph to Mermaid diagram format.

        :param node_labels: Optional dict mapping node to custom labels
        :param directed: Whether to use directed graph notation
        :param graph_name: Name for the graph
        :param indent: Indentation string
        :return: Mermaid diagram as string
        """
        if not self.graph.nodes():
            return (
                f'%% {graph_name}\ngraph TD\n{indent}EmptyGraph["No conversations yet"]'
            )

        # Generate node abbreviations
        node_abbr = {n: f"N{i}" for i, n in enumerate(self.graph.nodes())}

        if node_labels is None:
            node_labels = {}
            for n, d in self.graph.nodes(data=True):
                if "node" in d:
                    node_labels[n] = d["node"].message_summary.title
                else:
                    node_labels[n] = str(n)[:50]  # Truncate long node IDs

        # Mermaid header
        direction = "TD" if directed else "LR"
        lines = [f"%% {graph_name}", f"graph {direction}"]

        # Node definitions with labels and role-based categorization
        user_nodes = []
        assistant_nodes = []

        for n in self.graph.nodes():
            abbr = node_abbr[n]
            label = node_labels[n].replace('"', "'")  # Escape quotes
            lines.append(f'{indent}{abbr}["{label}"]')

            # Categorize nodes by role for styling
            node_data = self.graph.nodes[n]
            if "node" in node_data:
                role = node_data["node"].message.role
                if role == "user":
                    user_nodes.append(abbr)
                elif role == "assistant":
                    assistant_nodes.append(abbr)

        # Edge definitions
        edge_op = "-->" if directed else "---"
        for u, v in self.graph.edges():
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


class ChatMemory(AbstractChatMemory):
    """Simple linear chat memory that stores messages in a list."""

    def __init__(self):
        self.messages = []

    def append(self, human_message: BaseMessage, assistant_message: BaseMessage):
        """Append a conversation turn to the linear memory.

        :param human_message: The human message
        :param assistant_message: The assistant's response
        """
        self.messages.append(human_message)
        self.messages.append(assistant_message)

    def retrieve(self, query: str, n_results: int = 10) -> List[BaseMessage]:
        """Retrieve recent messages from the linear memory.

        For simplicity, this returns the most recent n_results messages.
        A more sophisticated implementation could use BM25 or semantic search.

        :param query: The query to search for (ignored in this simple implementation)
        :param n_results: Maximum number of messages to return
        :return: List of recent messages
        """
        return self.messages[-n_results:] if self.messages else []

    def reset(self):
        """Reset the linear memory."""
        self.messages.clear()
