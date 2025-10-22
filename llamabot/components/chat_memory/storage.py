"""Storage functions for chat memory."""

import networkx as nx
from typing import Optional, Union
from llamabot.components.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ObservationMessage,
    ThoughtMessage,
)
from llamabot.components.chat_memory.models import ConversationNode
from llamabot.components.chat_memory.selectors import NodeSelector


def append_linear(
    graph: nx.DiGraph,
    human_message: Union[HumanMessage, BaseMessage],
    assistant_message: Union[AIMessage, BaseMessage],
    next_node_id: int,
):
    """Append messages to linear memory.

    :param graph: The conversation graph
    :param human_message: The human message
    :param assistant_message: The assistant message
    :param next_node_id: The next node ID to use
    """
    # Find the leaf node to connect to
    parent_id = _find_leaf_node(graph)

    # Create human node
    human_node = ConversationNode(
        id=next_node_id, message=human_message, parent_id=parent_id
    )
    graph.add_node(next_node_id, node=human_node)

    # Create assistant node
    assistant_node = ConversationNode(
        id=next_node_id + 1, message=assistant_message, parent_id=next_node_id
    )
    graph.add_node(next_node_id + 1, node=assistant_node)

    # Add edges
    if parent_id:
        graph.add_edge(parent_id, next_node_id)
    graph.add_edge(next_node_id, next_node_id + 1)


def append_with_threading(
    graph: nx.DiGraph,
    human_message: Union[HumanMessage, BaseMessage],
    assistant_message: Union[AIMessage, BaseMessage],
    node_selector: NodeSelector,
    next_node_id: int,
):
    """Append messages with intelligent threading.

    :param graph: The conversation graph
    :param human_message: The human message
    :param assistant_message: The assistant message
    :param node_selector: The node selector to use
    :param next_node_id: The next node ID to use
    """
    # Use node selector to find best parent for human message
    parent_id = node_selector.select_parent(graph, human_message)

    # Create human node
    human_node = ConversationNode(
        id=next_node_id, message=human_message, parent_id=parent_id
    )
    graph.add_node(next_node_id, node=human_node)

    # Create assistant node
    assistant_node = ConversationNode(
        id=next_node_id + 1, message=assistant_message, parent_id=next_node_id
    )
    graph.add_node(next_node_id + 1, node=assistant_node)

    # Add edges
    if parent_id:
        graph.add_edge(parent_id, next_node_id)
    graph.add_edge(next_node_id, next_node_id + 1)


def _find_leaf_node(graph: nx.DiGraph) -> Optional[int]:
    """Find the leaf node in the graph.

    :param graph: The conversation graph
    :return: Node ID of the leaf node, or None if no nodes exist
    """
    if not graph.nodes():
        return None

    # Find nodes with no out_edges (leaf nodes)
    leaf_nodes = [node for node in graph.nodes() if graph.out_degree(node) == 0]

    if not leaf_nodes:
        return None

    # Return the leaf node (in a linear graph, there should be only one)
    return leaf_nodes[0]


def append_linear_message(
    graph: nx.DiGraph,
    message: BaseMessage,
    next_node_id: int,
):
    """Append a single message to linear memory.

    :param graph: The conversation graph
    :param message: Any message to append
    :param next_node_id: The next node ID to use
    """
    # Find the leaf node (last message in linear chain)
    parent_id = _find_leaf_node(graph)

    # Create node
    node = ConversationNode(id=next_node_id, message=message, parent_id=parent_id)
    graph.add_node(next_node_id, node=node)

    # Add edge with simple relationship label
    if parent_id:
        relationship = _get_simple_relationship(
            graph.nodes[parent_id]["node"].message, message
        )
        graph.add_edge(parent_id, next_node_id, relationship=relationship)


def append_threaded_message(
    graph: nx.DiGraph,
    message: BaseMessage,
    node_selector: NodeSelector,
    next_node_id: int,
):
    """Append message with semantic threading.

    :param graph: The conversation graph
    :param message: Any message to append
    :param node_selector: The node selector to use
    :param next_node_id: The next node ID to use
    """
    # Use node selector to find best parent
    parent_id = node_selector.select_parent(graph, message)

    # Create node
    node = ConversationNode(id=next_node_id, message=message, parent_id=parent_id)
    graph.add_node(next_node_id, node=node)

    # Add edge with relationship label
    if parent_id:
        relationship = _get_simple_relationship(
            graph.nodes[parent_id]["node"].message, message
        )
        graph.add_edge(parent_id, next_node_id, relationship=relationship)


def _get_simple_relationship(parent_msg: BaseMessage, child_msg: BaseMessage) -> str:
    """Generate simple relationship label based on message types.

    :param parent_msg: Parent message
    :param child_msg: Child message
    :return: Relationship label
    """
    # Map message roles to readable names
    role_map = {"user": "question", "assistant": "response", "system": "instruction"}

    parent_role = role_map.get(parent_msg.role, parent_msg.role)
    child_role = role_map.get(child_msg.role, child_msg.role)

    # Check for specific message types using isinstance()
    if isinstance(child_msg, ObservationMessage):
        return f"{parent_role}→observation"
    if isinstance(parent_msg, ObservationMessage):
        return "observation→response"
    if isinstance(parent_msg, ThoughtMessage):
        return "thought→action"

    return f"{parent_role}→{child_role}"
