"""Storage functions for chat memory."""

import networkx as nx
from typing import Optional, Union
from llamabot.components.messages import BaseMessage, HumanMessage, AIMessage
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
