"""Retrieval functions for chat memory."""

import networkx as nx
from typing import List
from llamabot.components.messages import BaseMessage


def get_recent_messages(graph: nx.DiGraph, n_results: int) -> List[BaseMessage]:
    """Get the most recent N messages from linear memory.

    :param graph: The conversation graph
    :param n_results: Number of messages to return
    :return: List of most recent messages
    """
    if not graph.nodes():
        return []

    # Get all nodes sorted by ID (chronological order)
    nodes = sorted(graph.nodes())

    # Get the most recent N messages
    recent_nodes = nodes[-n_results:] if len(nodes) > n_results else nodes

    # Extract messages in chronological order
    messages = []
    for node_id in recent_nodes:
        node_data = graph.nodes[node_id]["node"]
        messages.append(node_data.message)

    return messages


def semantic_search_with_context(
    graph: nx.DiGraph, query: str, n_results: int, context_depth: int
) -> List[BaseMessage]:
    """Find relevant nodes via semantic search, then traverse up thread paths for context.

    :param graph: The conversation graph
    :param query: The search query
    :param n_results: Number of relevant nodes to find
    :param context_depth: Number of nodes to traverse up each thread path
    :return: List of relevant messages with context
    """
    if not graph.nodes():
        return []

    # Find relevant nodes via semantic search
    relevant_nodes = bm25_search(graph, query, n_results)

    if not relevant_nodes:
        return []

    # Collect messages with context
    all_messages = []
    for node_id in relevant_nodes:
        # Get the relevant node
        node_data = graph.nodes[node_id]["node"]
        all_messages.append(node_data.message)

        # Get context by traversing up the thread path
        context_messages = traverse_thread_path(graph, node_id, context_depth)
        all_messages.extend(context_messages)

    return all_messages


def traverse_thread_path(
    graph: nx.DiGraph, node_id: int, depth: int
) -> List[BaseMessage]:
    """Traverse up a conversation thread path from a given node.

    :param graph: The conversation graph
    :param node_id: Starting node ID
    :param depth: Number of nodes to traverse up
    :return: List of messages in the thread path
    """
    if node_id not in graph.nodes():
        return []

    messages = []
    current_node_id = node_id

    for _ in range(depth):
        # Get current node's parent
        node_data = graph.nodes[current_node_id]["node"]
        parent_id = node_data.parent_id

        if parent_id is None or parent_id not in graph.nodes():
            break

        # Add parent message to results
        parent_data = graph.nodes[parent_id]["node"]
        messages.append(parent_data.message)

        # Move up to parent
        current_node_id = parent_id

    return messages


def bm25_search(graph: nx.DiGraph, query: str, n_results: int) -> List[int]:
    """Use BM25DocStore to find relevant nodes.

    :param graph: The conversation graph
    :param query: The search query
    :param n_results: Number of results to return
    :return: List of relevant node IDs
    """
    from llamabot.components.docstore import BM25DocStore

    # Extract all messages from the graph
    documents = []
    node_ids = []

    for node_id in sorted(graph.nodes()):
        node_data = graph.nodes[node_id]["node"]
        documents.append(node_data.message.content)
        node_ids.append(node_id)

    if not documents:
        return []

    # Use BM25DocStore for search
    docstore = BM25DocStore()
    docstore.extend(documents)

    # Get relevant documents
    relevant_docs = docstore.retrieve(query, n_results)

    # Map back to node IDs
    relevant_nodes = []
    for doc in relevant_docs:
        try:
            idx = documents.index(doc)
            relevant_nodes.append(node_ids[idx])
        except ValueError:
            continue

    return relevant_nodes
