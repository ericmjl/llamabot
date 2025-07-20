"""Visualization functions for chat memory."""

import networkx as nx


def to_mermaid(
    graph: nx.DiGraph, show_summaries: bool = True, max_content_length: int = 100
) -> str:
    """Convert graph to Mermaid diagram.

    :param graph: The conversation graph
    :param show_summaries: Whether to include message summaries
    :param max_content_length: Maximum length of message content to show
    :return: Mermaid diagram string
    """
    if not graph.nodes():
        return "graph TD\n"

    lines = ["graph TD"]

    # Add nodes
    for node_id in sorted(graph.nodes()):
        node_data = graph.nodes[node_id]["node"]
        message = node_data.message

        # Determine node label
        role_prefix = "H" if message.role == "user" else "A"
        node_label = f"{role_prefix}{node_id}"

        # Determine node content
        if show_summaries and node_data.summary:
            content = node_data.summary.title
        else:
            content = message.content
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."

        # Escape quotes and special characters for Mermaid
        content = content.replace('"', '\\"').replace("\n", " ")

        # Add node
        lines.append(f'    {node_label}["{node_label}: {content}"]')

    # Add edges
    for edge in graph.edges():
        source_id, target_id = edge
        source_role = (
            "H" if graph.nodes[source_id]["node"].message.role == "user" else "A"
        )
        target_role = (
            "H" if graph.nodes[target_id]["node"].message.role == "user" else "A"
        )

        source_label = f"{source_role}{source_id}"
        target_label = f"{target_role}{target_id}"

        lines.append(f"    {source_label} --> {target_label}")

    return "\n".join(lines)
