"""Visualization functions for chat memory."""

import networkx as nx


def to_mermaid(graph: nx.DiGraph, **kwargs) -> str:
    """Convert graph to Mermaid diagram with colored nodes for human (pastel blue) and assistant (pastel purple) messages, and prefixed node labels (H<n>, A<n>)."""
    lines = ["graph TD"]
    node_styles = []
    for node_id, node_data in graph.nodes(data=True):
        node = node_data["node"]
        content = node.message.content.replace("\n", " ").replace('"', "'")[:60]
        if node.message.role == "human" or node.message.role == "user":
            prefix = f"H{node_id}"
            color = "#a7c7e7"  # pastel blue
            style = f"style {node_id} fill:{color},stroke:#333,stroke-width:1px;"
        elif node.message.role == "assistant":
            prefix = f"A{node_id}"
            color = "#cdb4f6"  # pastel purple
            style = f"style {node_id} fill:{color},stroke:#333,stroke-width:1px;"
        else:
            prefix = f"N{node_id}"
            color = "#e0e0e0"  # default gray
            style = f"style {node_id} fill:{color},stroke:#333,stroke-width:1px;"
        lines.append(f'{node_id}["{prefix}: {content}"]')
        node_styles.append(style)
    for u, v in graph.edges():
        lines.append(f"{u} --> {v}")
    lines.extend(node_styles)
    return "\n".join(lines)
