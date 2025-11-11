"""Visualization functions for PocketFlow flows."""


def flow_to_mermaid(flow) -> str:
    """Convert a PocketFlow Flow object to a Mermaid diagram.

    Based on PocketFlow source: nodes store connections in `successors` dict
    and Flow uses `start_node` attribute (not `start`).

    :param flow: A PocketFlow Flow object
    :return: Mermaid diagram string
    """
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
                            f'|"{action}"|' if action != "default" and action else ""
                        )
                        lines.append(f"{node_id} -->{label} {target_id}")
                        # Recursively add edges from target
                        add_edges(target)

    add_edges(start_node)

    lines.extend(node_styles)
    return "\n".join(lines)
