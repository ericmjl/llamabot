"""Visualization functions for PocketFlow flows."""


def calculate_graph_metrics(start_node, node_id_map):
    """Calculate graph metrics to determine optimal layout direction.

    Uses BFS to assign levels to nodes, handling cycles by taking minimum level.

    :param start_node: The starting node of the graph
    :param node_id_map: Dictionary mapping nodes to their IDs
    :return: Tuple of (max_depth, max_width_at_level)
    """
    if start_node not in node_id_map:
        return (0, 0)

    # Use BFS to assign levels, handling cycles by taking minimum level
    node_levels = {}  # node -> minimum level
    queue = [(start_node, 0)]

    while queue:
        node, level = queue.pop(0)
        if node not in node_id_map:
            continue

        # Update level if we found a shorter path (handles cycles)
        if node not in node_levels or level < node_levels[node]:
            node_levels[node] = level
            # Process successors
            successors = getattr(node, "successors", {})
            if isinstance(successors, dict):
                for target in successors.values():
                    if target and target in node_id_map:
                        queue.append((target, level + 1))

    # Group nodes by level
    levels = {}
    for node, level in node_levels.items():
        if level not in levels:
            levels[level] = set()
        levels[level].add(node)

    max_depth = max(levels.keys()) if levels else 0
    max_width = max(len(nodes) for nodes in levels.values()) if levels else 0

    return (max_depth, max_width)


def identify_terminal_nodes(node_id_map):
    """Identify terminal nodes (nodes with no successors).

    :param node_id_map: Dictionary mapping nodes to their IDs
    :return: Set of node IDs that are terminal nodes
    """
    terminal_node_ids = set()

    for node, node_id in node_id_map.items():
        successors = getattr(node, "successors", {})
        # Check if node has no successors or only empty/None successors
        if not successors or not any(
            target for target in successors.values() if target
        ):
            terminal_node_ids.add(node_id)

    return terminal_node_ids


def flow_to_mermaid(flow) -> str:
    """Convert a PocketFlow Flow object to a Mermaid diagram.

    Based on PocketFlow source: nodes store connections in `successors` dict
    and Flow uses `start_node` attribute (not `start`).

    Automatically determines graph direction (TD vs LR) based on graph structure.
    Terminal nodes are colored differently (green) from regular nodes (blue).

    :param flow: A PocketFlow Flow object
    :return: Mermaid diagram string
    """
    # PocketFlow uses `start_node` attribute, not `start`
    if not hasattr(flow, "start_node") or flow.start_node is None:
        return 'graph TD\nA["Empty Flow"]'

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

    # Calculate graph metrics to determine direction
    max_depth, max_width = calculate_graph_metrics(start_node, node_id_map)
    # Use LR if graph is wider than it is deep, otherwise use TD
    graph_direction = "LR" if max_width > max_depth else "TD"
    lines = [f"graph {graph_direction}"]
    node_styles = []

    # Identify terminal nodes
    terminal_node_ids = identify_terminal_nodes(node_id_map)

    # Generate node definitions
    for node, node_id in node_id_map.items():
        # Check if node has a 'name' property (FuncNode instances)
        if hasattr(node, "name"):
            node_name = node.name
        else:
            node_name = node.__class__.__name__
        lines.append(f'{node_id}["{node_name}"]')
        # Style nodes: green for terminal nodes, light blue for regular nodes
        if node_id in terminal_node_ids:
            style = f"style {node_id} fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px;"
        else:
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
