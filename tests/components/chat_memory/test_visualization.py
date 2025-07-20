"""Tests for visualization functions."""

import networkx as nx
from unittest.mock import Mock
from llamabot.components.messages import user, assistant
from llamabot.components.chat_memory.visualization import to_mermaid


def test_to_mermaid_empty_graph():
    """Test converting empty graph to Mermaid."""
    graph = nx.DiGraph()
    result = to_mermaid(graph)

    assert "graph TD" in result
    assert "H1" not in result  # No nodes should be present


def test_to_mermaid_single_node():
    """Test converting single node to Mermaid."""
    graph = nx.DiGraph()
    msg = user("Hello")
    graph.add_node(1, node=Mock(message=msg, parent_id=None))

    result = to_mermaid(graph)

    assert "graph TD" in result
    assert "H1" in result
    assert "Hello" in result


def test_to_mermaid_linear_conversation():
    """Test converting linear conversation to Mermaid."""
    graph = nx.DiGraph()

    # Create: H1 -> A1 -> H2 -> A2
    h1 = user("Hello")
    a1 = assistant("Hi there!")
    h2 = user("How are you?")
    a2 = assistant("I'm doing well!")

    graph.add_node(1, node=Mock(message=h1, parent_id=None))
    graph.add_node(2, node=Mock(message=a1, parent_id=1))
    graph.add_node(3, node=Mock(message=h2, parent_id=2))
    graph.add_node(4, node=Mock(message=a2, parent_id=3))

    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)

    result = to_mermaid(graph)

    assert "graph TD" in result
    assert "H1" in result
    assert "A1" in result
    assert "H2" in result
    assert "A2" in result
    assert "H1 --> A1" in result
    assert "A1 --> H2" in result
    assert "H2 --> A2" in result


def test_to_mermaid_branching_conversation():
    """Test converting branching conversation to Mermaid."""
    graph = nx.DiGraph()

    # Create: H1 -> A1 -> H2 -> A2
    #                -> H3 -> A3
    h1 = user("Let's talk about Python")
    a1 = assistant("Python is great for data science")
    h2 = user("What about machine learning?")
    a2 = assistant("ML libraries include scikit-learn")
    h3 = user("Tell me about databases")
    a3 = assistant("SQL databases are...")

    graph.add_node(1, node=Mock(message=h1, parent_id=None))
    graph.add_node(2, node=Mock(message=a1, parent_id=1))
    graph.add_node(3, node=Mock(message=h2, parent_id=2))
    graph.add_node(4, node=Mock(message=a2, parent_id=3))
    graph.add_node(5, node=Mock(message=h3, parent_id=2))
    graph.add_node(6, node=Mock(message=a3, parent_id=5))

    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)
    graph.add_edge(2, 5)
    graph.add_edge(5, 6)

    result = to_mermaid(graph)

    assert "graph TD" in result
    assert "H1" in result
    assert "A1" in result
    assert "H2" in result
    assert "A2" in result
    assert "H3" in result
    assert "A3" in result
    assert "H1 --> A1" in result
    assert "A1 --> H2" in result
    assert "A1 --> H3" in result
    assert "H2 --> A2" in result
    assert "H3 --> A3" in result


def test_to_mermaid_with_summaries():
    """Test converting graph with summaries to Mermaid."""
    graph = nx.DiGraph()

    # Create node with summary
    h1 = user("Let's talk about Python programming")
    summary = Mock(title="Python Discussion", summary="User wants to discuss Python")

    graph.add_node(1, node=Mock(message=h1, parent_id=None, summary=summary))

    result = to_mermaid(graph)

    assert "graph TD" in result
    assert "H1" in result
    assert "Python Discussion" in result  # Summary title should be included


def test_to_mermaid_without_summaries():
    """Test converting graph without summaries to Mermaid."""
    graph = nx.DiGraph()

    # Create node without summary
    h1 = user("Let's talk about Python programming")

    graph.add_node(1, node=Mock(message=h1, parent_id=None, summary=None))

    result = to_mermaid(graph)

    assert "graph TD" in result
    assert "H1" in result
    assert "Let's talk about Python programming" in result


def test_to_mermaid_node_labeling():
    """Test that nodes are properly labeled in Mermaid."""
    graph = nx.DiGraph()

    # Create nodes with different roles
    h1 = user("Hello")
    a1 = assistant("Hi there!")

    graph.add_node(1, node=Mock(message=h1, parent_id=None))
    graph.add_node(2, node=Mock(message=a1, parent_id=1))

    result = to_mermaid(graph)

    # Check that human messages are labeled with H and assistant with A
    assert "H1" in result
    assert "A1" in result
    assert "H1 --> A1" in result


def test_to_mermaid_edge_direction():
    """Test that edges show correct direction in Mermaid."""
    graph = nx.DiGraph()

    # Create: H1 -> A1 -> H2
    h1 = user("Hello")
    a1 = assistant("Hi there!")
    h2 = user("How are you?")

    graph.add_node(1, node=Mock(message=h1, parent_id=None))
    graph.add_node(2, node=Mock(message=a1, parent_id=1))
    graph.add_node(3, node=Mock(message=h2, parent_id=2))

    graph.add_edge(1, 2)
    graph.add_edge(2, 3)

    result = to_mermaid(graph)

    # Check edge directions
    assert "H1 --> A1" in result
    assert "A1 --> H2" in result


def test_to_mermaid_message_content():
    """Test that message content is properly included in Mermaid."""
    graph = nx.DiGraph()

    # Create node with specific content
    h1 = user("What is machine learning?")

    graph.add_node(1, node=Mock(message=h1, parent_id=None))

    result = to_mermaid(graph)

    # Check that message content is included
    assert "What is machine learning?" in result


def test_to_mermaid_complex_structure():
    """Test converting complex conversation structure to Mermaid."""
    graph = nx.DiGraph()

    # Create complex structure: H1 -> A1 -> H2 -> A2 -> H4
    #                                    -> H3 -> A3
    h1 = user("Let's discuss AI")
    a1 = assistant("AI is fascinating")
    h2 = user("What about machine learning?")
    a2 = assistant("ML is a subset of AI")
    h3 = user("Tell me about deep learning")
    a3 = assistant("Deep learning uses neural networks")
    h4 = user("What neural networks?")

    graph.add_node(1, node=Mock(message=h1, parent_id=None))
    graph.add_node(2, node=Mock(message=a1, parent_id=1))
    graph.add_node(3, node=Mock(message=h2, parent_id=2))
    graph.add_node(4, node=Mock(message=a2, parent_id=3))
    graph.add_node(5, node=Mock(message=h3, parent_id=2))
    graph.add_node(6, node=Mock(message=a3, parent_id=5))
    graph.add_node(7, node=Mock(message=h4, parent_id=4))

    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)
    graph.add_edge(2, 5)
    graph.add_edge(5, 6)
    graph.add_edge(4, 7)

    result = to_mermaid(graph)

    # Check all nodes are present
    assert "H1" in result
    assert "A1" in result
    assert "H2" in result
    assert "A2" in result
    assert "H3" in result
    assert "A3" in result
    assert "H4" in result

    # Check all edges are present
    assert "H1 --> A1" in result
    assert "A1 --> H2" in result
    assert "H2 --> A2" in result
    assert "A1 --> H3" in result
    assert "H3 --> A3" in result
    assert "A2 --> H4" in result


def test_to_mermaid_with_custom_options():
    """Test converting graph with custom options to Mermaid."""
    graph = nx.DiGraph()

    # Create simple graph
    h1 = user("Hello")
    a1 = assistant("Hi there!")

    graph.add_node(1, node=Mock(message=h1, parent_id=None))
    graph.add_node(2, node=Mock(message=a1, parent_id=1))
    graph.add_edge(1, 2)

    # Test with custom options
    result = to_mermaid(graph, show_summaries=False, max_content_length=10)

    assert "graph TD" in result
    assert "H1" in result
    assert "A1" in result
    # Content should be truncated if longer than max_content_length
