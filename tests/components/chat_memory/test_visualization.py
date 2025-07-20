"""Tests for visualization functions."""

import networkx as nx
from unittest.mock import Mock
from llamabot.components.messages import user, assistant
from llamabot.components.chat_memory.visualization import to_mermaid


def test_to_mermaid_empty_graph():
    """Test converting empty graph to Mermaid."""
    graph = nx.DiGraph()
    result = to_mermaid(graph)
    assert result == "graph TD\n"


def test_to_mermaid_single_node():
    """Test converting single node to Mermaid."""
    graph = nx.DiGraph()

    # Create a proper mock node with message attributes
    h1 = user("Hello")
    mock_node = Mock()
    mock_node.message = h1
    mock_node.parent_id = None
    mock_node.summary = None

    graph.add_node(1, node=mock_node)

    result = to_mermaid(graph)

    assert "graph TD" in result
    assert "H1" in result
    assert "Hello" in result


def test_to_mermaid_linear_conversation():
    """Test converting linear conversation to Mermaid."""
    graph = nx.DiGraph()

    # Create proper mock nodes
    h1 = user("Hello")
    a1 = assistant("Hi there!")
    h2 = user("How are you?")

    mock_node1 = Mock()
    mock_node1.message = h1
    mock_node1.parent_id = None
    mock_node1.summary = None

    mock_node2 = Mock()
    mock_node2.message = a1
    mock_node2.parent_id = 1
    mock_node2.summary = None

    mock_node3 = Mock()
    mock_node3.message = h2
    mock_node3.parent_id = 2
    mock_node3.summary = None

    graph.add_node(1, node=mock_node1)
    graph.add_node(2, node=mock_node2)
    graph.add_node(3, node=mock_node3)

    graph.add_edge(1, 2)
    graph.add_edge(2, 3)

    result = to_mermaid(graph)

    # Check nodes are present
    assert "H1" in result
    assert "A2" in result
    assert "H3" in result

    # Check content is included
    assert "Hello" in result
    assert "Hi there!" in result
    assert "How are you?" in result

    # Check edge directions
    assert "H1 --> A2" in result
    assert "A2 --> H3" in result


def test_to_mermaid_branching_conversation():
    """Test converting branching conversation to Mermaid."""
    graph = nx.DiGraph()

    # Create proper mock nodes
    h1 = user("Let's discuss AI")
    a1 = assistant("AI is fascinating")
    h2 = user("What about machine learning?")
    a2 = assistant("ML is a subset of AI")
    h3 = user("Tell me about deep learning")
    a3 = assistant("Deep learning uses neural networks")

    mock_node1 = Mock()
    mock_node1.message = h1
    mock_node1.parent_id = None
    mock_node1.summary = None

    mock_node2 = Mock()
    mock_node2.message = a1
    mock_node2.parent_id = 1
    mock_node2.summary = None

    mock_node3 = Mock()
    mock_node3.message = h2
    mock_node3.parent_id = 2
    mock_node3.summary = None

    mock_node4 = Mock()
    mock_node4.message = a2
    mock_node4.parent_id = 3
    mock_node4.summary = None

    mock_node5 = Mock()
    mock_node5.message = h3
    mock_node5.parent_id = 2
    mock_node5.summary = None

    mock_node6 = Mock()
    mock_node6.message = a3
    mock_node6.parent_id = 5
    mock_node6.summary = None

    graph.add_node(1, node=mock_node1)
    graph.add_node(2, node=mock_node2)
    graph.add_node(3, node=mock_node3)
    graph.add_node(4, node=mock_node4)
    graph.add_node(5, node=mock_node5)
    graph.add_node(6, node=mock_node6)

    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)
    graph.add_edge(2, 5)
    graph.add_edge(5, 6)

    result = to_mermaid(graph)

    # Check nodes are present
    assert "H1" in result
    assert "A2" in result
    assert "H3" in result
    assert "A4" in result
    assert "H5" in result
    assert "A6" in result

    # Check content is included
    assert "Let's discuss AI" in result
    assert "AI is fascinating" in result
    assert "What about machine learning?" in result
    assert "ML is a subset of AI" in result
    assert "Tell me about deep learning" in result
    assert "Deep learning uses neural networks" in result

    # Check edge directions
    assert "H1 --> A2" in result
    assert "A2 --> H3" in result
    assert "H3 --> A4" in result
    assert "A2 --> H5" in result
    assert "H5 --> A6" in result


def test_to_mermaid_with_summaries():
    """Test converting graph with summaries to Mermaid."""
    graph = nx.DiGraph()

    # Create mock node with summary
    h1 = user("What is machine learning?")
    mock_node = Mock()
    mock_node.message = h1
    mock_node.parent_id = None

    # Create mock summary
    mock_summary = Mock()
    mock_summary.title = "Discussion about ML basics"
    mock_node.summary = mock_summary

    graph.add_node(1, node=mock_node)

    result = to_mermaid(graph, show_summaries=True)

    assert "graph TD" in result
    assert "H1" in result
    assert "Discussion about ML basics" in result


def test_to_mermaid_without_summaries():
    """Test converting graph without summaries to Mermaid."""
    graph = nx.DiGraph()

    # Create mock node with summary
    h1 = user("What is machine learning?")
    mock_node = Mock()
    mock_node.message = h1
    mock_node.parent_id = None

    # Create mock summary
    mock_summary = Mock()
    mock_summary.title = "Discussion about ML basics"
    mock_node.summary = mock_summary

    graph.add_node(1, node=mock_node)

    result = to_mermaid(graph, show_summaries=False)

    assert "graph TD" in result
    assert "H1" in result
    assert "What is machine learning?" in result
    assert "Discussion about ML basics" not in result


def test_to_mermaid_node_labeling():
    """Test that nodes are properly labeled based on message role."""
    graph = nx.DiGraph()

    # Create nodes with different roles
    h1 = user("Hello")
    a1 = assistant("Hi there!")

    mock_node1 = Mock()
    mock_node1.message = h1
    mock_node1.parent_id = None
    mock_node1.summary = None

    mock_node2 = Mock()
    mock_node2.message = a1
    mock_node2.parent_id = 1
    mock_node2.summary = None

    graph.add_node(1, node=mock_node1)
    graph.add_node(2, node=mock_node2)

    result = to_mermaid(graph)

    # Check that user messages get H prefix and assistant messages get A prefix
    assert "H1" in result
    assert "A2" in result


def test_to_mermaid_edge_direction():
    """Test that edges are properly directed."""
    graph = nx.DiGraph()

    h1 = user("Hello")
    a1 = assistant("Hi there!")
    h2 = user("How are you?")

    mock_node1 = Mock()
    mock_node1.message = h1
    mock_node1.parent_id = None
    mock_node1.summary = None

    mock_node2 = Mock()
    mock_node2.message = a1
    mock_node2.parent_id = 1
    mock_node2.summary = None

    mock_node3 = Mock()
    mock_node3.message = h2
    mock_node3.parent_id = 2
    mock_node3.summary = None

    graph.add_node(1, node=mock_node1)
    graph.add_node(2, node=mock_node2)
    graph.add_node(3, node=mock_node3)

    graph.add_edge(1, 2)
    graph.add_edge(2, 3)

    result = to_mermaid(graph)

    # Check edge directions
    assert "H1 --> A2" in result
    assert "A2 --> H3" in result


def test_to_mermaid_message_content():
    """Test that message content is properly included in Mermaid."""
    graph = nx.DiGraph()

    # Create node with specific content
    h1 = user("What is machine learning?")

    mock_node = Mock()
    mock_node.message = h1
    mock_node.parent_id = None
    mock_node.summary = None

    graph.add_node(1, node=mock_node)

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

    mock_node1 = Mock()
    mock_node1.message = h1
    mock_node1.parent_id = None
    mock_node1.summary = None

    mock_node2 = Mock()
    mock_node2.message = a1
    mock_node2.parent_id = 1
    mock_node2.summary = None

    mock_node3 = Mock()
    mock_node3.message = h2
    mock_node3.parent_id = 2
    mock_node3.summary = None

    mock_node4 = Mock()
    mock_node4.message = a2
    mock_node4.parent_id = 3
    mock_node4.summary = None

    mock_node5 = Mock()
    mock_node5.message = h3
    mock_node5.parent_id = 2
    mock_node5.summary = None

    mock_node6 = Mock()
    mock_node6.message = a3
    mock_node6.parent_id = 5
    mock_node6.summary = None

    mock_node7 = Mock()
    mock_node7.message = h4
    mock_node7.parent_id = 4
    mock_node7.summary = None

    graph.add_node(1, node=mock_node1)
    graph.add_node(2, node=mock_node2)
    graph.add_node(3, node=mock_node3)
    graph.add_node(4, node=mock_node4)
    graph.add_node(5, node=mock_node5)
    graph.add_node(6, node=mock_node6)
    graph.add_node(7, node=mock_node7)

    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)
    graph.add_edge(2, 5)
    graph.add_edge(5, 6)
    graph.add_edge(4, 7)

    result = to_mermaid(graph)

    # Check all nodes are present
    assert "H1" in result
    assert "A2" in result
    assert "H3" in result
    assert "A4" in result
    assert "H5" in result
    assert "A6" in result
    assert "H7" in result

    # Check all edges are present
    assert "H1 --> A2" in result
    assert "A2 --> H3" in result
    assert "H3 --> A4" in result
    assert "A2 --> H5" in result
    assert "H5 --> A6" in result
    assert "A4 --> H7" in result


def test_to_mermaid_with_custom_options():
    """Test converting graph with custom options to Mermaid."""
    graph = nx.DiGraph()

    # Create simple graph
    h1 = user("Hello")
    a1 = assistant("Hi there!")

    mock_node1 = Mock()
    mock_node1.message = h1
    mock_node1.parent_id = None
    mock_node1.summary = None

    mock_node2 = Mock()
    mock_node2.message = a1
    mock_node2.parent_id = 1
    mock_node2.summary = None

    graph.add_node(1, node=mock_node1)
    graph.add_node(2, node=mock_node2)
    graph.add_edge(1, 2)

    # Test with custom options
    result = to_mermaid(graph, show_summaries=False, max_content_length=10)

    assert "graph TD" in result
    assert "H1" in result
    assert "A2" in result
    # Content should be truncated if longer than max_content_length
