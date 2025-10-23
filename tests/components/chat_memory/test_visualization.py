"""Tests for visualization functions."""

import networkx as nx
from unittest.mock import Mock
from llamabot.components.messages import user, assistant
from llamabot.components.chat_memory.visualization import to_mermaid


def test_to_mermaid_empty_graph():
    """Test converting empty graph to Mermaid."""
    graph = nx.DiGraph()
    result = to_mermaid(graph)
    assert result.strip() == "graph TD"


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
    assert '1["H1: Hello"]' in result
    assert "style 1 fill:#a7c7e7" in result


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

    assert '1["H1: Hello"]' in result
    assert '2["A2: Hi there!"]' in result
    assert '3["H3: How are you?"]' in result
    assert "1 --> 2" in result
    assert "2 --> 3" in result
    assert "style 1 fill:#a7c7e7" in result
    assert "style 2 fill:#cdb4f6" in result
    assert "style 3 fill:#a7c7e7" in result


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

    assert '1["H1: Let' in result
    assert '2["A2: AI is fascinating"]' in result
    assert '3["H3: What about machine learning?"]' in result
    assert '4["A4: ML is a subset of AI"]' in result
    assert '5["H5: Tell me about deep learning"]' in result
    assert '6["A6: Deep learning uses neural networks"]' in result
    assert "1 --> 2" in result
    assert "2 --> 3" in result
    assert "2 --> 5" in result
    assert "3 --> 4" in result
    assert "5 --> 6" in result
    assert "style 1 fill:#a7c7e7" in result
    assert "style 2 fill:#cdb4f6" in result
    assert "style 3 fill:#a7c7e7" in result
    assert "style 4 fill:#cdb4f6" in result
    assert "style 5 fill:#a7c7e7" in result
    assert "style 6 fill:#cdb4f6" in result


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
    assert '1["H1: What is machine learning?"]' in result
    assert "style 1 fill:#a7c7e7" in result


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
    assert '1["H1: What is machine learning?"]' in result
    assert "style 1 fill:#a7c7e7" in result


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

    assert '1["H1: Hello"]' in result
    assert '2["A2: Hi there!"]' in result
    assert "style 1 fill:#a7c7e7" in result
    assert "style 2 fill:#cdb4f6" in result


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

    assert "1 --> 2" in result
    assert "2 --> 3" in result


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

    assert '1["H1: Let' in result
    assert '2["A2: AI is fascinating"]' in result
    assert '3["H3: What about machine learning?"]' in result
    assert '4["A4: ML is a subset of AI"]' in result
    assert '5["H5: Tell me about deep learning"]' in result
    assert '6["A6: Deep learning uses neural networks"]' in result
    assert '7["H7: What neural networks?"]' in result
    assert "1 --> 2" in result
    assert "2 --> 3" in result
    assert "2 --> 5" in result
    assert "3 --> 4" in result
    assert "4 --> 7" in result
    assert "5 --> 6" in result
    assert "style 1 fill:#a7c7e7" in result
    assert "style 2 fill:#cdb4f6" in result
    assert "style 3 fill:#a7c7e7" in result
    assert "style 4 fill:#cdb4f6" in result
    assert "style 5 fill:#a7c7e7" in result
    assert "style 6 fill:#cdb4f6" in result
    assert "style 7 fill:#a7c7e7" in result


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
    assert '1["H1: Hello"]' in result
    assert '2["A2: Hi there!"]' in result
    # Content should be truncated if longer than max_content_length


def test_to_mermaid_with_quotes_in_content():
    """Test that double quotes in message content are sanitized."""
    graph = nx.DiGraph()

    # Create node with double quotes in content
    h1 = user('Observation: {"response":"Soil nutrient content"}')

    mock_node = Mock()
    mock_node.message = h1
    mock_node.parent_id = None
    mock_node.summary = None

    graph.add_node(1, node=mock_node)

    result = to_mermaid(graph)

    # Verify double quotes are replaced with single quotes
    assert "{'response':'Soil nutrient content'}" in result
    # Verify the result is valid Mermaid syntax (no unescaped double quotes in content)
    assert '1["H1: Observation: {' in result
