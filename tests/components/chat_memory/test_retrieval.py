"""Tests for retrieval functions."""

import networkx as nx
from unittest.mock import Mock, patch
from llamabot.components.messages import user, assistant
from llamabot.components.chat_memory.retrieval import (
    get_recent_messages,
    semantic_search_with_context,
    traverse_thread_path,
)


def test_get_recent_messages_empty_graph():
    """Test getting recent messages from empty graph."""
    graph = nx.DiGraph()
    result = get_recent_messages(graph, n_results=5)
    assert result == []


def test_get_recent_messages_single_message():
    """Test getting recent messages from single message."""
    graph = nx.DiGraph()
    msg = user("Hello")
    graph.add_node(1, node=Mock(message=msg, parent_id=None))

    result = get_recent_messages(graph, n_results=5)
    assert len(result) == 1
    assert result[0] == msg


def test_get_recent_messages_multiple_messages():
    """Test getting recent messages from multiple messages."""
    graph = nx.DiGraph()

    # Create: H1 -> A2 -> H3 -> A4
    h1 = user("Hello")
    a2 = assistant("Hi there!")
    h3 = user("How are you?")
    a4 = assistant("I'm doing well!")

    graph.add_node(1, node=Mock(message=h1, parent_id=None))
    graph.add_node(2, node=Mock(message=a2, parent_id=1))
    graph.add_node(3, node=Mock(message=h3, parent_id=2))
    graph.add_node(4, node=Mock(message=a4, parent_id=3))

    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)

    result = get_recent_messages(graph, n_results=3)
    assert len(result) == 3
    # Should return newest messages last
    assert result[0] == a2
    assert result[1] == h3
    assert result[2] == a4


def test_get_recent_messages_limited_results():
    """Test getting limited number of recent messages."""
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

    result = get_recent_messages(graph, n_results=2)
    assert len(result) == 2
    assert result[0] == h2
    assert result[1] == a2


def test_traverse_thread_path_empty():
    """Test traversing thread path from non-existent node."""
    graph = nx.DiGraph()
    result = traverse_thread_path(graph, node_id=1, depth=5)
    assert result == []


def test_traverse_thread_path_single_node():
    """Test traversing thread path from single node."""
    graph = nx.DiGraph()
    msg = user("Hello")
    graph.add_node(1, node=Mock(message=msg, parent_id=None))

    result = traverse_thread_path(graph, node_id=1, depth=5)
    assert len(result) == 1
    assert result[0] == msg


def test_traverse_thread_path_linear():
    """Test traversing linear thread path."""
    graph = nx.DiGraph()

    # Create: H1 -> A2 -> H3 -> A4
    h1 = user("Hello")
    a2 = assistant("Hi there!")
    h3 = user("How are you?")
    a4 = assistant("I'm doing well!")

    graph.add_node(1, node=Mock(message=h1, parent_id=None))
    graph.add_node(2, node=Mock(message=a2, parent_id=1))
    graph.add_node(3, node=Mock(message=h3, parent_id=2))
    graph.add_node(4, node=Mock(message=a4, parent_id=3))

    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)

    # Traverse from A4 with depth 3
    result = traverse_thread_path(graph, node_id=4, depth=3)
    assert len(result) == 3
    assert result[0] == a2  # Grandparent (most ancient in this path)
    assert result[1] == h3  # Parent
    assert result[2] == a4  # Start node (most recent)


def test_traverse_thread_path_limited_depth():
    """Test traversing thread path with limited depth."""
    graph = nx.DiGraph()

    # Create: H1 -> A1 -> H2 -> A2 -> H3 -> A3
    h1 = user("Hello")
    a1 = assistant("Hi there!")
    h2 = user("How are you?")
    a2 = assistant("I'm doing well!")
    h3 = user("What's the weather?")
    a3 = assistant("It's sunny!")

    graph.add_node(1, node=Mock(message=h1, parent_id=None))
    graph.add_node(2, node=Mock(message=a1, parent_id=1))
    graph.add_node(3, node=Mock(message=h2, parent_id=2))
    graph.add_node(4, node=Mock(message=a2, parent_id=3))
    graph.add_node(5, node=Mock(message=h3, parent_id=4))
    graph.add_node(6, node=Mock(message=a3, parent_id=5))

    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)
    graph.add_edge(4, 5)
    graph.add_edge(5, 6)

    # Traverse from A3 with depth 2
    result = traverse_thread_path(graph, node_id=6, depth=2)
    assert len(result) == 2
    assert result[0] == h3  # Parent (most ancient in this path)
    assert result[1] == a3  # Start node (most recent)


def test_traverse_thread_path_branching():
    """Test traversing thread path with branching."""
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

    # Traverse from A2
    result = traverse_thread_path(graph, node_id=4, depth=3)
    assert len(result) == 3
    assert result[0] == a1  # Grandparent (most ancient in this path)
    assert result[1] == h2  # Parent
    assert result[2] == a2  # Start node (most recent)

    # Traverse from A3
    result = traverse_thread_path(graph, node_id=6, depth=3)
    assert len(result) == 3
    assert result[0] == a1  # Grandparent (most ancient in this path)
    assert result[1] == h3  # Parent
    assert result[2] == a3  # Start node (most recent)


@patch("llamabot.components.chat_memory.retrieval.bm25_search")
def test_semantic_search_with_context(mock_bm25_search):
    """Test semantic search with context retrieval."""
    graph = nx.DiGraph()

    # Create: H1 -> A1 -> H2 -> A2
    h1 = user("Let's talk about Python")
    a1 = assistant("Python is great for data science")
    h2 = user("What about machine learning?")
    a2 = assistant("ML libraries include scikit-learn")

    graph.add_node(1, node=Mock(message=h1, parent_id=None))
    graph.add_node(2, node=Mock(message=a1, parent_id=1))
    graph.add_node(3, node=Mock(message=h2, parent_id=2))
    graph.add_node(4, node=Mock(message=a2, parent_id=3))

    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)

    # Mock BM25 search to return node 4 (A2)
    mock_bm25_search.return_value = [4]

    query = "machine learning libraries"
    result = semantic_search_with_context(graph, query, n_results=1, context_depth=2)

    # Should return A2 + context (H2, A1)
    assert len(result) == 3
    assert a2 in result
    assert h2 in result
    assert a1 in result


@patch("llamabot.components.chat_memory.retrieval.bm25_search")
def test_semantic_search_with_context_multiple_results(mock_bm25_search):
    """Test semantic search with multiple results."""
    graph = nx.DiGraph()

    # Create: H1 -> A2 -> H3 -> A4 -> H5 -> A6
    h1 = user("Let's talk about Python")
    a2 = assistant("Python is great for data science")
    h3 = user("What about machine learning?")
    a4 = assistant("ML libraries include scikit-learn")
    h5 = user("What other ML libraries?")
    a6 = assistant("There's also TensorFlow and PyTorch")

    graph.add_node(1, node=Mock(message=h1, parent_id=None))
    graph.add_node(2, node=Mock(message=a2, parent_id=1))
    graph.add_node(3, node=Mock(message=h3, parent_id=2))
    graph.add_node(4, node=Mock(message=a4, parent_id=3))
    graph.add_node(5, node=Mock(message=h5, parent_id=4))
    graph.add_node(6, node=Mock(message=a6, parent_id=5))

    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)
    graph.add_edge(4, 5)
    graph.add_edge(5, 6)

    # Mock BM25 search to return nodes 4 and 6
    mock_bm25_search.return_value = [4, 6]

    query = "machine learning libraries"
    result = semantic_search_with_context(graph, query, n_results=2, context_depth=1)

    # Should return A4 + context, A6 + context
    assert len(result) == 4
    assert a4 in result
    assert h5 in result
    assert a6 in result
    assert h3 in result  # h3 is the context for a4 (most ancient)


def test_semantic_search_with_context_empty_graph():
    """semantic_search_with_context returns [] for an empty graph."""
    graph = nx.DiGraph()
    result = semantic_search_with_context(
        graph, "anything", n_results=5, context_depth=3
    )
    assert result == []
