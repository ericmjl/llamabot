"""Tests for node selection strategies."""

import networkx as nx
from unittest.mock import Mock, patch
from llamabot.components.messages import user, assistant
from llamabot.components.chat_memory.selectors import (
    LinearNodeSelector,
    LLMNodeSelector,
)


def test_linear_selector_select_parent_empty_graph():
    """Test selecting parent when graph is empty."""
    selector = LinearNodeSelector()
    graph = nx.DiGraph()
    message = user("Hello")

    result = selector.select_parent(graph, message)
    assert result is None


def test_linear_selector_select_parent_single_assistant_node():
    """Test selecting parent when only one assistant node exists."""
    selector = LinearNodeSelector()
    graph = nx.DiGraph()

    # Add a single assistant node
    assistant_msg = assistant("Hello there!")
    graph.add_node(1, node=Mock(message=assistant_msg, parent_id=None))

    message = user("How are you?")
    result = selector.select_parent(graph, message)
    assert result == 1


def test_linear_selector_select_parent_multiple_nodes():
    """Test selecting the leaf assistant node from multiple nodes."""
    selector = LinearNodeSelector()
    graph = nx.DiGraph()

    # Create a simple conversation: H1 -> A1 -> H2 -> A2
    h1 = user("Hello")
    a1 = assistant("Hi there!")
    h2 = user("How are you?")
    a2 = assistant("I'm doing well!")

    graph.add_node(1, node=Mock(message=h1, parent_id=None))
    graph.add_node(2, node=Mock(message=a1, parent_id=1))
    graph.add_node(3, node=Mock(message=h2, parent_id=2))
    graph.add_node(4, node=Mock(message=a2, parent_id=3))

    # Add edges
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)

    message = user("What's the weather?")
    result = selector.select_parent(graph, message)
    assert result == 4  # Should select A2 (leaf assistant node)


def test_linear_selector_select_parent_no_assistant_nodes():
    """Test selecting parent when no assistant nodes exist."""
    selector = LinearNodeSelector()
    graph = nx.DiGraph()

    # Add only human nodes
    h1 = user("Hello")
    h2 = user("How are you?")

    graph.add_node(1, node=Mock(message=h1, parent_id=None))
    graph.add_node(2, node=Mock(message=h2, parent_id=1))
    graph.add_edge(1, 2)

    message = user("What's up?")
    result = selector.select_parent(graph, message)
    assert result is None


def test_linear_selector_select_parent_only_human_leaf():
    """Test selecting parent when leaf node is human."""
    selector = LinearNodeSelector()
    graph = nx.DiGraph()

    # Create: H1 -> A1 -> H2 (H2 is leaf, but human)
    h1 = user("Hello")
    a1 = assistant("Hi there!")
    h2 = user("How are you?")

    graph.add_node(1, node=Mock(message=h1, parent_id=None))
    graph.add_node(2, node=Mock(message=a1, parent_id=1))
    graph.add_node(3, node=Mock(message=h2, parent_id=2))

    graph.add_edge(1, 2)
    graph.add_edge(2, 3)

    message = user("What's the weather?")
    result = selector.select_parent(graph, message)
    assert result == 2  # Should select A1 (last assistant node)


def test_llm_selector_init():
    """Test LLMNodeSelector initialization."""
    selector = LLMNodeSelector(model="gpt-4o-mini")
    assert selector.model == "gpt-4o-mini"


@patch("llamabot.components.chat_memory.selectors.get_llm")
def test_llm_selector_select_parent_empty_graph(mock_get_llm):
    """Test selecting parent when graph is empty."""
    selector = LLMNodeSelector(model="gpt-4o-mini")
    graph = nx.DiGraph()
    message = user("Hello")

    result = selector.select_parent(graph, message)
    assert result is None
    mock_get_llm.assert_not_called()


@patch("llamabot.components.chat_memory.selectors.get_llm")
def test_llm_selector_select_parent_single_candidate(mock_get_llm):
    """Test selecting parent when only one candidate exists."""
    selector = LLMNodeSelector(model="gpt-4o-mini")
    graph = nx.DiGraph()

    # Add a single assistant node
    assistant_msg = assistant("Hello there!")
    graph.add_node(1, node=Mock(message=assistant_msg, parent_id=None))

    message = user("How are you?")
    result = selector.select_parent(graph, message)
    assert result == 1
    mock_get_llm.assert_not_called()  # No LLM call needed for single candidate


@patch("llamabot.components.chat_memory.selectors.get_llm")
def test_llm_selector_select_parent_multiple_candidates(mock_get_llm):
    """Test selecting parent with multiple candidates using LLM."""
    # Mock LLM response
    mock_llm = Mock()
    mock_llm.return_value = "2"  # LLM selects node 2
    mock_get_llm.return_value = mock_llm

    selector = LLMNodeSelector(model="gpt-4o-mini")
    graph = nx.DiGraph()

    # Create multiple assistant nodes
    a1 = assistant("Hello there!")
    a2 = assistant("I'm doing well!")
    a3 = assistant("Nice to meet you!")

    graph.add_node(1, node=Mock(message=a1, parent_id=None))
    graph.add_node(2, node=Mock(message=a2, parent_id=1))
    graph.add_node(3, node=Mock(message=a3, parent_id=2))

    message = user("What's the weather?")
    result = selector.select_parent(graph, message)
    assert result == 2
    mock_get_llm.assert_called_once_with("gpt-4o-mini")


@patch("llamabot.components.chat_memory.selectors.get_llm")
def test_llm_selector_select_parent_llm_invalid_response(mock_get_llm):
    """Test handling invalid LLM response."""
    # Mock LLM response that's not a valid node ID
    mock_llm = Mock()
    mock_llm.return_value = "invalid"
    mock_get_llm.return_value = mock_llm

    selector = LLMNodeSelector(model="gpt-4o-mini")
    graph = nx.DiGraph()

    # Create multiple assistant nodes
    a1 = assistant("Hello there!")
    a2 = assistant("I'm doing well!")

    graph.add_node(1, node=Mock(message=a1, parent_id=None))
    graph.add_node(2, node=Mock(message=a2, parent_id=1))

    message = user("What's the weather?")
    result = selector.select_parent(graph, message)
    # Should fall back to most recent valid node
    assert result == 2


@patch("llamabot.components.chat_memory.selectors.get_llm")
def test_llm_selector_select_parent_llm_exception(mock_get_llm):
    """Test handling LLM exceptions."""
    # Mock LLM that raises an exception
    mock_llm = Mock()
    mock_llm.side_effect = Exception("API error")
    mock_get_llm.return_value = mock_llm

    selector = LLMNodeSelector(model="gpt-4o-mini")
    graph = nx.DiGraph()

    # Create multiple assistant nodes
    a1 = assistant("Hello there!")
    a2 = assistant("I'm doing well!")

    graph.add_node(1, node=Mock(message=a1, parent_id=None))
    graph.add_node(2, node=Mock(message=a2, parent_id=1))

    message = user("What's the weather?")
    result = selector.select_parent(graph, message)
    # Should fall back to most recent valid node
    assert result == 2


def test_llm_selector_get_candidate_nodes():
    """Test getting candidate assistant nodes."""
    selector = LLMNodeSelector(model="gpt-4o-mini")
    graph = nx.DiGraph()

    # Create mixed nodes: H1 -> A1 -> H2 -> A2
    h1 = user("Hello")
    a1 = assistant("Hi there!")
    h2 = user("How are you?")
    a2 = assistant("I'm doing well!")

    graph.add_node(1, node=Mock(message=h1, parent_id=None))
    graph.add_node(2, node=Mock(message=a1, parent_id=1))
    graph.add_node(3, node=Mock(message=h2, parent_id=2))
    graph.add_node(4, node=Mock(message=a2, parent_id=3))

    candidates = selector._get_candidate_nodes(graph)
    assert candidates == [2, 4]  # Only assistant nodes


def test_llm_selector_validate_node_selection():
    """Test validating node selection."""
    selector = LLMNodeSelector(model="gpt-4o-mini")
    graph = nx.DiGraph()

    # Add assistant nodes
    a1 = assistant("Hello there!")
    a2 = assistant("I'm doing well!")

    graph.add_node(1, node=Mock(message=a1, parent_id=None))
    graph.add_node(2, node=Mock(message=a2, parent_id=1))

    # Valid selection
    assert selector._validate_node_selection(graph, "2") == 2
    assert selector._validate_node_selection(graph, 2) == 2

    # Invalid selections
    assert selector._validate_node_selection(graph, "3") is None  # Non-existent
    assert selector._validate_node_selection(graph, "1") is None  # Not assistant
    assert selector._validate_node_selection(graph, "invalid") is None  # Not int
