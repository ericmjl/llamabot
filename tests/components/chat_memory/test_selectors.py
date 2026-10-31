"""Tests for node selection strategies."""

import networkx as nx
from unittest.mock import Mock
from llamabot.components.messages import user, assistant
from llamabot.components.chat_memory.selectors import (
    LinearNodeSelector,
    LLMNodeSelector,
    NodeSelection,
)
from unittest.mock import patch


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


def test_llm_selector_select_parent_empty_graph():
    """Test LLM selector with empty graph."""
    selector = LLMNodeSelector(model="gpt-4o-mini")
    graph = nx.DiGraph()

    message = user("Hello")
    result = selector.select_parent(graph, message)
    assert result is None


def test_llm_selector_select_parent_single_candidate():
    """Test LLM selector with single candidate."""
    selector = LLMNodeSelector(model="gpt-4o-mini")
    graph = nx.DiGraph()

    # Create single assistant node
    a1 = assistant("Hello there!")

    graph.add_node(1, node=Mock(message=a1, parent_id=None))

    message = user("What's the weather?")
    result = selector.select_parent(graph, message)
    # Should return the single candidate without calling LLM
    assert result == 1


def selector_bot_stand_in(call_result=None, call_error: Exception | None = None):
    """Build a StructuredBot stand-in that does not contact a model provider.

    :param call_result: Value returned when the stand-in is called.
    :param call_error: Exception raised when the stand-in is called.
    :return: Mock used as the object returned by ``StructuredBot(...)``.
    """
    bot = Mock()
    if call_error is not None:
        bot.side_effect = call_error
    else:
        bot.return_value = call_result
    return bot


def test_llm_selector_select_parent_multiple_candidates():
    """A valid model choice is returned even when it is not the newest node."""
    selector = LLMNodeSelector(model="gpt-4o-mini")
    graph = nx.DiGraph()

    # Create multiple assistant nodes
    a1 = assistant("Hello there!")
    a2 = assistant("I'm doing well!")
    a3 = assistant("Weather is nice!")

    graph.add_node(1, node=Mock(message=a1, parent_id=None))
    graph.add_node(2, node=Mock(message=a2, parent_id=1))
    graph.add_node(3, node=Mock(message=a3, parent_id=2))

    message = user("What's the weather?")
    # Node 1 is valid and is not the fallback (the most recent candidate).
    selection = NodeSelection(
        selected_node_id=1, reasoning="The greeting is the relevant parent."
    )
    stand_in = selector_bot_stand_in(call_result=selection)
    with patch(
        "llamabot.bot.structuredbot.StructuredBot",
        return_value=stand_in,
    ):
        result = selector.select_parent(graph, message)
    assert result == 1
    stand_in.assert_called_once()


def test_llm_selector_select_parent_llm_invalid_response():
    """Test handling invalid LLM response."""
    selector = LLMNodeSelector(model="gpt-4o-mini")
    graph = nx.DiGraph()

    # Create multiple assistant nodes
    a1 = assistant("Hello there!")
    a2 = assistant("I'm doing well!")

    graph.add_node(1, node=Mock(message=a1, parent_id=None))
    graph.add_node(2, node=Mock(message=a2, parent_id=1))

    message = user("What's the weather?")
    # A dict has no selected_node_id attribute, so selection fails closed.
    stand_in = selector_bot_stand_in(
        call_result={"selected_node_id": 999, "reasoning": "Invalid node"}
    )
    with patch(
        "llamabot.bot.structuredbot.StructuredBot",
        return_value=stand_in,
    ):
        result = selector.select_parent(graph, message)
        # Should fall back to most recent valid node
        assert result == 2


def test_llm_selector_select_parent_llm_exception():
    """Test handling LLM exceptions."""
    selector = LLMNodeSelector(model="gpt-4o-mini")
    graph = nx.DiGraph()

    # Create multiple assistant nodes
    a1 = assistant("Hello there!")
    a2 = assistant("I'm doing well!")

    graph.add_node(1, node=Mock(message=a1, parent_id=None))
    graph.add_node(2, node=Mock(message=a2, parent_id=1))

    message = user("What's the weather?")
    stand_in = selector_bot_stand_in(call_error=Exception("LLM error"))
    with patch(
        "llamabot.bot.structuredbot.StructuredBot",
        return_value=stand_in,
    ):
        result = selector.select_parent(graph, message)
        # Should fall back to most recent valid node
        assert result == 2


def test_llm_selector_get_candidate_nodes():
    """Test getting candidate assistant nodes."""
    from llamabot.components.chat_memory.selectors import get_candidate_nodes

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

    candidates = get_candidate_nodes(graph)
    assert candidates == [2, 4]  # Only assistant nodes


def test_llm_selector_validate_node_selection():
    """Test validating node selection."""
    from llamabot.components.chat_memory.selectors import validate_node_selection

    graph = nx.DiGraph()

    # Add nodes: node 1 is a user, node 2 is an assistant
    h1 = user("Hello there!")
    a2 = assistant("I'm doing well!")

    graph.add_node(1, node=Mock(message=h1, parent_id=None))
    graph.add_node(2, node=Mock(message=a2, parent_id=1))

    # Valid selection
    assert validate_node_selection(graph, 2, [1, 2]) is True
    assert validate_node_selection(graph, 2, [2]) is True

    # Invalid selections
    assert validate_node_selection(graph, 3, [1, 2]) is False  # Non-existent
    assert validate_node_selection(graph, 1, [1, 2]) is False  # Not assistant
