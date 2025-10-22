"""Tests for storage functions."""

import networkx as nx
from unittest.mock import Mock
from llamabot.components.messages import user, assistant
from llamabot.components.chat_memory.storage import append_linear, append_with_threading


def test_append_linear_empty_graph():
    """Test appending to empty graph."""
    graph = nx.DiGraph()
    human_msg = user("Hello")
    assistant_msg = assistant("Hi there!")

    append_linear(graph, human_msg, assistant_msg, next_node_id=1)

    assert len(graph.nodes()) == 2
    assert graph.nodes[1]["node"].message == human_msg
    assert graph.nodes[1]["node"].parent_id is None  # Root node
    assert graph.nodes[2]["node"].message == assistant_msg
    assert graph.nodes[2]["node"].parent_id == 1

    # Check edges
    assert graph.has_edge(1, 2)


def test_append_linear_existing_graph():
    """Test appending to existing graph."""
    graph = nx.DiGraph()

    # Add existing conversation: H1 -> A1
    h1 = user("Hello")
    a1 = assistant("Hi there!")
    graph.add_node(1, node=Mock(message=h1, parent_id=None))
    graph.add_node(2, node=Mock(message=a1, parent_id=1))
    graph.add_edge(1, 2)

    # Append new conversation turn
    h2 = user("How are you?")
    a2 = assistant("I'm doing well!")

    append_linear(graph, h2, a2, next_node_id=3)

    assert len(graph.nodes()) == 4
    assert graph.nodes[3]["node"].message == h2
    assert graph.nodes[3]["node"].parent_id == 2  # Should connect to A1
    assert graph.nodes[4]["node"].message == a2
    assert graph.nodes[4]["node"].parent_id == 3

    # Check edges
    assert graph.has_edge(2, 3)  # A1 -> H2
    assert graph.has_edge(3, 4)  # H2 -> A2


def test_append_linear_find_leaf_assistant_node():
    """Test that linear append connects to the last assistant node."""
    graph = nx.DiGraph()

    # Create conversation: H1 -> A1 -> H2 -> A2 -> H3 -> A3
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

    # Append new conversation turn
    h4 = user("Tell me a joke")
    a4 = assistant("Why did the chicken cross the road?")

    append_linear(graph, h4, a4, next_node_id=7)

    # Should connect to A3 (last assistant node), which is the leaf node
    assert graph.nodes[7]["node"].parent_id == 6
    assert graph.has_edge(6, 7)


def test_append_with_threading_empty_graph():
    """Test threading append to empty graph."""
    graph = nx.DiGraph()
    human_msg = user("Hello")
    assistant_msg = assistant("Hi there!")
    node_selector = Mock()
    node_selector.select_parent.return_value = None  # Root node

    append_with_threading(
        graph, human_msg, assistant_msg, node_selector, next_node_id=1
    )

    assert len(graph.nodes()) == 2
    assert graph.nodes[1]["node"].message == human_msg
    assert graph.nodes[1]["node"].parent_id is None
    assert graph.nodes[2]["node"].message == assistant_msg
    assert graph.nodes[2]["node"].parent_id == 1

    node_selector.select_parent.assert_called_once_with(graph, human_msg)


def test_append_with_threading_existing_graph():
    """Test threading append to existing graph."""
    graph = nx.DiGraph()

    # Add existing conversation: H1 -> A1
    h1 = user("Hello")
    a1 = assistant("Hi there!")
    graph.add_node(1, node=Mock(message=h1, parent_id=None))
    graph.add_node(2, node=Mock(message=a1, parent_id=1))
    graph.add_edge(1, 2)

    # Append new conversation turn
    h2 = user("How are you?")
    a2 = assistant("I'm doing well!")
    node_selector = Mock()
    node_selector.select_parent.return_value = 2  # Connect to A1

    append_with_threading(graph, h2, a2, node_selector, next_node_id=3)

    assert len(graph.nodes()) == 4
    assert graph.nodes[3]["node"].message == h2
    assert graph.nodes[3]["node"].parent_id == 2
    assert graph.nodes[4]["node"].message == a2
    assert graph.nodes[4]["node"].parent_id == 3

    # Check edges
    assert graph.has_edge(2, 3)  # A1 -> H2
    assert graph.has_edge(3, 4)  # H2 -> A2

    node_selector.select_parent.assert_called_once_with(graph, h2)


def test_append_with_threading_node_selector_failure():
    """Test handling node selector failure."""
    graph = nx.DiGraph()

    # Add existing conversation
    h1 = user("Hello")
    a1 = assistant("Hi there!")
    graph.add_node(1, node=Mock(message=h1, parent_id=None))
    graph.add_node(2, node=Mock(message=a1, parent_id=1))
    graph.add_edge(1, 2)

    # Append new conversation turn with failing selector
    h2 = user("How are you?")
    a2 = assistant("I'm doing well!")
    node_selector = Mock()
    node_selector.select_parent.return_value = None  # Selector fails

    append_with_threading(graph, h2, a2, node_selector, next_node_id=3)

    # Should still create nodes, but H2 becomes root
    assert len(graph.nodes()) == 4
    assert graph.nodes[3]["node"].message == h2
    assert graph.nodes[3]["node"].parent_id is None  # Root node
    assert graph.nodes[4]["node"].message == a2
    assert graph.nodes[4]["node"].parent_id == 3


def test_append_with_threading_complex_branching():
    """Test threading append with complex branching."""
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

    # Append new message that should connect to A2
    h4 = user("What other ML libraries?")
    a4 = assistant("There's also TensorFlow and PyTorch")
    node_selector = Mock()
    node_selector.select_parent.return_value = 4  # Connect to A2

    append_with_threading(graph, h4, a4, node_selector, next_node_id=7)

    assert len(graph.nodes()) == 8
    assert graph.nodes[7]["node"].message == h4
    assert graph.nodes[7]["node"].parent_id == 4
    assert graph.nodes[8]["node"].message == a4
    assert graph.nodes[8]["node"].parent_id == 7

    # Check edges
    assert graph.has_edge(4, 7)  # A2 -> H4
    assert graph.has_edge(7, 8)  # H4 -> A4


def test_append_preserves_node_data():
    """Test that append functions preserve node data structure."""
    graph = nx.DiGraph()
    human_msg = user("Hello")
    assistant_msg = assistant("Hi there!")

    append_linear(graph, human_msg, assistant_msg, next_node_id=1)

    # Check node data structure
    human_node = graph.nodes[1]["node"]
    assistant_node = graph.nodes[2]["node"]

    assert hasattr(human_node, "id")
    assert hasattr(human_node, "message")
    assert hasattr(human_node, "parent_id")
    assert hasattr(human_node, "timestamp")

    assert human_node.id == 1
    assert human_node.message == human_msg
    assert human_node.parent_id is None

    assert assistant_node.id == 2
    assert assistant_node.message == assistant_msg
    assert assistant_node.parent_id == 1


def test_append_increments_node_ids():
    """Test that node IDs are properly incremented."""
    graph = nx.DiGraph()

    # First conversation turn
    h1 = user("Hello")
    a1 = assistant("Hi there!")
    append_linear(graph, h1, a1, next_node_id=1)

    # Second conversation turn
    h2 = user("How are you?")
    a2 = assistant("I'm doing well!")
    append_linear(graph, h2, a2, next_node_id=3)

    # Check node IDs
    assert graph.nodes[1]["node"].id == 1
    assert graph.nodes[2]["node"].id == 2
    assert graph.nodes[3]["node"].id == 3
    assert graph.nodes[4]["node"].id == 4


def test_relationship_detection_with_new_message_types():
    """Test relationship detection with ThoughtMessage and ObservationMessage."""
    from llamabot.components.messages import (
        HumanMessage,
        AIMessage,
        thought,
        observation,
    )
    from llamabot.components.chat_memory.storage import _get_simple_relationship

    # Test thought message relationships
    user_msg = HumanMessage(content="Hello")
    thought_msg = thought("I need to think about this")

    # User -> Thought
    relationship = _get_simple_relationship(user_msg, thought_msg)
    assert relationship == "question→response"

    # Thought -> Action (observation)
    observation_msg = observation("Tool executed successfully")
    relationship = _get_simple_relationship(thought_msg, observation_msg)
    assert relationship == "response→observation"

    # Observation -> Response
    response_msg = AIMessage(content="Based on the observation, here's my answer")
    relationship = _get_simple_relationship(observation_msg, response_msg)
    assert relationship == "observation→response"

    # Test that string matching is no longer used
    # Create messages that contain "Observation:" and "Thought:" in content but aren't the right types
    fake_observation = AIMessage(
        content="This contains Observation: but isn't an ObservationMessage"
    )
    fake_thought = AIMessage(
        content="This contains Thought: but isn't a ThoughtMessage"
    )

    # These should use the default role-based relationship, not the special cases
    relationship = _get_simple_relationship(user_msg, fake_observation)
    assert relationship == "question→response"  # Not "question→observation"

    relationship = _get_simple_relationship(fake_thought, response_msg)
    assert relationship == "response→response"  # Not "thought→action"
