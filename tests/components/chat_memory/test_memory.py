"""Tests for the main ChatMemory class."""

import pytest
import networkx as nx
from unittest.mock import Mock, patch
from llamabot.components.messages import user, assistant
from llamabot.components.chat_memory.memory import ChatMemory
from llamabot.components.chat_memory.selectors import (
    LinearNodeSelector,
    LLMNodeSelector,
)


def test_chat_memory_init_default():
    """Test ChatMemory initialization with defaults."""
    memory = ChatMemory()

    assert isinstance(memory.graph, nx.DiGraph)
    assert isinstance(memory.node_selector, LinearNodeSelector)
    assert memory.summarizer is None
    assert memory.context_depth == 5
    assert memory._next_node_id == 1


def test_chat_memory_init_custom():
    """Test ChatMemory initialization with custom parameters."""
    node_selector = LLMNodeSelector(model="gpt-4o-mini")
    summarizer = Mock()

    memory = ChatMemory(
        node_selector=node_selector, summarizer=summarizer, context_depth=10
    )

    assert isinstance(memory.graph, nx.DiGraph)
    assert memory.node_selector == node_selector
    assert memory.summarizer == summarizer
    assert memory.context_depth == 10
    assert memory._next_node_id == 1


def test_chat_memory_init_invalid_context_depth():
    """Test ChatMemory initialization with invalid context depth."""
    with pytest.raises(ValueError, match="context_depth must be non-negative"):
        ChatMemory(context_depth=-1)


def test_chat_memory_threaded_factory():
    """Test ChatMemory.threaded() factory method."""
    memory = ChatMemory.threaded(model="gpt-4o-mini")

    assert isinstance(memory.graph, nx.DiGraph)
    assert isinstance(memory.node_selector, LLMNodeSelector)
    assert memory.node_selector.model == "gpt-4o-mini"
    assert memory.summarizer is not None
    assert memory.context_depth == 5


def test_chat_memory_threaded_factory_custom_params():
    """Test ChatMemory.threaded() with custom parameters."""
    memory = ChatMemory.threaded(model="gpt-4o-mini", context_depth=15)

    assert isinstance(memory.node_selector, LLMNodeSelector)
    assert memory.node_selector.model == "gpt-4o-mini"
    assert memory.context_depth == 15


def test_chat_memory_append_linear():
    """Test appending messages in linear mode."""
    memory = ChatMemory()  # Uses LinearNodeSelector by default

    h1 = user("Hello")
    a1 = assistant("Hi there!")

    memory.append(h1)
    memory.append(a1)

    assert len(memory.graph.nodes()) == 2
    assert memory.graph.nodes[1]["node"].message == h1
    assert memory.graph.nodes[2]["node"].message == a1
    assert memory._next_node_id == 3  # Incremented by 2 (1 for h1, 1 for a1)


def test_chat_memory_append_threaded():
    """Test appending messages in threaded mode."""
    node_selector = Mock()
    node_selector.select_parent.return_value = None  # Root node

    memory = ChatMemory(node_selector=node_selector)

    h1 = user("Hello")
    a1 = assistant("Hi there!")

    memory.append(h1)
    memory.append(a1)

    assert len(memory.graph.nodes()) == 2
    assert memory.graph.nodes[1]["node"].message == h1
    assert memory.graph.nodes[2]["node"].message == a1
    assert memory._next_node_id == 3

    # Should be called twice - once for each message
    assert node_selector.select_parent.call_count == 2
    node_selector.select_parent.assert_any_call(memory.graph, h1)
    node_selector.select_parent.assert_any_call(memory.graph, a1)


def test_chat_memory_append_multiple_turns():
    """Test appending multiple conversation turns."""
    memory = ChatMemory()

    # First turn
    h1 = user("Hello")
    a1 = assistant("Hi there!")
    memory.append(h1)
    memory.append(a1)

    # Second turn
    h2 = user("How are you?")
    a2 = assistant("I'm doing well!")
    memory.append(h2)
    memory.append(a2)

    assert len(memory.graph.nodes()) == 4
    assert memory._next_node_id == 5  # 1, 2, 3, 4 used, next is 5

    # Check connections
    assert memory.graph.has_edge(1, 2)  # H1 -> A1
    assert memory.graph.has_edge(2, 3)  # A1 -> H2
    assert memory.graph.has_edge(3, 4)  # H2 -> A2


def test_chat_memory_retrieve_linear_empty():
    """Test retrieving from empty linear memory."""
    memory = ChatMemory()

    result = memory.retrieve("test query")
    assert result == []


def test_chat_memory_retrieve_linear_with_messages():
    """Test retrieving from linear memory with messages."""
    memory = ChatMemory()

    # Add some messages
    h1 = user("Hello")
    a2 = assistant("Hi there!")
    h3 = user("How are you?")
    a4 = assistant("I'm doing well!")

    memory.append(h1)
    memory.append(a2)
    memory.append(h3)
    memory.append(a4)

    result = memory.retrieve("test query", n_results=3)
    assert len(result) == 3
    # Should return newest messages last
    assert result[0] == a2
    assert result[1] == h3
    assert result[2] == a4


def test_chat_memory_retrieve_threaded():
    """Test retrieving from threaded memory."""
    node_selector = Mock()
    node_selector.select_parent.return_value = None

    memory = ChatMemory(node_selector=node_selector)

    # Add some messages
    h1 = user("Let's talk about Python")
    a1 = assistant("Python is great for data science")
    h2 = user("What about machine learning?")
    a2 = assistant("ML libraries include scikit-learn")

    memory.append(h1)
    memory.append(a1)
    memory.append(h2)
    memory.append(a2)

    with patch(
        "llamabot.components.chat_memory.memory.semantic_search_with_context"
    ) as mock_search:
        mock_search.return_value = [a2, h2, a1]

        result = memory.retrieve("machine learning", n_results=1, context_depth=2)

        assert result == [a2, h2, a1]
        mock_search.assert_called_once_with(memory.graph, "machine learning", 1, 2)


def test_chat_memory_retrieve_custom_context_depth():
    """Test retrieving with custom context depth."""
    memory = ChatMemory()

    # Add some messages
    h1 = user("Hello")
    a1 = assistant("Hi there!")
    memory.append(h1)
    memory.append(a1)

    # Test with custom context depth
    _ = memory.retrieve("test", context_depth=10)
    # Should use custom context depth instead of default, flesh out later.


def test_chat_memory_reset():
    """Test resetting memory."""
    memory = ChatMemory()

    # Add some messages
    h1 = user("Hello")
    a1 = assistant("Hi there!")
    memory.append(h1)
    memory.append(a1)

    assert len(memory.graph.nodes()) == 2

    # Reset memory
    memory.reset()

    assert len(memory.graph.nodes()) == 0
    assert memory._next_node_id == 1


def test_chat_memory_to_mermaid():
    """Test converting memory to Mermaid diagram."""
    memory = ChatMemory()

    # Add some messages
    h1 = user("Hello")
    a1 = assistant("Hi there!")
    memory.append(h1)
    memory.append(a1)

    result = memory.to_mermaid()

    assert "graph TD" in result
    assert '1["H1: Hello"]' in result
    assert '2["A2: Hi there!"]' in result
    assert "1 --> 2" in result
    assert "style 1 fill:#a7c7e7" in result
    assert "style 2 fill:#cdb4f6" in result


def test_chat_memory_save_load():
    """Test saving and loading memory."""
    memory = ChatMemory()

    # Add some messages
    h1 = user("Hello")
    a1 = assistant("Hi there!")
    memory.append(h1)
    memory.append(a1)

    # Save to temporary file
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_path = f.name

    try:
        memory.save(temp_path)

        # Load into new memory instance
        loaded_memory = ChatMemory.load(temp_path)

        assert len(loaded_memory.graph.nodes()) == 2
        assert loaded_memory.graph.nodes[1]["node"].message.content == "Hello"
        assert loaded_memory.graph.nodes[2]["node"].message.content == "Hi there!"

    finally:
        os.unlink(temp_path)


def test_chat_memory_export_json():
    """Test exporting memory to JSON format."""
    memory = ChatMemory()

    # Add some messages
    h1 = user("Hello")
    a1 = assistant("Hi there!")
    memory.append(h1)
    memory.append(a1)

    result = memory.export(format="json")

    import json

    data = json.loads(result)

    assert "version" in data
    assert "metadata" in data
    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) == 2


def test_chat_memory_export_jsonl():
    """Test exporting memory to JSONL format."""
    memory = ChatMemory()

    # Add some messages
    h1 = user("Hello")
    a1 = assistant("Hi there!")
    memory.append(h1)
    memory.append(a1)

    result = memory.export(format="jsonl")

    lines = result.strip().split("\n")
    assert len(lines) == 2  # Two messages

    import json

    for line in lines:
        data = json.loads(line)
        assert "role" in data
        assert "content" in data


def test_chat_memory_export_plain_text():
    """Test exporting memory to plain text format."""
    memory = ChatMemory()

    # Add some messages
    h1 = user("Hello")
    a1 = assistant("Hi there!")
    memory.append(h1)
    memory.append(a1)

    result = memory.export(format="plain_text")

    assert "Hello" in result
    assert "Hi there!" in result
    assert "user:" in result.lower()
    assert "assistant:" in result.lower()


def test_chat_memory_export_invalid_format():
    """Test exporting memory with invalid format."""
    memory = ChatMemory()

    with pytest.raises(ValueError, match="Unsupported export format"):
        memory.export(format="invalid_format")


def test_chat_memory_node_id_increment():
    """Test that node IDs are properly incremented."""
    memory = ChatMemory()

    # First conversation turn
    h1 = user("Hello")
    a1 = assistant("Hi there!")
    memory.append(h1)
    memory.append(a1)

    assert memory._next_node_id == 3  # 1, 2 used, next is 3

    # Second conversation turn
    h2 = user("How are you?")
    a2 = assistant("I'm doing well!")
    memory.append(h2)
    memory.append(a2)

    assert memory._next_node_id == 5  # 1, 2, 3, 4 used, next is 5


def test_chat_memory_append_individual_messages():
    """Test appending individual messages to memory with relationship labels."""
    from llamabot.components.messages import thought, observation

    memory = ChatMemory()

    msg1 = user("Question 1")
    msg2 = thought("Let me think")
    msg3 = observation("Result")
    msg4 = assistant("Answer")

    memory.append(msg1)
    memory.append(msg2)
    memory.append(msg3)
    memory.append(msg4)

    assert len(memory.graph.nodes) == 4
    assert len(memory.graph.edges) == 3

    # Check edge relationships
    edges = list(memory.graph.edges(data=True))
    assert edges[0][2]["relationship"] == "question→response"
    assert edges[1][2]["relationship"] == "response→observation"
    assert edges[2][2]["relationship"] == "observation→response"


def test_chat_memory_threading_detection():
    """Test that memory correctly detects threading mode."""
    # Linear memory
    linear_memory = ChatMemory()
    assert isinstance(linear_memory.node_selector, LinearNodeSelector)

    # Threaded memory
    threaded_memory = ChatMemory.threaded(model="gpt-4o-mini")
    assert isinstance(threaded_memory.node_selector, LLMNodeSelector)
