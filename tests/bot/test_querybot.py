"""Fast unit tests for QueryBot."""

from unittest.mock import MagicMock

from llamabot.bot.querybot import QueryBot
from llamabot.components.messages import AIMessage, HumanMessage
from llamabot.components.docstore import AbstractDocumentStore


def test_querybot_basic_functionality():
    """Test basic functionality of QueryBot with mocked docstore."""
    # Create a mock docstore
    mock_docstore = MagicMock(spec=AbstractDocumentStore)
    mock_docstore.retrieve.return_value = [
        "Mocked document chunk 1",
        "Mocked document chunk 2",
    ]

    # Create QueryBot with mock docstore
    bot = QueryBot(
        system_prompt="You are a helpful assistant.",
        docstore=mock_docstore,
        mock_response="This is a mock response.",
    )

    # Test with string query
    response = bot("How are you doing?")
    assert isinstance(response, AIMessage)
    assert response.content == "This is a mock response."

    # Verify docstore.retrieve was called with the query
    mock_docstore.retrieve.assert_called_once_with("How are you doing?", 20)


def test_querybot_with_memory():
    """Test QueryBot with memory functionality."""
    # Create mock docstore and memory
    mock_docstore = MagicMock(spec=AbstractDocumentStore)
    mock_memory = MagicMock(spec=AbstractDocumentStore)

    mock_docstore.retrieve.return_value = ["Document chunk 1", "Document chunk 2"]
    mock_memory.retrieve.return_value = ["Memory chunk 1", "Memory chunk 2"]

    # Create QueryBot with mock docstore and memory
    bot = QueryBot(
        system_prompt="You are a helpful assistant.",
        docstore=mock_docstore,
        memory=mock_memory,
        mock_response="Response with memory.",
    )

    # Test query
    _ = bot("Tell me what you know")

    # Verify both docstore and memory were used
    mock_docstore.retrieve.assert_called_once()
    mock_memory.retrieve.assert_called_once()

    # Verify memory.append was called with the response content
    mock_memory.append.assert_called_once_with("Response with memory.")


def test_querybot_with_message_input():
    """Test QueryBot accepts HumanMessage as input."""
    # Create mock docstore
    mock_docstore = MagicMock(spec=AbstractDocumentStore)
    mock_docstore.retrieve.return_value = ["Document content"]

    # Create QueryBot
    bot = QueryBot(
        system_prompt="You are a helpful assistant.",
        docstore=mock_docstore,
        mock_response="Response to message input.",
    )

    # Test with HumanMessage
    human_msg = HumanMessage(content="How are you doing?")
    response = bot(human_msg)

    # Verify docstore.retrieve was called with the message content
    mock_docstore.retrieve.assert_called_once_with("How are you doing?", 20)
    assert response.content == "Response to message input."


def test_querybot_custom_n_results():
    """Test QueryBot with custom number of results."""
    # Create mock docstore
    mock_docstore = MagicMock(spec=AbstractDocumentStore)
    mock_docstore.retrieve.return_value = ["Document content"]

    # Create QueryBot
    bot = QueryBot(
        system_prompt="You are a helpful assistant.",
        docstore=mock_docstore,
        mock_response="Custom n_results response.",
    )

    # Test with custom n_results
    response = bot("How are you doing?", n_results=5)

    # Verify docstore.retrieve was called with the custom n_results
    mock_docstore.retrieve.assert_called_once_with("How are you doing?", 5)
    assert response.content == "Custom n_results response."
