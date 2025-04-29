"""Tests for QueryBot."""

from llamabot.bot.querybot import QueryBot
from llamabot.components.messages import HumanMessage
import tempfile
from pathlib import Path


def test_querybot_lancedb():
    """Test initialization of QueryBot with LanceDB."""
    # Create a temporary directory for test files and DB storage
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test file with predictable content
        test_file_path = Path(temp_dir) / "test_document.txt"
        test_content = "This is a test document for QueryBot with LanceDB. " * 20
        test_file_path.write_text(test_content)

        # Create a QueryBot with LanceDB
        bot = QueryBot(
            system_prompt="You are a helpful assistant.",
            collection_name="test_lancedb_collection",
            document_paths=test_file_path,
            mock_response="This is a mock response from LanceDB QueryBot.",
            stream_target="stdout",
            docstore_type="lancedb",
            docstore_kwargs={"storage_path": Path(temp_dir) / "lancedb"},
        )

        # Test basic query
        response = bot("How are you doing?")
        assert response.content == "This is a mock response from LanceDB QueryBot."

        # Clean up
        bot.docstore.reset()


def test_querybot_chromadb():
    """Test initialization of QueryBot with ChromaDB."""
    # Create a temporary directory for test files and DB storage
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test file with predictable content
        test_file_path = Path(temp_dir) / "test_document.txt"
        test_content = "This is a test document for QueryBot with ChromaDB. " * 20
        test_file_path.write_text(test_content)

        # Create a QueryBot with ChromaDB
        bot = QueryBot(
            system_prompt="You are a helpful assistant.",
            collection_name="test_chromadb_collection",
            document_paths=test_file_path,
            mock_response="This is a mock response from ChromaDB QueryBot.",
            stream_target="stdout",
            docstore_type="chromadb",
            docstore_kwargs={"storage_path": Path(temp_dir) / "chromadb"},
        )

        # Test basic query
        response = bot("How are you doing?")
        assert response.content == "This is a mock response from ChromaDB QueryBot."

        # Clean up
        bot.docstore.reset()


def test_querybot_input_types():
    """Test QueryBot supports different input types for the query parameter."""
    # Create a temporary directory for test files and DB storage
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test file
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text(
            "This is a test document for testing different input types."
        )

        # Initialize QueryBot with mock response
        bot = QueryBot(
            system_prompt="You are a helpful assistant.",
            collection_name="test_input_types",
            document_paths=test_file,
            mock_response="This is a mock response.",
            docstore_type="lancedb",
            docstore_kwargs={"storage_path": Path(temp_dir) / "lancedb"},
        )

        # Test with string input
        response_str = bot("How are you doing?")
        assert response_str.content == "This is a mock response."

        # Test with HumanMessage input
        human_msg = HumanMessage(content="How are you doing?")
        response_human = bot(human_msg)
        assert response_human.content == "This is a mock response."

        # Clean up
        bot.docstore.reset()


def test_collection_name_slugification():
    """Test collection names are properly slugified in QueryBot."""
    # Create a temporary directory for test files and DB storage
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test file
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("Test document for collection name slugification.")

        # Test with a collection name that needs slugification
        collection_name = "Test Collection Name With Spaces!"

        # Create QueryBot with LanceDB
        bot_lancedb = QueryBot(
            system_prompt="You are a helpful assistant.",
            collection_name=collection_name,  # This will be slugified internally
            document_paths=test_file,
            mock_response="This is a mock response.",
            docstore_type="lancedb",
            docstore_kwargs={"storage_path": Path(temp_dir) / "lancedb"},
        )

        # Functional test that QueryBot works with slugified name
        response = bot_lancedb("How are you doing?")
        assert response.content == "This is a mock response."

        # Clean up
        bot_lancedb.docstore.reset()

        # Create QueryBot with ChromaDB
        bot_chromadb = QueryBot(
            system_prompt="You are a helpful assistant.",
            collection_name=collection_name,  # This will be slugified internally
            document_paths=test_file,
            mock_response="This is a mock response.",
            docstore_type="chromadb",
            docstore_kwargs={"storage_path": Path(temp_dir) / "chromadb"},
        )

        # Functional test that QueryBot works with slugified name
        response = bot_chromadb("How are you doing?")
        assert response.content == "This is a mock response."

        # Clean up
        bot_chromadb.docstore.reset()


def test_custom_docstore_path_lancedb():
    """Test QueryBot with custom docstore path for LanceDB."""
    # Create a temporary directory for test files and DB storage
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test file
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text(
            "This is a test document for testing custom docstore path."
        )

        # Custom path for LanceDB docstore
        lancedb_path = Path(temp_dir) / "custom_lancedb"

        # Initialize QueryBot with custom docstore path for LanceDB
        bot = QueryBot(
            system_prompt="You are a helpful assistant.",
            collection_name="test_custom_path",
            document_paths=test_file,
            mock_response="This is a mock response.",
            docstore_type="lancedb",
            docstore_kwargs={"storage_path": lancedb_path},
        )

        # Test query functionality
        response = bot("How are you doing?")
        assert response.content == "This is a mock response."

        # Clean up
        bot.docstore.reset()


def test_custom_docstore_path_chromadb():
    """Test QueryBot with custom docstore path for ChromaDB."""
    # Create a temporary directory for test files and DB storage
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test file
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text(
            "This is a test document for testing custom docstore path."
        )

        # Custom path for ChromaDB docstore
        chromadb_path = Path(temp_dir) / "custom_chromadb"

        # Initialize QueryBot with custom docstore path for ChromaDB
        bot = QueryBot(
            system_prompt="You are a helpful assistant.",
            collection_name="test_custom_path_chroma",
            document_paths=test_file,
            mock_response="This is a mock response.",
            docstore_type="chromadb",
            docstore_kwargs={"storage_path": chromadb_path},
        )

        # Test query functionality
        response = bot("How are you doing?")
        assert response.content == "This is a mock response."

        # Clean up
        bot.docstore.reset()


def test_docstore_kwargs():
    """Test QueryBot with additional docstore_kwargs."""
    # Create a temporary directory for test files and DB storage
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test file
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("This is a test document for testing docstore kwargs.")

        # Initialize QueryBot with docstore_kwargs
        bot = QueryBot(
            system_prompt="You are a helpful assistant.",
            collection_name="test_docstore_kwargs",
            document_paths=test_file,
            mock_response="This is a mock response.",
            docstore_type="lancedb",
            docstore_kwargs={
                "storage_path": Path(temp_dir) / "lancedb",
                "auto_create_fts_index": True,
            },
        )

        # Test query functionality
        response = bot("How are you doing?")
        assert response.content == "This is a mock response."

        # Clean up
        bot.docstore.reset()
