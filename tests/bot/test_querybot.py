"""Tests for QueryBot."""

from llamabot.bot.querybot import QueryBot
from llamabot.components.messages import HumanMessage
from hypothesis import HealthCheck, strategies as st, given, settings
import re
import tempfile
from pathlib import Path


@given(
    system_prompt=st.text().filter(lambda x: not re.search(r"(.)\1{3,}", x)),
    collection_name=st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=4, max_size=63
    ).filter(lambda x: not re.search(r"(.)\1{3,}", x)),
    dummy_text=st.text(min_size=400).filter(lambda x: not re.search(r"(.)\1{3,}", x)),
    mock_response=st.text(min_size=4).filter(lambda x: not re.search(r"(.)\1{3,}", x)),
    stream_target=st.one_of(st.just("panel"), st.just("stdout")),
)
@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None,
    max_examples=10,
)
def test_querybot_lancedb(
    tmp_path, system_prompt, collection_name, dummy_text, mock_response, stream_target
):
    """Test initialization of QueryBot with LanceDB."""
    # Create a file with test content
    tempfile = tmp_path / "test.txt"
    tempfile.write_text(dummy_text)

    # Create a temporary directory for LanceDB storage
    with tempfile.TemporaryDirectory() as temp_dir:
        bot = QueryBot(
            system_prompt=system_prompt,
            collection_name=collection_name,
            document_paths=tempfile,
            mock_response=mock_response,
            stream_target=stream_target,
            docstore_type="lancedb",
            docstore_kwargs={"storage_path": Path(temp_dir) / "lancedb"},
        )

        # Test basic query
        response = bot("How are you doing?")
        assert response.content == mock_response

        # Clean up
        bot.docstore.reset()


@given(
    system_prompt=st.text().filter(lambda x: not re.search(r"(.)\1{3,}", x)),
    collection_name=st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=4, max_size=63
    ).filter(lambda x: not re.search(r"(.)\1{3,}", x)),
    dummy_text=st.text(min_size=400).filter(lambda x: not re.search(r"(.)\1{3,}", x)),
    mock_response=st.text(min_size=4).filter(lambda x: not re.search(r"(.)\1{3,}", x)),
    stream_target=st.one_of(st.just("panel"), st.just("stdout")),
)
@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None,
    max_examples=10,
)
def test_querybot_chromadb(
    tmp_path, system_prompt, collection_name, dummy_text, mock_response, stream_target
):
    """Test initialization of QueryBot with ChromaDB."""
    # Create a file with test content
    tempfile = tmp_path / "test.txt"
    tempfile.write_text(dummy_text)

    # Create a temporary directory for ChromaDB storage
    with tempfile.TemporaryDirectory() as temp_dir:
        bot = QueryBot(
            system_prompt=system_prompt,
            collection_name=collection_name,
            document_paths=tempfile,
            mock_response=mock_response,
            stream_target=stream_target,
            docstore_type="chromadb",
            docstore_kwargs={"storage_path": Path(temp_dir) / "chromadb"},
        )

        # Test basic query
        response = bot("How are you doing?")
        assert response.content == mock_response

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
