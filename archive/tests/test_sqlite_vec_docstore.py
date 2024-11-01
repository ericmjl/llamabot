"""Tests for the SQLiteVecDocStore."""

import tempfile
from pathlib import Path
from archive.sqlite_vec_docstore import SQLiteVecDocStore


def sqlitevec():
    """Return a SQLiteVecDocStore."""
    store = SQLiteVecDocStore(
        db_path=Path("/tmp/test_sqlite_vec.db"), table_name="test_documents"
    )
    store.reset()
    return store


def test_sqlitevec_specific():
    """Test SQLiteVecDocStore specific functionality."""
    store = sqlitevec()

    # Test document addition and retrieval
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "A quick brown dog jumps over the lazy fox",
        "The lazy fox sleeps while the quick brown dog works",
    ]

    # Add documents one by one
    for doc in documents:
        store.append(doc)

    # Test semantic search capabilities
    results = store.retrieve("fox and dog", n_results=2)
    assert len(results) == 2
    assert all(isinstance(doc, str) for doc in results)

    # Test exact document exists
    results = store.retrieve(documents[0], n_results=1)
    assert results[0] == documents[0]

    # Test extend functionality
    new_documents = ["New document about animals", "Another document about jumping"]
    store.extend(new_documents)

    # Test retrieval after extend
    results = store.retrieve("jumping animals", n_results=1)
    assert len(results) == 1

    # Test reset functionality
    store.reset()
    results = store.retrieve("fox", n_results=1)
    assert len(results) == 0

    # Clean up
    store.reset()


def test_querybot_with_sqlitevec():
    """Test QueryBot with SQLiteVecDocStore."""
    from llamabot.bot.querybot import QueryBot

    # Create a temporary file with some test content
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(
            """
        A young wizard discovers he has magical powers and attends a school of witchcraft and wizardry.
        Two hobbits journey to destroy a powerful ring in a volcano while being pursued by dark forces.
        A group of superheroes must work together to save Earth from an alien invasion.
        """
        )
        temp_path = Path(f.name)

    try:
        # Initialize QueryBot with SQLiteVecDocStore
        bot = QueryBot(
            system_prompt="You are a helpful assistant that answers questions about movies.",
            collection_name="test_movies",
            document_paths=temp_path,
            docstore_type="sqlitevec",  # Specify SQLiteVecDocStore
            mock_response="This is a test response",  # For testing purposes
            stream_target="stdout",
        )

        # Test querying
        response = bot("Tell me about fantasy stories")
        assert response is not None

        # Test with a different query
        response = bot("What superhero content do you have?")
        assert response is not None

        # Clean up
        bot.docstore.reset()

    finally:
        # Clean up temporary file
        temp_path.unlink()
