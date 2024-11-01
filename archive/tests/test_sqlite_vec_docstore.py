"""Tests for the SQLiteVecDocStore."""

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
