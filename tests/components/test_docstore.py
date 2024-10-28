"""Tests for the document store."""

from pathlib import Path
from llamabot.components.docstore import (
    BM25DocStore,
    ChromaDBDocStore,
    LanceDBDocStore,
    SQLiteVecDocStore,
)
from hypothesis import HealthCheck, given, settings, strategies as st


def lancedb():
    """Return a LanceDBDocStore."""
    store = LanceDBDocStore(table_name="test_lancedb", storage_path=Path("/tmp"))
    store.reset()
    return store


def chromadb():
    """Return a ChromaDBDocStore."""
    store = ChromaDBDocStore(collection_name="test_chromadb", storage_path=Path("/tmp"))
    store.reset()
    return store


def bm25():
    """Return a BM25DocStore."""
    return BM25DocStore()


def sqlitevec():
    """Return a SQLiteVecDocStore."""
    store = SQLiteVecDocStore(
        db_path=Path("/tmp/test_sqlite_vec.db"), table_name="test_documents"
    )
    store.reset()
    return store


docstore_strategies = [
    st.just(lancedb()),
    st.just(chromadb()),
    st.just(bm25()),
    st.just(sqlitevec()),
]


@given(docstore=st.one_of(docstore_strategies))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
def test_add_documents(tmp_path: Path, docstore):
    """Test the add_documents method of DocumentStore."""
    # Add a single document
    document_path = tmp_path / "document.txt"
    document_path.touch()
    document_path.write_text("content of the document")
    docstore.add_documents(document_paths=document_path)

    # Retrieve the document from the store
    retrieved_documents = docstore.retrieve("query", n_results=1)

    # Assert that the retrieved document matches the added document
    assert retrieved_documents == ["content of the document"]

    # Add multiple documents
    document_paths = [tmp_path / "document1.txt", tmp_path / "document2.txt"]
    for i, document_path in enumerate(document_paths):
        document_path.touch()
        document_path.write_text(f"content of document{i+1}")
    docstore.add_documents(document_paths=document_paths)

    # Retrieve the documents from the store
    retrieved_documents = docstore.retrieve("document1", n_results=1)

    # Assert that the retrieved documents match the added documents
    assert set(retrieved_documents) == set(["content of document1"])

    # Clean up the temporary collection
    docstore.reset()


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


def test_sqlitevec_semantic_search():
    """Test semantic search capabilities of SQLiteVecDocStore."""
    store = SQLiteVecDocStore(
        db_path=Path("/tmp/test_semantic_search.db"),
        table_name="test_semantic_search",
        embedding_model="all-MiniLM-L6-v2",
    )

    # Example documents - movie plots
    documents = [
        "A young wizard discovers he has magical powers and attends a school of witchcraft and wizardry.",
        "Two hobbits journey to destroy a powerful ring in a volcano while being pursued by dark forces.",
        "A group of superheroes must work together to save Earth from an alien invasion.",
        "A criminal mastermind plants ideas in people's minds through their dreams.",
        "A computer programmer discovers that reality is a simulation created by machines.",
    ]

    # Add documents to the store
    for doc in documents:
        store.append(doc)

    # Test fantasy-related content
    results = store.retrieve("magic and fantasy", n_results=2)
    assert len(results) == 2
    assert any("wizard" in doc.lower() for doc in results)
    assert any("hobbits" in doc.lower() for doc in results)

    # Test sci-fi content
    results = store.retrieve("science fiction and computers", n_results=2)
    assert len(results) == 2
    assert any("simulation" in doc.lower() for doc in results)
    assert any("programmer" in doc.lower() for doc in results)

    # Test action content
    results = store.retrieve("action and fighting", n_results=2)
    assert len(results) == 2
    assert any("superheroes" in doc.lower() for doc in results)

    # Test adding more documents
    more_documents = [
        "A team of explorers travel through a wormhole in search of a new habitable planet.",
        "A archaeologist searches for ancient artifacts while avoiding traps and rivals.",
    ]
    store.extend(more_documents)

    # Test search across all documents
    results = store.retrieve("space exploration and science", n_results=2)
    assert len(results) == 2
    assert any("wormhole" in doc.lower() for doc in results)

    # Clean up
    store.reset()
