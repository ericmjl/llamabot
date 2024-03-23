"""Tests for the document store."""


from pathlib import Path
from llamabot.components.docstore import BM25DocStore, ChromaDBDocStore, LanceDBDocStore
from hypothesis import HealthCheck, given, settings, strategies as st, reproduce_failure


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


docstore_strategies = [
    st.just(lancedb()),
    st.just(chromadb()),
    st.just(bm25()),
]


@reproduce_failure("6.99.6", b"AAA=")
@given(docstore=st.one_of(docstore_strategies))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
def test_add_documents(tmp_path: Path, docstore):
    """Test the add_documents method of DocumentStore."""
    # Add a single document
    # document_path = Path("path/to/document.txt")
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
