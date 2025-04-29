"""Tests for the document store."""

from pathlib import Path
from llamabot.components.docstore import (
    BM25DocStore,
    ChromaDBDocStore,
    LanceDBDocStore,
)
from hypothesis import HealthCheck, given, settings, strategies as st
import tempfile
from unittest.mock import patch


def lancedb():
    """Return a LanceDBDocStore."""
    temp_dir = Path(tempfile.mkdtemp())
    store = LanceDBDocStore(table_name="test_lancedb", storage_path=temp_dir)
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


def test_chromadb_append_with_metadata():
    """Test ChromaDBDocStore append method with metadata."""
    # Setup ChromaDBDocStore
    store = ChromaDBDocStore(
        collection_name="test_chromadb_metadata", storage_path=Path("/tmp")
    )
    store.reset()

    # Test document and metadata
    document = "This is a test document with metadata"
    metadata = {"source": "test", "category": "documentation"}

    # Add document with metadata
    store.append(document, metadata=metadata)

    # Retrieve document
    retrieved_docs = store.retrieve("test document", n_results=1)

    # Assert document was stored correctly
    assert retrieved_docs == [document]

    # Get all documents to check metadata
    all_results = store.collection.get()

    # Assert metadata was stored correctly
    # Check if metadatas exists and is not empty
    assert all_results is not None
    assert "metadatas" in all_results
    assert all_results["metadatas"] is not None
    assert len(all_results["metadatas"]) > 0
    assert all_results["metadatas"][0]["source"] == "test"
    assert all_results["metadatas"][0]["category"] == "documentation"

    # Clean up
    store.reset()


def test_chromadb_append_with_embedding():
    """Test ChromaDBDocStore append method with pre-computed embedding."""
    # Setup ChromaDBDocStore
    store = ChromaDBDocStore(
        collection_name="test_chromadb_embedding", storage_path=Path("/tmp")
    )
    store.reset()

    # Test document and mock embedding
    document = "This is a test document with pre-computed embedding"
    mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

    # Mock the embedding function to verify our embedding is used
    with patch.object(store.collection, "add") as mock_add:
        store.append(document, embedding=mock_embedding)
        # Verify the pre-computed embedding was passed to the add method
        _, kwargs = mock_add.call_args
        assert "embeddings" in kwargs
        assert kwargs["embeddings"] == mock_embedding

    # Clean up
    store.reset()


def test_chromadb_extend_with_metadata_and_embeddings():
    """Test ChromaDBDocStore extend method with metadatas and pre-computed embeddings."""
    # Setup ChromaDBDocStore
    store = ChromaDBDocStore(
        collection_name="test_chromadb_batch", storage_path=Path("/tmp")
    )
    store.reset()

    # Test documents, metadata and embeddings
    documents = ["Document 1", "Document 2", "Document 3"]
    metadatas = [
        {"source": "test1", "priority": "high"},
        {"source": "test2", "priority": "medium"},
        {"source": "test3", "priority": "low"},
    ]
    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]

    # Test with mocking to verify parameters
    with patch.object(store.collection, "add") as mock_add:
        store.extend(documents, metadatas=metadatas, embeddings=embeddings)

        # Verify the documents, metadatas and embeddings were passed correctly
        _, kwargs = mock_add.call_args
        assert "documents" in kwargs
        assert kwargs["documents"] == documents
        assert "metadatas" in kwargs
        assert kwargs["metadatas"] == metadatas
        assert "embeddings" in kwargs
        assert kwargs["embeddings"] == embeddings

    # Clean up
    store.reset()


def test_chromadb_append_and_retrieve():
    """Test adding documents with append and retrieving them."""
    # Setup ChromaDBDocStore
    store = ChromaDBDocStore(
        collection_name="test_chromadb_append", storage_path=Path("/tmp")
    )
    store.reset()

    # Add documents with append
    documents = [
        "Python is a programming language with clean syntax.",
        "FastAPI is a modern web framework for building APIs.",
        "ChromaDB is a vector database for storing embeddings.",
    ]

    for doc in documents:
        store.append(doc)

    # Test retrieval with various queries
    python_docs = store.retrieve("Python programming", n_results=1)
    assert len(python_docs) == 1
    assert "Python" in python_docs[0]

    api_docs = store.retrieve("web APIs", n_results=1)
    assert len(api_docs) == 1
    assert "FastAPI" in api_docs[0]

    # Clean up
    store.reset()
