"""Tests for the document store."""

import pytest
from pathlib import Path
from llamabot.components.docstore import (
    BM25DocStore,
    LanceDBDocStore,
)
import tempfile


def lancedb():
    """Return a LanceDBDocStore."""
    temp_dir = Path(tempfile.mkdtemp())
    store = LanceDBDocStore(table_name="test_lancedb", storage_path=temp_dir)
    store.reset()
    return store


# def chromadb():
#     """Return a ChromaDBDocStore."""
#     store = ChromaDBDocStore(collection_name="test_chromadb", storage_path=Path("/tmp"))
#     store.reset()
#     return store


def bm25():
    """Return a BM25DocStore."""
    return BM25DocStore()


# docstore_strategies = [
#     st.just(lancedb()),
#     # st.just(chromadb()),
#     st.just(bm25()),
# ]


# def test_chromadb_append_with_metadata():
#     """Test ChromaDBDocStore append method with metadata."""
#     # Setup ChromaDBDocStore with tempdir
#     with tempfile.TemporaryDirectory() as temp_dir:
#         store = ChromaDBDocStore(
#             collection_name="test_chromadb_metadata", storage_path=Path(temp_dir)
#         )
#         store.reset()

#         # Test document and metadata
#         document = "This is a test document with metadata"
#         metadata = {"source": "test", "category": "documentation"}

#         # Add document with metadata
#         store.append(document, metadata=metadata)

#         # Retrieve document
#         retrieved_docs = store.retrieve("test document", n_results=1)

#         # Assert document was stored correctly
#         assert retrieved_docs == [document]

#         # Get all documents to check metadata
#         all_results = store.collection.get()

#         # Assert metadata was stored correctly
#         # Check if metadatas exists and is not empty
#         assert all_results is not None
#         assert "metadatas" in all_results
#         assert all_results["metadatas"] is not None
#         assert len(all_results["metadatas"]) > 0
#         assert all_results["metadatas"][0]["source"] == "test"
#         assert all_results["metadatas"][0]["category"] == "documentation"

#         # Clean up
#         store.reset()


# def test_chromadb_append_with_embedding():
#     """Test ChromaDBDocStore append method with pre-computed embedding."""
#     # Setup ChromaDBDocStore with tempdir
#     with tempfile.TemporaryDirectory() as temp_dir:
#         store = ChromaDBDocStore(
#             collection_name="test_chromadb_embedding", storage_path=Path(temp_dir)
#         )
#         store.reset()

#         # Test document and embedding
#         document = "This is a test document with pre-computed embedding"
#         embedding = [0.1] * 384  # 384-dimensional embedding vector

#         # Add document with pre-computed embedding
#         store.append(document, embedding=embedding)

#         # Retrieve document to verify it was stored
#         retrieved_docs = store.retrieve("test document", n_results=1)
#         assert retrieved_docs == [document]

#         # Clean up
#         store.reset()


# def test_chromadb_extend_with_metadata_and_embeddings():
#     """Test ChromaDBDocStore extend method with metadatas and pre-computed embeddings."""
#     # Setup ChromaDBDocStore
#     with tempfile.TemporaryDirectory() as temp_dir:
#         store = ChromaDBDocStore(
#             collection_name="test_chromadb_batch", storage_path=Path(temp_dir)
#         )
#         store.reset()

#         # Test documents, metadata and embeddings
#         documents = ["Document 1", "Document 2", "Document 3"]
#         metadatas = [
#             {"source": "test1", "priority": "high"},
#             {"source": "test2", "priority": "medium"},
#             {"source": "test3", "priority": "low"},
#         ]
#         embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]

#         # Add documents with metadata and embeddings
#         store.extend(documents, metadatas=metadatas, embeddings=embeddings)

#         # Get all documents to check they were added correctly
#         all_results = store.collection.get()

#         # Verify documents were stored
#         assert len(all_results["documents"]) == 3
#         assert set(all_results["documents"]) == set(documents)

#         # Verify metadata was stored correctly
#         assert all_results["metadatas"] is not None

#         # Each document should have its corresponding metadata
#         for i, doc in enumerate(all_results["documents"]):
#             doc_index = documents.index(doc)
#             metadata = all_results["metadatas"][i]
#             expected_metadata = metadatas[doc_index]

#             assert metadata["source"] == expected_metadata["source"]
#             assert metadata["priority"] == expected_metadata["priority"]

#         # Clean up
#         store.reset()


# def test_chromadb_append_and_retrieve():
#     """Test adding documents with append and retrieving them."""
#     # Setup ChromaDBDocStore with tempdir
#     with tempfile.TemporaryDirectory() as temp_dir:
#         store = ChromaDBDocStore(
#             collection_name="test_chromadb_append", storage_path=Path(temp_dir)
#         )
#         store.reset()

#         # Add documents with append
#         documents = [
#             "Python is a programming language with clean syntax.",
#             "FastAPI is a modern web framework for building APIs.",
#             "ChromaDB is a vector database for storing embeddings.",
#         ]

#         for doc in documents:
#             store.append(doc)

#         # Test retrieval with various queries
#         python_docs = store.retrieve("Python programming", n_results=1)
#         assert len(python_docs) == 1
#         assert "Python" in python_docs[0]

#         api_docs = store.retrieve("web APIs", n_results=1)
#         assert len(api_docs) == 1
#         assert "FastAPI" in api_docs[0]

#         # Clean up
#         store.reset()


@pytest.mark.xfail(reason="LanceDB file system issues - to be fixed later")
def test_lancedb_append():
    """Test LanceDBDocStore append method."""
    # Setup LanceDBDocStore with tempdir
    with tempfile.TemporaryDirectory() as temp_dir:
        store = LanceDBDocStore(
            table_name="test_lancedb_append", storage_path=Path(temp_dir)
        )
        store.reset()

        # Test document
        document = "This is a test document for LanceDB append"

        # Add document
        store.append(document)

        # Retrieve document to verify it was stored
        retrieved_docs = store.retrieve("test document", n_results=1)
        assert len(retrieved_docs) == 1
        assert retrieved_docs[0] == document

        # Clean up
        store.reset()


@pytest.mark.xfail(reason="LanceDB file system issues - to be fixed later")
def test_lancedb_extend():
    """Test LanceDBDocStore extend method."""
    # Setup LanceDBDocStore with tempdir
    with tempfile.TemporaryDirectory() as temp_dir:
        store = LanceDBDocStore(
            table_name="test_lancedb_extend", storage_path=Path(temp_dir)
        )
        store.reset()

        # Test documents
        documents = [
            "Python is a programming language with clean syntax.",
            "FastAPI is a modern web framework for building APIs.",
            "LanceDB is a vector database for storing embeddings.",
        ]

        # Add documents with extend
        store.extend(documents)

        # Test retrieval with various queries
        python_docs = store.retrieve("Python programming", n_results=1)
        assert len(python_docs) == 1
        assert "Python" in python_docs[0]

        api_docs = store.retrieve("web APIs", n_results=1)
        assert len(api_docs) == 1
        assert "FastAPI" in api_docs[0]

        # Verify all documents were added
        all_docs = store.retrieve("embeddings OR syntax OR APIs", n_results=3)
        assert len(all_docs) == 3
        assert set(all_docs) == set(documents)

        # Clean up
        store.reset()


@pytest.mark.xfail(reason="LanceDB file system issues - to be fixed later")
def test_lancedb_append_avoid_duplicates():
    """Test LanceDBDocStore append method avoids duplicates."""
    # Setup LanceDBDocStore with tempdir
    with tempfile.TemporaryDirectory() as temp_dir:
        store = LanceDBDocStore(
            table_name="test_lancedb_duplicates", storage_path=Path(temp_dir)
        )
        store.reset()

        # Test document
        document = "This is a unique test document for LanceDB"

        # Add document twice
        store.append(document)
        store.append(document)  # Should not add duplicate

        # Check the document was only added once
        # This is implementation-specific, but we have the existing_records tracking
        assert document in store.existing_records
        assert store.existing_records.count(document) == 1

        # Clean up
        store.reset()
