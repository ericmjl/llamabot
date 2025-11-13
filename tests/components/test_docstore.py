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


# Partitioning tests
def test_lancedb_partitioning_append():
    """Test LanceDBDocStore append with partitioning."""
    with tempfile.TemporaryDirectory() as temp_dir:
        store = LanceDBDocStore(
            table_name="test_lancedb_partition_append",
            storage_path=Path(temp_dir),
            enable_partitioning=True,
        )
        store.reset()

        # Add documents to different partitions
        store.append("Python tutorial", partition="tutorials")
        store.append("Python reference", partition="reference")
        store.append("Default document")  # Should go to default partition

        # Retrieve from specific partition
        tutorial_results = store.retrieve(
            "Python", n_results=5, partitions=["tutorials"]
        )
        assert len(tutorial_results) >= 1
        assert "tutorial" in tutorial_results[0].lower()

        # Retrieve from all partitions
        all_results = store.retrieve("Python", n_results=5)
        assert len(all_results) >= 2

        # Clean up
        store.reset()


def test_lancedb_partitioning_extend_single_partition():
    """Test LanceDBDocStore extend with single partition."""
    with tempfile.TemporaryDirectory() as temp_dir:
        store = LanceDBDocStore(
            table_name="test_lancedb_partition_extend_single",
            storage_path=Path(temp_dir),
            enable_partitioning=True,
        )
        store.reset()

        documents = [
            "Python tutorial part 1",
            "Python tutorial part 2",
            "Python tutorial part 3",
        ]
        store.extend(documents, partition="tutorials")

        # Retrieve from tutorials partition
        results = store.retrieve("Python", n_results=5, partitions=["tutorials"])
        assert len(results) >= 3

        # Clean up
        store.reset()


def test_lancedb_partitioning_extend_multiple_partitions():
    """Test LanceDBDocStore extend with multiple partitions."""
    with tempfile.TemporaryDirectory() as temp_dir:
        store = LanceDBDocStore(
            table_name="test_lancedb_partition_extend_multi",
            storage_path=Path(temp_dir),
            enable_partitioning=True,
        )
        store.reset()

        documents = [
            "Python tutorial",
            "Python reference",
            "Python API docs",
        ]
        partitions = ["tutorials", "reference", "api_docs"]
        store.extend(documents, partitions=partitions)

        # Retrieve from specific partition
        tutorial_results = store.retrieve(
            "Python", n_results=5, partitions=["tutorials"]
        )
        assert len(tutorial_results) >= 1
        assert "tutorial" in tutorial_results[0].lower()

        # Retrieve from multiple partitions
        multi_results = store.retrieve(
            "Python", n_results=5, partitions=["tutorials", "reference"]
        )
        assert len(multi_results) >= 2

        # Clean up
        store.reset()


def test_lancedb_partitioning_extend_partitions_length_mismatch():
    """Test that extend raises error when partitions length doesn't match documents."""
    with tempfile.TemporaryDirectory() as temp_dir:
        store = LanceDBDocStore(
            table_name="test_lancedb_partition_extend_error",
            storage_path=Path(temp_dir),
            enable_partitioning=True,
        )
        store.reset()

        documents = ["Doc 1", "Doc 2", "Doc 3"]
        partitions = ["partition1", "partition2"]  # Wrong length

        with pytest.raises(ValueError, match="Length of partitions must match"):
            store.extend(documents, partitions=partitions)

        # Clean up
        store.reset()


def test_lancedb_partitioning_retrieve_all_partitions():
    """Test retrieving from all partitions when partitions=None."""
    with tempfile.TemporaryDirectory() as temp_dir:
        store = LanceDBDocStore(
            table_name="test_lancedb_partition_retrieve_all",
            storage_path=Path(temp_dir),
            enable_partitioning=True,
        )
        store.reset()

        store.append("Tutorial content", partition="tutorials")
        store.append("Reference content", partition="reference")
        store.append("API content", partition="api_docs")

        # Retrieve from all partitions (partitions=None)
        all_results = store.retrieve("content", n_results=10)
        assert len(all_results) >= 3

        # Clean up
        store.reset()


def test_lancedb_partitioning_default_partition():
    """Test that documents without partition go to default partition."""
    with tempfile.TemporaryDirectory() as temp_dir:
        store = LanceDBDocStore(
            table_name="test_lancedb_partition_default",
            storage_path=Path(temp_dir),
            enable_partitioning=True,
            default_partition="default",
        )
        store.reset()

        store.append("Document without partition")

        # Should be in default partition
        default_results = store.retrieve(
            "Document", n_results=5, partitions=["default"]
        )
        assert len(default_results) >= 1

        # Clean up
        store.reset()


def test_lancedb_partitioning_list_partitions():
    """Test list_partitions helper method."""
    with tempfile.TemporaryDirectory() as temp_dir:
        store = LanceDBDocStore(
            table_name="test_lancedb_partition_list",
            storage_path=Path(temp_dir),
            enable_partitioning=True,
        )
        store.reset()

        store.append("Doc 1", partition="tutorials")
        store.append("Doc 2", partition="reference")
        store.append("Doc 3", partition="api_docs")

        partitions = store.list_partitions()
        assert "tutorials" in partitions
        assert "reference" in partitions
        assert "api_docs" in partitions

        # Clean up
        store.reset()


def test_lancedb_partitioning_list_partitions_not_enabled():
    """Test that list_partitions raises error when partitioning not enabled."""
    with tempfile.TemporaryDirectory() as temp_dir:
        store = LanceDBDocStore(
            table_name="test_lancedb_partition_list_error",
            storage_path=Path(temp_dir),
            enable_partitioning=False,
        )
        store.reset()

        with pytest.raises(ValueError, match="Partitioning is not enabled"):
            store.list_partitions()

        # Clean up
        store.reset()


def test_lancedb_partitioning_reset_partition():
    """Test reset_partition helper method."""
    with tempfile.TemporaryDirectory() as temp_dir:
        store = LanceDBDocStore(
            table_name="test_lancedb_partition_reset",
            storage_path=Path(temp_dir),
            enable_partitioning=True,
        )
        store.reset()

        store.append("Tutorial 1", partition="tutorials")
        store.append("Tutorial 2", partition="tutorials")
        store.append("Reference 1", partition="reference")

        # Reset tutorials partition
        store.reset_partition("tutorials")

        # Tutorials should be empty
        tutorial_results = store.retrieve(
            "Tutorial", n_results=5, partitions=["tutorials"]
        )
        assert len(tutorial_results) == 0

        # Reference should still exist
        reference_results = store.retrieve(
            "Reference", n_results=5, partitions=["reference"]
        )
        assert len(reference_results) >= 1

        # Clean up
        store.reset()


def test_lancedb_partitioning_get_partition_count():
    """Test get_partition_count helper method."""
    with tempfile.TemporaryDirectory() as temp_dir:
        store = LanceDBDocStore(
            table_name="test_lancedb_partition_count",
            storage_path=Path(temp_dir),
            enable_partitioning=True,
        )
        store.reset()

        store.append("Tutorial 1", partition="tutorials")
        store.append("Tutorial 2", partition="tutorials")
        store.append("Reference 1", partition="reference")

        assert store.get_partition_count("tutorials") == 2
        assert store.get_partition_count("reference") == 1

        # Clean up
        store.reset()


def test_lancedb_partitioning_schema_migration_error():
    """Test that enabling partitioning on existing table raises error."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create table without partitioning
        store1 = LanceDBDocStore(
            table_name="test_lancedb_partition_migration",
            storage_path=Path(temp_dir),
            enable_partitioning=False,
        )
        store1.reset()
        store1.append("Existing document")

        # Try to open with partitioning enabled
        with pytest.raises(ValueError, match="does not have partition field"):
            LanceDBDocStore(
                table_name="test_lancedb_partition_migration",
                storage_path=Path(temp_dir),
                enable_partitioning=True,
            )

        # Clean up
        store1.reset()


def test_lancedb_partitioning_contains():
    """Test __contains__ method with partitioning enabled."""
    with tempfile.TemporaryDirectory() as temp_dir:
        store = LanceDBDocStore(
            table_name="test_lancedb_partition_contains",
            storage_path=Path(temp_dir),
            enable_partitioning=True,
        )
        store.reset()

        document = "Test document for contains"
        store.append(document, partition="tutorials")

        assert document in store
        assert "Non-existent document" not in store

        # Clean up
        store.reset()
