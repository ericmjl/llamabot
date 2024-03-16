"""Tests for the document store."""


from pathlib import Path
from llamabot.components.docstore import DocumentStore


def test_document_store():
    """Test the DocumentStore class."""
    docstore = DocumentStore(collection_name="test_collection")
    assert docstore.collection_name == "test_collection"
    docstore.client.delete_collection("test_collection")


def test_add_documents(tmp_path: Path):
    """Test the add_documents method of DocumentStore."""
    # Create a temporary collection for testing
    collection_name = "test_collection"
    storage_path = Path.home() / ".llamabot" / "test_chroma.db"
    docstore = DocumentStore(collection_name=collection_name, storage_path=storage_path)

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

    # Reset the document store
    docstore.reset()

    # Add multiple documents
    document_paths = [tmp_path / "document1.txt", tmp_path / "document2.txt"]
    for i, document_path in enumerate(document_paths):
        document_path.touch()
        document_path.write_text(f"content of document{i+1}")
    docstore.add_documents(document_paths=document_paths)

    # Retrieve the documents from the store
    retrieved_documents = docstore.retrieve("query", n_results=2)

    # Assert that the retrieved documents match the added documents
    assert set(retrieved_documents) == set(
        ["content of document1", "content of document2"]
    )

    # Clean up the temporary collection
    docstore.client.delete_collection(collection_name)
