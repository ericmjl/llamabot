"""Generic document store for LlamaBot.

The interface that we need is implemented here:

- append,
- extend,
- retrieve

ChromaDB is a great default choice because of its simplicity and FOSS nature.
Hence we use it by default.
"""
from pathlib import Path
import chromadb
from hashlib import sha256
from chromadb import QueryResult


class DocumentStore:
    """A document store for LlamaBot that wraps around ChromaDB."""

    def __init__(
        self,
        collection_name: str,
        storage_path: Path = Path.home() / ".llamabot" / "chroma.db",
    ):
        client = chromadb.PersistentClient(path=str(storage_path))
        collection = client.create_collection(collection_name, get_or_create=True)

        self.storage_path = storage_path
        self.client = client
        self.collection = collection
        self.collection_name = collection_name

    def append(self, document: str, metadata: dict = {}):
        """Append a document to the store.

        :param document: The document to append.
        """
        doc_id = sha256(document.encode()).hexdigest()

        self.collection.add(documents=document, ids=doc_id, metadatas=metadata)

    def extend(self, documents: list[str]):
        """Extend the document store.

        :param documents: Iterable of documents.
        """
        for document in documents:
            self.append(document)

    def retrieve(self, query: str, n_results: int = 10) -> list[str]:
        """Retrieve documents from the store.

        :param query: The query to use to retrieve documents.
        """
        results: QueryResult = self.collection.query(
            query_texts=query, n_results=n_results
        )
        return results["documents"][0]

    def reset(self):
        """Reset the document store."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            self.collection_name, get_or_create=True
        )
