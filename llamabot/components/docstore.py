"""Generic document store for LlamaBot."""
from pathlib import Path
import chromadb
from hashlib import sha256


class DocumentStore:
    """A document store for LlamaBot that wraps around ChromaDB."""

    def __init__(
        self,
        collection_name: str,
        db_path: Path = Path.home() / ".llamabot" / "chroma.db",
    ):
        client = chromadb.PersistentClient(path=str(db_path))
        collection = client.create_collection(collection_name, get_or_create=True)

        self.db_path = db_path
        self.client = client
        self.collection = collection
        self.collection_name = collection_name

    def append(self, document: str):
        """Append a document to the store.

        :param document: The document to append.
        """
        doc_id = sha256(document.encode()).hexdigest()
        self.collection.add(documents=document, ids=doc_id)

    def retrieve(self, query: str, n_results: int = 10) -> list[str]:
        """Retrieve documents from the store.

        :param query: The query to use to retrieve documents.
        """
        results = self.collection.query(query_texts=query, n_results=n_results)
        return results["documents"]
