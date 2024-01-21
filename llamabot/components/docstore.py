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
from llamabot.doc_processor import magic_load_doc, split_document
from tqdm.auto import tqdm


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
        self.existing_records = collection.get()

    def append(self, document: str, metadata: dict = {}):
        """Append a document to the store.

        :param document: The document to append.
        """
        doc_id = sha256(document.encode()).hexdigest()

        add_kwargs = dict(
            documents=document,
            ids=doc_id,
        )
        if metadata:
            add_kwargs["metadatas"] = metadata

        self.collection.add(**add_kwargs)

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

    def add_documents(
        self,
        document_paths: Path | list[Path],
        chunk_size: int = 2_000,
        chunk_overlap: int = 500,
    ):
        """Add documents to the QueryBot DocumentStore."""
        if isinstance(document_paths, Path):
            document_paths = [document_paths]

        for document_path in tqdm(document_paths):
            document = magic_load_doc(document_path)
            splitted_document = split_document(
                document, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            chunks_to_add = [
                doc.text
                for doc in splitted_document
                if doc.text not in self.existing_records["documents"]
            ]
            self.extend(chunks_to_add)
