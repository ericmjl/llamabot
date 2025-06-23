"""Generic document store for LlamaBot.

The interface that we need is implemented here:

- append,
- extend,
- retrieve

LanceDB is a great default choice because of its simplicity and FOSS nature.
Hence we use it by default.
"""

from pathlib import Path

import slugify


class AbstractDocumentStore:
    """Abstract document store for LlamaBot."""

    def __init__(self):
        raise NotImplementedError()

    def append(self, document: str):
        """Append a document to the store.

        :param document: The document to append.
        :raises NotImplementedError: If the document store is not implemented.
        """
        raise NotImplementedError()

    def extend(self, documents: list[str]):
        """Extend a list of documents to the store.

        :param documents: The documents to append.
        :raises NotImplementedError: If the document store is not implemented.
        """
        raise NotImplementedError()

    def retrieve(self, query: str, n_results: int = 10) -> list[str]:
        """Retrieve a list of documents from the store.

        :param query: The query to make of the documents.
        :param n_results: The number of results to retrieve.
        :raises NotImplementedError: If the document store is not implemented.
        """
        raise NotImplementedError()

    def reset(self):
        """Reset the document store.

        :raises NotImplementedError: If the document store is not implemented.
        """
        raise NotImplementedError()


# NOTE: 11 May 2025 -- ChromaDB's dependency chain
# is introducing difficulties for me here.
# I have opted to remove support for it temporarily until later notice.
#
# class ChromaDBDocStore(AbstractDocumentStore):
#     """A document store for LlamaBot that wraps around ChromaDB."""

#     def __init__(
#         self,
#         collection_name: str,
#         storage_path: Path = Path.home() / ".llamabot" / "chroma.db",
#     ):
#         try:
#             import chromadb
#         except ImportError:
#             raise ImportError(
#                 "ChromaDB is required for ChromaDBDocStore. "
#                 "Please `pip install llamabot[rag]` to use the ChromaDB document store."
#             )

#         collection_name = slugify.slugify(collection_name, separator="-")

#         client = chromadb.PersistentClient(path=str(storage_path))
#         collection = client.create_collection(collection_name, get_or_create=True)

#         self.storage_path = storage_path
#         self.client = client
#         self.collection = collection
#         self.collection_name = collection_name
#         self.existing_records = collection.get()

#     def __post_add_documents__(self):
#         """Execute code after adding documents to the store."""
#         pass

#     def append(
#         self,
#         document: str,
#         metadata: Optional[dict] = None,
#         embedding: Optional[list[float]] = None,
#     ):
#         """Append a document to the store.

#         By default, the documents will be automatically embedded
#         using the default embedder that is set in the ChromaDB client.
#         However, there may be cases where we want to pre-compute the embeddings
#         and pass them into the store, e.g. when adding documents in parallel.
#         In this case, we can pass the pre-computed embeddings into the store.
#         Use `append` to do single adds, and `extend` to do bulk adds.

#         :param document: The document to append.
#         :param metadata: The metadata to append.
#         :param embedding: The embedding to append, optional
#         """
#         doc_id = sha256(document.encode()).hexdigest()
#         self.collection.add(
#             documents=document,
#             ids=doc_id,
#             metadatas=metadata,
#             embeddings=embedding,
#         )

#     def extend(
#         self,
#         documents: list[str],
#         metadatas: Optional[list[dict]] = None,
#         embeddings: Optional[list[list[float]]] = None,
#     ):
#         """Extend the document store.

#         This is effectively a bulk add of documents.
#         See the docstring for `append` for more details.

#         :param documents: Iterable of documents.
#         :param metadatas: Iterable of metadatas.
#         :param embeddings: Iterable of pre-computed embeddings.
#         """
#         # Compute doc_id
#         ids = [sha256(doc.encode()).hexdigest() for doc in documents]

#         # Check that the lengths of the lists are the same
#         if metadatas is not None and len(documents) != len(metadatas):
#             raise ValueError(
#                 "The lengths of the documents and metadatas must be the same."
#             )
#         if embeddings is not None and len(documents) != len(embeddings):
#             raise ValueError(
#                 "The lengths of the documents and embeddings must be the same."
#             )

#         self.collection.add(
#             ids=ids,
#             documents=documents,
#             metadatas=metadatas,
#             embeddings=embeddings,
#         )

#     def retrieve(self, query: str, n_results: int = 10) -> list[str]:
#         """Retrieve documents from the store.

#         :param query: The query to use to retrieve documents.
#         """
#         try:
#             import chromadb
#         except ImportError:
#             raise ImportError(
#                 "ChromaDB is required for ChromaDBDocStore. "
#                 "Please `pip install llamabot[rag]` to use the ChromaDB document store."
#             )

#         # Use Vectordb to get documents.
#         results: chromadb.QueryResult = self.collection.query(
#             query_texts=query, n_results=n_results
#         )
#         vectordb_documents: list[str] = results["documents"][0]
#         return vectordb_documents

#     def reset(self):
#         """Reset the document store."""
#         self.client.delete_collection(self.collection_name)
#         self.collection = self.client.create_collection(
#             self.collection_name, get_or_create=True
#         )


class LanceDBDocStore(AbstractDocumentStore):
    """A document store for LlamaBot that wraps around LanceDB."""

    def __init__(
        self,
        table_name: str,
        storage_path: Path = Path.home() / ".llamabot" / "lancedb",
        embedding_registry: str = "sentence-transformers",
        embedding_model: str = "minishlab/potion-base-8M",
        auto_create_fts_index: bool = True,
    ):
        try:
            import lancedb
            from lancedb.embeddings import EmbeddingFunctionRegistry, get_registry
            from lancedb.rerankers.colbert import ColbertReranker
            from lancedb.pydantic import LanceModel, Vector
        except ImportError:
            raise ImportError(
                "LanceDB is required for LanceDBDocStore. "
                "Please `pip install llamabot[rag]` to use the LanceDB document store."
            )

        registry: EmbeddingFunctionRegistry = get_registry()
        self.embedding_func = registry.get(embedding_registry).create(
            name=embedding_model
        )

        class DocstoreEntry(LanceModel):
            """LanceDB DocumentStore Entry."""

            document: str = self.embedding_func.SourceField()
            vector: Vector(self.embedding_func.ndims()) = (
                self.embedding_func.VectorField()
            )

        table_name = slugify.slugify(table_name, separator="-")

        self.schema = DocstoreEntry
        self.table_name = table_name
        storage_path.mkdir(parents=True, exist_ok=True)  # Ensure storage path exists
        self.db = lancedb.connect(storage_path)

        try:
            self.table = self.db.open_table(table_name)
        except Exception:
            self.table = self.db.create_table(table_name, schema=self.schema)

        try:
            self.existing_records = [
                item.document
                for item in self.table.search().limit(None).to_pydantic(self.schema)
            ]
        except ValueError:
            self.existing_records = []

        if auto_create_fts_index:
            # NOTE: 23 Jun 2025 -- set use_tantivy=False to avoid
            # the following error:
            #
            #   ValueError: field_names must be a string when use_tantivy=False
            self.table.create_fts_index(
                field_names="document", replace=True, use_tantivy=False
            )

        self.reranker = ColbertReranker(column="document")

    def __contains__(self, other: str) -> bool:
        """Returns boolean whether the 'other' document is in the store.

        :param other: The document to search for.
        :return: True if the document is in the store, False otherwise.
        """
        all_items = self.table.search().limit(None).to_pydantic(self.schema)
        texts = set([item.document for item in all_items])
        return other in texts

    def append(
        self,
        document: str,
    ):
        """Append a document to the store.

        :param document: The document to append.
        """
        # Avoid duplication of documents in LanceDB.

        # Create document entry, only include document and vector fields
        # Completely ignore metadata
        document_to_add = {"document": document}

        if document not in self.existing_records:
            self.table.add([document_to_add])
            self.existing_records.append(document)

        # Ensure FTS index exists
        self.table.optimize()

    def extend(
        self,
        documents: list[str],
    ):
        """Extend a list of documents to the store.

        :param documents: The documents to append.
        """
        # Completely ignore metadata parameter

        stuff_to_add = []
        for i, doc in enumerate(documents):
            # Skip if document already exists
            if doc in self.existing_records:
                continue

            # Create document entry with document text only
            entry = {"document": doc}
            stuff_to_add.append(entry)

        # Use add instead of merge_insert to avoid schema conflicts
        if stuff_to_add:
            self.table.add(stuff_to_add)
            self.existing_records.extend(documents)
            # Ensure FTS index exists
            self.table.optimize()

    def retrieve(self, query: str, n_results: int = 10) -> list[str]:
        """Retrieve a list of documents from the store.

        :param query: The query to use to retrieve documents.
        :param n_results: The number of results to retrieve.
        :return: A list of documents.
        """
        results = (
            self.table.search(query, query_type="auto")
            .rerank(self.reranker)
            .limit(n_results)
            .to_pydantic(self.schema)
        )
        return [r.document for r in results]

    def reset(self):
        """Reset the document store."""
        self.db.drop_table(self.table_name)
        self.table = self.db.create_table(self.table_name, schema=self.schema)
        self.existing_records = []


class BM25DocStore(AbstractDocumentStore):
    """In-memory BM25-based retrieval document store."""

    def __init__(self):
        self.documents: list = []

    def __post_add_documents__(self):
        """Execute code after adding documents to the store."""
        pass

    def append(self, document: str):
        """Append a document to the store.

        :param document: The document to append.
        """
        self.documents.append(document)

    def extend(self, documents: list[str]):
        """Extend a list of documents to the store.

        :param documents: The documents to append.
        """
        self.documents.extend(documents)

    def retrieve(self, query: str, n_results: int = 10) -> list[str]:
        """Retrieve documents from the store.

        :param query: The query to use to retrieve documents.
        :param n_results: The number of results to retrieve.
        :return: A list of documents.
        """

        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank_bm25 is required for BM25DocStore. "
                "Please install it with `pip install llamabot[rag]`"
            )

        # Use BM25 to get documents.
        tokenized_docs = [doc.split() for doc in self.documents]
        search_engine = BM25Okapi(tokenized_docs)
        docs: list[str] = search_engine.get_top_n(
            query.split(), self.documents, n=n_results
        )
        return docs

    def reset(self):
        """Reset the document store."""
        self.documents = []
