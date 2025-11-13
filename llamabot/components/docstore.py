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
    """A document store for LlamaBot that wraps around LanceDB.

    Supports optional partitioning to organize documents into logical groups.
    When partitioning is enabled, documents can be assigned to partitions and
    retrieval can be filtered by partition. This is useful for organizing
    documents by source, category, or any other logical grouping.

    :param table_name: Name of the table to create or open.
    :param storage_path: Path to the LanceDB storage directory.
    :param embedding_registry: Registry name for the embedding function.
    :param embedding_model: Model name for the embedding function.
    :param auto_create_fts_index: Whether to automatically create a full-text
        search index on the document field.
    :param enable_partitioning: If True, enables partitioning support. When enabled,
        documents must be assigned to partitions and retrieval can be filtered by
        partition. Cannot be enabled on existing tables without partition field.
    :param default_partition: Default partition name to use when partition is not
        specified and partitioning is enabled.
    :raises ValueError: If partitioning is enabled on an existing table that
        doesn't have a partition field. Use `.reset()` to recreate the table.
    """

    def __init__(
        self,
        table_name: str,
        storage_path: Path = Path.home() / ".llamabot" / "lancedb",
        embedding_registry: str = "sentence-transformers",
        embedding_model: str = "minishlab/potion-base-8M",
        auto_create_fts_index: bool = True,
        enable_partitioning: bool = False,
        default_partition: str = "default",
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
            name=embedding_model, trust_remote_code=True
        )

        # Store partitioning configuration
        self.enable_partitioning = enable_partitioning
        self.default_partition = default_partition

        # Track operations for automatic optimization
        # LanceDB recommends optimizing after 20+ operations or 100k+ records
        self._operation_count = 0
        self._optimize_threshold = 20

        # Conditionally add partition field to schema
        if enable_partitioning:

            class DocstoreEntry(LanceModel):
                """LanceDB DocumentStore Entry with partitioning."""

                document: str = self.embedding_func.SourceField()
                vector: Vector(self.embedding_func.ndims()) = (
                    self.embedding_func.VectorField()
                )
                partition: str

        else:

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

        # Check if table exists and handle schema migration
        table_exists = table_name in self.db.table_names()
        if table_exists and enable_partitioning:
            # Try to open the table to check if it has partition field
            try:
                existing_table = self.db.open_table(table_name)
                # Check schema by trying to query with partition field
                # This will fail if partition field doesn't exist
                try:
                    existing_table.search().where("partition = 'test'").limit(
                        1
                    ).to_list()
                except (Exception, ValueError) as e:
                    # Check if error is related to missing partition field
                    error_msg = str(e).lower()
                    if (
                        "partition" in error_msg
                        or "column" in error_msg
                        or "field" in error_msg
                    ):
                        # Table exists but doesn't have partition field
                        raise ValueError(
                            f"Table '{table_name}' exists but does not have partition field. "
                            "To enable partitioning, please reset the table using `.reset()` "
                            "or create a new table with a different name."
                        ) from e
                    # Re-raise if it's a different error
                    raise
                self.table = existing_table
            except ValueError as e:
                if "does not have partition field" in str(e):
                    raise
                # Table might not exist or other error, create it
                self.table = self.db.create_table(table_name, schema=self.schema)
        else:
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
            # NOTE: Using LanceDB's native FTS implementation instead of Tantivy
            # for better performance and incremental indexing capabilities
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
        partition: str | None = None,
    ):
        """Append a document to the store.

        :param document: The document to append.
        :param partition: Optional partition name. If partitioning is enabled and
            not provided, defaults to `default_partition`. Ignored if partitioning
            is disabled.
        """
        # Avoid duplication of documents in LanceDB.

        # Create document entry
        if self.enable_partitioning:
            partition_name = (
                partition if partition is not None else self.default_partition
            )
            document_to_add = {"document": document, "partition": partition_name}
        else:
            document_to_add = {"document": document}

        if document not in self.existing_records:
            self.table.add([document_to_add])
            self.existing_records.append(document)
            self._operation_count += 1
            self._maybe_optimize()

    def extend(
        self,
        documents: list[str],
        partition: str | None = None,
        partitions: list[str] | None = None,
    ):
        """Extend a list of documents to the store.

        When partitioning is enabled, you can assign documents to partitions in two ways:

        1. **Single partition for all documents**: Use `partition` parameter to assign
           all documents to the same partition.

           Example:
               ``store.extend(["doc1", "doc2"], partition="tutorials")``

        2. **One partition per document**: Use `partitions` parameter to assign each
           document to its corresponding partition. The list must be the same length
           as documents.

           Example:
               ``store.extend(["doc1", "doc2"], partitions=["tutorials", "reference"])``

        If neither `partition` nor `partitions` is provided and partitioning is enabled,
        all documents will be assigned to the default partition.

        :param documents: The documents to append.
        :param partition: Optional partition name for all documents. Ignored if
            `partitions` is provided. If partitioning is enabled and neither parameter
            is provided, defaults to `default_partition`.
        :param partitions: Optional list of partition names, one per document.
            Must be same length as documents if provided. Takes precedence over
            `partition` parameter.
        :raises ValueError: If `partitions` length doesn't match `documents` length.
        """
        if self.enable_partitioning:
            if partitions is not None:
                if len(partitions) != len(documents):
                    raise ValueError(
                        "Length of partitions must match length of documents."
                    )
                partition_list = partitions
            elif partition is not None:
                partition_list = [partition] * len(documents)
            else:
                partition_list = [self.default_partition] * len(documents)
        else:
            partition_list = [None] * len(documents)

        stuff_to_add = []
        for i, doc in enumerate(documents):
            # Skip if document already exists
            if doc in self.existing_records:
                continue

            # Create document entry
            if self.enable_partitioning:
                entry = {"document": doc, "partition": partition_list[i]}
            else:
                entry = {"document": doc}
            stuff_to_add.append(entry)

        # Use add instead of merge_insert to avoid schema conflicts
        if stuff_to_add:
            self.table.add(stuff_to_add)
            self.existing_records.extend(documents)
            # Count as one operation (batch insert)
            self._operation_count += 1
            self._maybe_optimize()

    def retrieve(
        self, query: str, n_results: int = 10, partitions: list[str] | None = None
    ) -> list[str]:
        """Retrieve a list of documents from the store.

        When partitioning is enabled, you can filter results by partition:

        - Provide a list of partition names to search only those partitions
        - Provide `None` (default) to search across all partitions
        - This parameter is ignored if partitioning is disabled

        :param query: The query to use to retrieve documents.
        :param n_results: The number of results to retrieve.
        :param partitions: Optional list of partition names to search. If None and
            partitioning enabled, searches all partitions. Ignored if partitioning
            disabled.
        :return: A list of documents.
        """
        search_query = self.table.search(query, query_type="auto")

        # Apply partition filtering if partitioning is enabled
        if self.enable_partitioning and partitions is not None:
            if len(partitions) == 1:
                where_clause = f"partition = '{partitions[0]}'"
            else:
                # Build IN clause for multiple partitions
                partition_list = "', '".join(partitions)
                where_clause = f"partition IN ('{partition_list}')"
            search_query = search_query.where(where_clause)

        results = (
            search_query.rerank(self.reranker).limit(n_results).to_pydantic(self.schema)
        )
        return [r.document for r in results]

    def list_partitions(self) -> list[str]:
        """List all available partition names.

        :return: List of partition names.
        :raises ValueError: If partitioning is not enabled.
        """
        if not self.enable_partitioning:
            raise ValueError("Partitioning is not enabled for this document store.")

        # Query all records and get distinct partition values
        all_items = self.table.search().limit(None).to_pydantic(self.schema)
        partitions = set()
        for item in all_items:
            if hasattr(item, "partition"):
                partitions.add(item.partition)
        return sorted(list(partitions))

    def reset_partition(self, partition: str):
        """Reset a specific partition by deleting all documents in it.

        :param partition: The partition name to reset.
        :raises ValueError: If partitioning is not enabled.
        """
        if not self.enable_partitioning:
            raise ValueError("Partitioning is not enabled for this document store.")

        # Query to find which documents are in this partition before deleting
        partition_items = (
            self.table.search()
            .where(f"partition = '{partition}'")
            .limit(None)
            .to_pydantic(self.schema)
        )
        partition_docs = {item.document for item in partition_items}

        # Delete all documents with this partition
        self.table.delete(f"partition = '{partition}'")

        # Update existing_records by removing documents from this partition
        self.existing_records = [
            doc for doc in self.existing_records if doc not in partition_docs
        ]
        # Count delete as an operation
        self._operation_count += 1
        self._maybe_optimize()

    def get_partition_count(self, partition: str) -> int:
        """Get the count of documents in a specific partition.

        :param partition: The partition name.
        :return: Number of documents in the partition.
        :raises ValueError: If partitioning is not enabled.
        """
        if not self.enable_partitioning:
            raise ValueError("Partitioning is not enabled for this document store.")

        # Query all records in this partition
        partition_items = (
            self.table.search()
            .where(f"partition = '{partition}'")
            .limit(None)
            .to_pydantic(self.schema)
        )
        return len(list(partition_items))

    def _maybe_optimize(self):
        """Optimize the table if operation count exceeds threshold.

        LanceDB recommends optimizing after 20+ operations or 100k+ records
        to maintain optimal performance.
        """
        if self._operation_count >= self._optimize_threshold:
            try:
                self.table.optimize()
                self._operation_count = 0  # Reset counter after optimization
            except Exception:
                # Silently fail if optimization errors occur
                # (e.g., FTS index not ready yet)
                pass

    def optimize(self):
        """Manually optimize the table for better performance.

        This method performs compaction, pruning, and index optimization.
        LanceDB recommends calling this periodically, especially after
        adding or modifying many records (e.g., after 20+ operations or 100k+ records).
        """
        self.table.optimize()
        self._operation_count = 0  # Reset counter after manual optimization

    def reset(self):
        """Reset the document store."""
        self.db.drop_table(self.table_name)
        self.table = self.db.create_table(self.table_name, schema=self.schema)
        self.existing_records = []
        self._operation_count = 0


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
