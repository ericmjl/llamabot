"""Archived code for using sqlite-vec as a document store.

We archived this because sqlite-vec is very bleeding-edge.
It doesn't support arm64 linux.
As such, it breaks devcontainer (which is based on Linux) builds on Apple Silicon.
When sqlite-vec supports arm64 linux, we should re-enable this.
"""

from pathlib import Path
from typing import List

from sentence_transformers import SentenceTransformer


from llamabot.components.docstore import AbstractDocumentStore
import sqlite3

import sqlite_vec


class SQLiteVecDocStore(AbstractDocumentStore):
    """A document store for LlamaBot that uses sqlite-vec."""

    def __init__(
        self,
        db_path: Path = Path.home() / ".llamabot" / "sqlite_vec.db",
        table_name: str = "documents",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.db_path = db_path
        self.table_name = table_name
        self.embedding_model = SentenceTransformer(embedding_model)
        self.vector_dim = self.embedding_model.get_sentence_embedding_dimension()

        self.db = sqlite3.connect(str(db_path))
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)

        self._create_table()

    def _create_table(self):
        """Create the necessary tables."""
        # Create the documents table
        self.db.execute(
            f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id INTEGER PRIMARY KEY,
            document TEXT
        )
        """
        )

        # Create the vectors virtual table with vec_ prefix
        self.db.execute(
            f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_{self.table_name}
        USING vec0(embedding float[{self.vector_dim}])
        """
        )
        self.db.commit()

    def _embed(self, text: str) -> bytes:
        """Create embedding for text."""
        import struct

        embedding = self.embedding_model.encode(text)
        return struct.pack(f"{len(embedding)}f", *embedding)

    def append(self, document: str):
        """Append a document to the store."""
        # Insert the document and get its ID
        cursor = self.db.cursor()
        cursor.execute(
            f"INSERT INTO {self.table_name} (document) VALUES (?)", (document,)
        )
        doc_id = cursor.lastrowid

        # Create and insert the embedding
        embedding = self._embed(document)
        cursor.execute(
            f"INSERT INTO vec_{self.table_name} (rowid, embedding) VALUES (?, ?)",
            (doc_id, embedding),
        )

        self.db.commit()

    def extend(self, documents: List[str]):
        """Extend a list of documents to the store."""
        for document in documents:
            self.append(document)

    def retrieve(self, query: str, n_results: int = 10) -> List[str]:
        """Retrieve documents from the store."""
        query_embedding = self._embed(query)

        # First get the matching vector IDs
        vector_matches = self.db.execute(
            f"""
            SELECT
                rowid,
                distance
            FROM vec_{self.table_name}
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
            """,
            [query_embedding, n_results],
        ).fetchall()

        # Then get the corresponding documents
        if vector_matches:
            vector_ids = [match[0] for match in vector_matches]
            placeholders = ",".join("?" * len(vector_ids))
            documents = self.db.execute(
                f"""
                SELECT document
                FROM {self.table_name}
                WHERE id IN ({placeholders})
                """,
                vector_ids,
            ).fetchall()

            return [doc[0] for doc in documents]
        return []

    def reset(self):
        """Reset the document store."""
        self.db.execute(f"DROP TABLE IF EXISTS {self.table_name}")
        self.db.execute(f"DROP TABLE IF EXISTS vec_{self.table_name}")
        self.db.commit()
        self._create_table()

    def __post_add_documents__(self):
        """Execute code after adding documents to the store."""
        # No additional indexing needed for sqlite-vec
        pass
