"""Class for storing and retrieving messages from a message history.

Message history is always stored in a chroma database.
We can retrieve the last N messages from the database
or use vector similarity search to identify similar messages to retrieve (RAGHistory).
"""

from pathlib import Path
from hashlib import sha256
from llamabot.components.messages import (
    BaseMessage,
    RetrievedMessage,
    retrieve_messages_up_to_budget,
)


class History:
    """History of messages."""

    def __init__(self, session_name: str):
        self.messages: list[BaseMessage] = []
        self.session_name = session_name

    def append(self, message: BaseMessage):
        """Append a message to the history."""
        self.messages.append(message)

    def retrieve(self, query: BaseMessage, character_budget: int) -> list[BaseMessage]:
        """Retrieve messages from the history up to the the character budget.

        We use the character budget in order to simplify how we retrieve messages.

        :param query: The query to use to retrieve messages. Not used in this class.
        :param character_budget: The number of characters to retrieve.
        """
        return retrieve_messages_up_to_budget(self.messages, character_budget)

    def __getitem__(self, index):
        """Get the message at the given index."""
        return self.messages[index]


class RAGHistory:
    """History of messages with vector similarity enabled."""

    def __init__(
        self,
        session_name: str,
        db_path: Path = Path.home() / ".llamabot" / "chroma.db",
    ):
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "ChromaDB is required for RAGHistory. "
                "Please `pip install llamabot[rag]` to use the RAG-enabled history."
            )

        client = chromadb.PersistentClient(path=str(db_path))
        collection = client.create_collection(session_name, get_or_create=True)

        self.db_path = db_path
        self.client = client
        self.collection = collection
        self.session_name = session_name

    def append(self, message: BaseMessage):
        """Append a message to the history.

        :param message: The message to append.
        """
        doc_id = sha256(message.content.encode()).hexdigest()
        self.collection.add(documents=message.content, ids=doc_id)

    def retrieve(
        self, query: BaseMessage, character_budget: int, n_results: int = 50
    ) -> list[BaseMessage]:
        """Retrieve messages from the history up to the the character budget.

        :param query: The query to use to retrieve messages.
        :param character_budget: The number of characters to retrieve.
        :param n_results: The number of results to retrieve.
        :returns: The retrieved messages.
        """
        results: dict = self.collection.query(
            query_texts=query.content, n_results=n_results
        )
        messages: list[RetrievedMessage] = [
            RetrievedMessage(content=doc) for doc in results["documents"][0]
        ]
        return retrieve_messages_up_to_budget(messages, character_budget)

    def __getitem__(self, index) -> RetrievedMessage:
        """Get the message at the given index.

        :param index: The index to use.
        :returns: The retrieved message.
        """
        documents = self.collection.get()["documents"]
        document = RetrievedMessage(content=documents[index])
        return document
