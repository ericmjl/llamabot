"""Class definition for QueryBot."""

import contextvars
from pathlib import Path
from typing import Optional, Union
from dotenv import load_dotenv

from llamabot.config import default_language_model

from llamabot.bot.simplebot import SimpleBot
from llamabot.components.messages import AIMessage, BaseMessage, HumanMessage
from llamabot.components.docstore import LanceDBDocStore, ChromaDBDocStore
from llamabot.components.chatui import ChatUIMixin
from llamabot.components.messages import (
    RetrievedMessage,
)

load_dotenv()


CACHE_DIR = Path.home() / ".llamabot" / "cache"
prompt_recorder_var = contextvars.ContextVar("prompt_recorder")


class QueryBot(SimpleBot, ChatUIMixin):
    """Initialize QueryBot.

    QueryBot is a bot that can answer questions based on a set of documents.
    It uses a document store to retrieve relevant documents for a given query.

    You can either connect to an existing document store
    by providing the appropriate collection_name and docstore_type,
    or create a new one.
    If the collection already exists, QueryBot will connect to it;
    otherwise, a new collection will be created.
    The `docstore_path` parameter allows you to specify a custom storage location
    (defaults to `~/.llamabot/lancedb` or `~/.llamabot/chroma.db`
    depending on `docstore_type`).
    The collection_name is used as the table name in LanceDB
    or collection name in ChromaDB
    and will be automatically slugified for compatibility.
    If document_paths are provided, they will be added to the store.

    :param system_prompt: The system prompt to use for the bot.
    :param collection_name: The name of the collection to store documents in.
    :param initial_message: Optional initial message to start the conversation.
    :param document_paths: Optional path or list of paths to documents to add to the store.
    :param docstore_type: The type of document store to use ("lancedb" or "chromadb").
    :param docstore_path: Optional custom path for document store storage.
    :param docstore_kwargs: Additional keyword arguments to pass to the document store.
    :param mock_response: Optional mock response for testing purposes.
    :param temperature: Temperature parameter for the language model (0.0 = deterministic).
    :param model_name: Name of the language model to use.
    :param stream_target: Where to stream responses ("stdout" or "panel").
    """

    def __init__(
        self,
        system_prompt: str,
        collection_name: str,
        initial_message: Optional[str] = None,
        document_paths: Optional[Path | list[Path]] = None,
        docstore_type: str = "lancedb",
        docstore_path: Optional[Path] = None,
        docstore_kwargs: dict = {},
        mock_response: str | None = None,
        temperature: float = 0.0,
        model_name: str = default_language_model(),
        stream_target: str = "stdout",
        **kwargs,
    ):
        SimpleBot.__init__(
            self,
            system_prompt=system_prompt,
            temperature=temperature,
            model_name=model_name,
            mock_response=mock_response,
            stream_target=stream_target,
            **kwargs,
        )

        if docstore_path:
            docstore_kwargs["storage_path"] = docstore_path

        # Initialize the appropriate document store
        if docstore_type == "lancedb":
            self.docstore = LanceDBDocStore(
                table_name=collection_name,
                **docstore_kwargs,
            )
        elif docstore_type == "chromadb":
            self.docstore = ChromaDBDocStore(
                collection_name=collection_name,
                **docstore_kwargs,
            )
        else:
            raise ValueError(f"Unknown docstore type: {docstore_type}")

        # Add documents to the store
        if document_paths:
            self.docstore.add_documents(document_paths=document_paths)

        self.response_budget = 2_000

        ChatUIMixin.__init__(self, initial_message)

    def __call__(
        self, query: Union[str, HumanMessage, BaseMessage], n_results: int = 20
    ) -> AIMessage:
        """Query documents within QueryBot's document store.

        We use RAG to query out documents.

        :param query: The query to make of the documents.
        """
        messages = [self.system_prompt]

        retreived_messages = set()

        q = query
        if isinstance(query, (BaseMessage, HumanMessage)):
            q = query.content
        retrieved_messages = retreived_messages.union(
            self.docstore.retrieve(q, n_results)
        )
        retrieved = [RetrievedMessage(content=chunk) for chunk in retrieved_messages]
        messages.extend(retrieved)
        messages.append(HumanMessage(content=q))
        if self.stream_target == "stdout":
            response: AIMessage = self.stream_stdout(messages)
            return response
        elif self.stream_target == "panel":
            return self.stream_panel(messages)
