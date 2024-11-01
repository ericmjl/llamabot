"""Class definition for QueryBot."""

import contextvars
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from llamabot.config import default_language_model

from llamabot.bot.simplebot import SimpleBot
from llamabot.components.messages import AIMessage, HumanMessage
from llamabot.components.docstore import LanceDBDocStore
from llamabot.components.chatui import ChatUIMixin
from llamabot.components.messages import (
    RetrievedMessage,
)
from slugify import slugify

load_dotenv()


CACHE_DIR = Path.home() / ".llamabot" / "cache"
prompt_recorder_var = contextvars.ContextVar("prompt_recorder")


class QueryBot(SimpleBot, ChatUIMixin):
    """Initialize QueryBot.

    :param system_prompt: The system prompt to use.
    :param collection_name: The name of the collection to use.
    :param document_paths: The paths to the documents to use.
    :param docstore_type: The type of document store to use ("lancedb", "sqlitevec", etc.)
    :param mock_response: A mock response to use for testing.
    :param stream_target: The target to stream to ("panel" or "stdout").
    """

    def __init__(
        self,
        system_prompt: str,
        collection_name: str,
        initial_message: Optional[str] = None,
        document_paths: Optional[Path | list[Path]] = None,
        docstore_type: str = "lancedb",  # Add this parameter
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

        collection_name = slugify(collection_name, separator="_")

        # Initialize the appropriate document store
        if docstore_type == "lancedb":
            self.docstore = LanceDBDocStore(
                table_name=collection_name,
                storage_path=Path.home() / ".llamabot" / "lancedb",
            )
        else:
            raise ValueError(f"Unknown docstore type: {docstore_type}")

        # Add documents to the store
        if document_paths:
            self.docstore.add_documents(document_paths=document_paths)

        self.response_budget = 2_000

        ChatUIMixin.__init__(self, initial_message)

    def __call__(self, query: str, n_results: int = 20) -> AIMessage:
        """Query documents within QueryBot's document store.

        We use RAG to query out documents.

        :param query: The query to make of the documents.
        """
        messages = [self.system_prompt]

        retreived_messages = set()
        retrieved_messages = retreived_messages.union(
            self.docstore.retrieve(query, n_results)
        )
        retrieved = [RetrievedMessage(content=chunk) for chunk in retrieved_messages]
        messages.extend(retrieved)
        messages.append(HumanMessage(content=query))
        if self.stream_target == "stdout":
            response: AIMessage = self.stream_stdout(messages)
            return response
        elif self.stream_target == "panel":
            return self.stream_panel(messages)
