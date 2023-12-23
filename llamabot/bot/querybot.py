"""Class definition for QueryBot."""
import contextvars
import os
from pathlib import Path
from dotenv import load_dotenv


from llamabot.config import default_language_model
from llamabot.doc_processor import magic_load_doc, split_document
from llamabot.bot.simplebot import SimpleBot
from llamabot.components.messages import AIMessage, HumanMessage
from llamabot.components.docstore import DocumentStore

from llamabot.components.messages import (
    RetrievedMessage,
    retrieve_messages_up_to_budget,
)
from llamabot.bot.model_tokens import model_context_window_sizes, DEFAULT_TOKEN_BUDGET


load_dotenv()

DEFAULT_SIMILARITY_TOP_KS = {
    "gpt-3.5-turbo": 2,
    "gpt-3.5-turbo-0301": 2,
    "gpt-3.5-turbo-0613": 2,
    "gpt-3.5-turbo-16k": 5,
    "gpt-3.5-turbo-16k-0613": 5,
    "gpt-4": 3,
    "gpt-4-0314": 3,
    "gpt-4-0613": 3,
    "gpt-4-32k": 10,
}


SIMILARITY_TOP_K = DEFAULT_SIMILARITY_TOP_KS.get(os.getenv("OPENAI_DEFAULT_MODEL"), 3)


CACHE_DIR = Path.home() / ".llamabot" / "cache"
prompt_recorder_var = contextvars.ContextVar("prompt_recorder")


class QueryBot:
    """QueryBot is a bot that uses simple RAG to answer questions about a document."""

    def __init__(
        self,
        system_prompt: str,
        document_paths: Path | list[Path],
        collection_name: str,
        temperature: float = 0.0,
        model_name: str = default_language_model(),
        stream=True,
    ):
        self.bot = SimpleBot(
            system_prompt=system_prompt,
            temperature=temperature,
            model_name=model_name,
            stream=stream,
        )
        self.document_store = DocumentStore(collection_name=collection_name)
        self.add(document_paths=document_paths)
        self.response_budget = 2_000
        self.model_name = model_name

    def add(
        self,
        document_paths: Path | list[Path],
        chunk_size: int = 2_000,
        chunk_overlap: int = 500,
    ):
        """Add documents to the QueryBot DocumentStore."""
        if isinstance(document_paths, Path):
            document_paths = [document_paths]

        for document_path in document_paths:
            document = magic_load_doc(document_path)
            splitted_document = split_document(
                document, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            splitted_document = [doc.text for doc in splitted_document]
            self.document_store.extend(splitted_document)

    def __call__(self, query: str, n_results: int = 20) -> AIMessage:
        """Query documents within QueryBot's document store.

        We use RAG to query out documents.

        :param query: The query to make of the documents.
        """
        messages = []

        context_budget = model_context_window_sizes.get(
            self.model_name, DEFAULT_TOKEN_BUDGET
        )
        retrieved = retrieve_messages_up_to_budget(
            messages=[
                RetrievedMessage(content=chunk)
                for chunk in self.document_store.retrieve(query, n_results=n_results)
            ],
            character_budget=context_budget - self.response_budget,
        )
        messages.extend(retrieved)
        messages.append(HumanMessage(content=query))
        response: str = self.bot.generate_response(messages)
        return AIMessage(content=response)
