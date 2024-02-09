"""Class definition for QueryBot."""
import contextvars
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


from llamabot.config import default_language_model
from llamabot.bot.simplebot import SimpleBot
from llamabot.components.messages import AIMessage, HumanMessage
from llamabot.components.docstore import DocumentStore
from llamabot.components.api import APIMixin
from llamabot.components.messages import (
    RetrievedMessage,
    retrieve_messages_up_to_budget,
)
from llamabot.bot.model_tokens import model_context_window_sizes, DEFAULT_TOKEN_BUDGET
from slugify import slugify

load_dotenv()


CACHE_DIR = Path.home() / ".llamabot" / "cache"
prompt_recorder_var = contextvars.ContextVar("prompt_recorder")


class QueryBot(SimpleBot, DocumentStore, APIMixin):
    """QueryBot is a bot that uses simple RAG to answer questions about a document."""

    def __init__(
        self,
        system_prompt: str,
        collection_name: str,
        document_paths: Optional[Path | list[Path]] = None,
        temperature: float = 0.0,
        model_name: str = default_language_model(),
        stream=True,
    ):
        SimpleBot.__init__(
            self,
            system_prompt=system_prompt,
            temperature=temperature,
            model_name=model_name,
            stream=stream,
        )
        DocumentStore.__init__(self, collection_name=slugify(collection_name))
        if document_paths:
            self.add_documents(document_paths=document_paths)
        self.response_budget = 2_000

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
                for chunk in self.retrieve(query, n_results=n_results)
            ],
            character_budget=context_budget - self.response_budget,
        )
        messages.extend(retrieved)
        messages.append(HumanMessage(content=query))
        response: AIMessage = self.generate_response(messages)
        return response
