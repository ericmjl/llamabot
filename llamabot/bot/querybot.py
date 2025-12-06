"""Class definition for QueryBot."""

import contextvars
from pathlib import Path
from typing import List, Optional, Union
from dotenv import load_dotenv
from datetime import datetime

from llamabot.config import default_language_model

from llamabot.bot.simplebot import (
    SimpleBot,
    extract_content,
    extract_tool_calls,
    make_response,
    stream_chunks,
)
from llamabot.components.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    to_basemessage,
)
from llamabot.components.docstore import AbstractDocumentStore
from llamabot.components.messages import (
    RetrievedMessage,
)
from llamabot.recorder import sqlite_log, is_span_recording_enabled, Span

load_dotenv()


CACHE_DIR = Path.home() / ".llamabot" / "cache"
prompt_recorder_var = contextvars.ContextVar("prompt_recorder")


class QueryBot(SimpleBot):
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
        docstore: AbstractDocumentStore,
        memory: Optional[AbstractDocumentStore] = None,
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

        self.docstore = docstore
        self.memory = memory

    def __call__(
        self, query: Union[str, HumanMessage, BaseMessage], n_results: int = 20
    ) -> AIMessage:
        """Query documents within QueryBot's document store.

        We use RAG to query out documents.

        :param query: The query to make of the documents.
        """
        query_content = query if isinstance(query, str) else query.content

        # Check if span recording is enabled
        if is_span_recording_enabled():
            outer_span = Span(
                "querybot_call",
                query=query_content,
                n_results=n_results,
                model=self.model_name,
            )
            with outer_span:
                return self._call_with_spans(query_content, n_results, outer_span)
        else:
            return self._call_without_spans(query_content, n_results)

    def _call_without_spans(self, query: str, n_results: int) -> AIMessage:
        """Internal method for calling bot without spans."""
        # Initialize run metadata
        self.run_meta = {
            "start_time": datetime.now(),
            "query": query,
            "n_results": n_results,
            "retrieval_metrics": {
                "docstore_retrieval_time": 0,
                "memory_retrieval_time": 0,
                "docstore_results": 0,
                "memory_results": 0,
            },
        }

        messages = [self.system_prompt]
        retreived_messages = set()

        retrieved_messages = retreived_messages.union(
            self.docstore.retrieve(query, n_results)
        )
        retrieved: List = [
            RetrievedMessage(content=chunk) for chunk in retrieved_messages
        ]
        messages.extend(retrieved)

        # Record docstore metrics
        self.run_meta["retrieval_metrics"]["docstore_results"] = len(retrieved_messages)

        if self.memory:
            memory_start = datetime.now()
            memory_messages = self.memory.retrieve(query, n_results)
            self.run_meta["retrieval_metrics"]["memory_retrieval_time"] = (
                datetime.now() - memory_start
            ).total_seconds()
            memories: List = [
                RetrievedMessage(content=chunk) for chunk in memory_messages
            ]
            messages.extend(memories)
            self.run_meta["retrieval_metrics"]["memory_results"] = len(memory_messages)

        messages.append(HumanMessage(content=query))

        processed_messages = to_basemessage(messages)
        response = make_response(self, processed_messages, self.stream_target != "none")
        response = stream_chunks(response, target=self.stream_target)
        tool_calls = extract_tool_calls(response)
        content = extract_content(response)
        response_message = AIMessage(content=content, tool_calls=tool_calls)
        sqlite_log(self, messages + [response_message])

        if self.memory:
            self.memory.append(response_message.content)

        # Record end time and duration
        self.run_meta["end_time"] = datetime.now()
        self.run_meta["duration"] = (
            self.run_meta["end_time"] - self.run_meta["start_time"]
        ).total_seconds()

        return response_message

    def _call_with_spans(
        self, query: str, n_results: int, outer_span: Span
    ) -> AIMessage:
        """Internal method for calling bot with spans."""
        # Initialize run metadata
        self.run_meta = {
            "start_time": datetime.now(),
            "query": query,
            "n_results": n_results,
            "retrieval_metrics": {
                "docstore_retrieval_time": 0,
                "memory_retrieval_time": 0,
                "docstore_results": 0,
                "memory_results": 0,
            },
        }

        messages = [self.system_prompt]

        # Create span for document retrieval
        with outer_span.span("retrieval", n_results=n_results) as retrieval_span:
            retreived_messages = set()
            retrieved_messages = retreived_messages.union(
                self.docstore.retrieve(query, n_results)
            )
            retrieved: List = [
                RetrievedMessage(content=chunk) for chunk in retrieved_messages
            ]
            messages.extend(retrieved)

            # Record docstore metrics
            self.run_meta["retrieval_metrics"]["docstore_results"] = len(
                retrieved_messages
            )
            retrieval_span["documents_found"] = len(retrieved_messages)

        if self.memory:
            with outer_span.span("memory_retrieval") as memory_span:
                memory_start = datetime.now()
                memory_messages = self.memory.retrieve(query, n_results)
                self.run_meta["retrieval_metrics"]["memory_retrieval_time"] = (
                    datetime.now() - memory_start
                ).total_seconds()
                memories: List = [
                    RetrievedMessage(content=chunk) for chunk in memory_messages
                ]
                messages.extend(memories)
                self.run_meta["retrieval_metrics"]["memory_results"] = len(
                    memory_messages
                )
                memory_span["memory_results"] = len(memory_messages)

        messages.append(HumanMessage(content=query))

        # Create span for LLM request
        with outer_span.span("llm_request", model=self.model_name):
            processed_messages = to_basemessage(messages)
            response = make_response(
                self, processed_messages, self.stream_target != "none"
            )
            response = stream_chunks(response, target=self.stream_target)
            tool_calls = extract_tool_calls(response)
            content = extract_content(response)
            response_message = AIMessage(content=content, tool_calls=tool_calls)

        # Create span for LLM response
        with outer_span.span("llm_response") as llm_response_span:
            llm_response_span["response_length"] = len(content) if content else 0
            llm_response_span["tool_calls_count"] = len(tool_calls) if tool_calls else 0
            sqlite_log(self, messages + [response_message])

        if self.memory:
            self.memory.append(response_message.content)

        # Record end time and duration
        self.run_meta["end_time"] = datetime.now()
        self.run_meta["duration"] = (
            self.run_meta["end_time"] - self.run_meta["start_time"]
        ).total_seconds()

        return response_message
