"""Async bot classes for ``await bot(...)`` and ``stream_async``.

Sync bots (:class:`~llamabot.bot.simplebot.SimpleBot` and subclasses) use
blocking :meth:`~llamabot.bot.simplebot.SimpleBot.__call__`. For FastAPI,
SSE, or other async contexts, construct an ``Async*`` bot with the same
constructor arguments; message composition is shared with the sync types via
``compose_*`` helpers on the respective base classes.
"""

import asyncio
import inspect
import uuid
from typing import (
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

from loguru import logger
from pydantic import BaseModel

from llamabot.bot.querybot import QueryBot
from llamabot.bot.simplebot import (
    SimpleBot,
    async_stream_chunks,
    extract_content,
    extract_tool_calls,
    make_async_response,
    stream_tokens_for_messages,
)
from llamabot.bot.structuredbot import StructuredBot
from llamabot.bot.toolbot import ToolBot
from llamabot.components.messages import AIMessage, BaseMessage, HumanMessage
from llamabot.recorder import (
    Span,
    get_caller_variable_name,
    get_current_span,
    sqlite_log,
)


class AsyncSimpleBot(SimpleBot):
    """Async :class:`~llamabot.bot.simplebot.SimpleBot` with ``await`` and streaming."""

    async def stream_async(
        self,
        *human_messages: Union[str, BaseMessage, list[Union[str, BaseMessage]]],
    ) -> AsyncGenerator[str, None]:
        """Stream assistant text deltas (same inputs as :meth:`~llamabot.bot.simplebot.SimpleBot.__call__`)."""
        messages, processed_messages = self.compose_messages_for_human_messages(
            *human_messages
        )

        def finalize(final) -> None:
            """Persist logging and memory after the stream is assembled.

            :param final: Assembled response from LiteLLM ``stream_chunk_builder``.
            """
            tool_calls = extract_tool_calls(final)
            content = extract_content(final)
            response_message = AIMessage(content=content, tool_calls=tool_calls)
            sqlite_log(self, messages + [response_message])
            if self.memory and processed_messages:
                self.memory.append(processed_messages[-1])
                self.memory.append(response_message)

        async for delta in stream_tokens_for_messages(
            self, messages, finalize=finalize
        ):
            yield delta

    async def __call__(
        self,
        *human_messages: Union[str, BaseMessage, list[Union[str, BaseMessage]]],
    ) -> AIMessage:
        """Async completion: same role as :meth:`~llamabot.bot.simplebot.SimpleBot.__call__`."""
        query_content = " ".join(
            [
                msg.content if hasattr(msg, "content") else str(msg)
                for msg in human_messages
            ]
        )
        operation_name = get_caller_variable_name(self)
        if operation_name is None:
            operation_name = "simplebot_call"

        current_span = get_current_span()
        if current_span:
            outer_span = Span(
                operation_name,
                trace_id=current_span.trace_id,
                parent_span_id=current_span.span_id,
                query=query_content,
                model=self.model_name,
                temperature=self.temperature,
            )
            if outer_span.trace_id not in self._trace_ids:
                self._trace_ids.append(outer_span.trace_id)
        else:
            new_trace_id = str(uuid.uuid4())
            outer_span = Span(
                operation_name,
                trace_id=new_trace_id,
                query=query_content,
                model=self.model_name,
                temperature=self.temperature,
            )
            if outer_span.trace_id not in self._trace_ids:
                self._trace_ids.append(outer_span.trace_id)

        with outer_span:
            messages, processed_messages = self.compose_messages_for_human_messages(
                *human_messages
            )
            self.record_span_message_metadata(outer_span, messages, processed_messages)

            result_holder: dict[str, AIMessage | None] = {"msg": None}

            def finalize(final) -> None:
                """Record span fields, logging, and memory from the final response.

                :param final: Assembled response from LiteLLM ``stream_chunk_builder``.
                """
                tool_calls = extract_tool_calls(final)
                content = extract_content(final)
                response_message = AIMessage(content=content, tool_calls=tool_calls)
                result_holder["msg"] = response_message
                outer_span["response_length"] = len(content) if content else 0
                outer_span["response"] = content[:500] if content else None
                outer_span["tool_calls_count"] = len(tool_calls) if tool_calls else 0
                if tool_calls:
                    tool_names = [call.function.name for call in tool_calls]
                    outer_span["tool_calls"] = ", ".join(tool_names)
                    outer_span["tool_calls_count"] = len(tool_calls)
                sqlite_log(self, messages + [response_message])
                if self.memory and processed_messages:
                    self.memory.append(processed_messages[-1])
                    self.memory.append(response_message)

            async for delta in stream_tokens_for_messages(
                self, messages, finalize=finalize
            ):
                if self.stream_target == "stdout":
                    print(delta, end="")

            response_message = result_holder["msg"]
            if response_message is None:
                raise RuntimeError("Async completion produced no assistant message")
            return response_message


class AsyncQueryBot(QueryBot):
    """Async :class:`~llamabot.bot.querybot.QueryBot`."""

    async def stream_async(
        self,
        query: Union[str, HumanMessage, BaseMessage],
        n_results: int = 20,
    ) -> AsyncGenerator[str, None]:
        """Stream assistant text after RAG retrieval (same shape as sync :meth:`~llamabot.bot.querybot.QueryBot.__call__`)."""
        query_content = query if isinstance(query, str) else query.content

        operation_name = get_caller_variable_name(self)
        if operation_name is None:
            operation_name = "querybot_stream_async"

        current_span = get_current_span()
        if current_span:
            outer_span = Span(
                operation_name,
                trace_id=current_span.trace_id,
                parent_span_id=current_span.span_id,
                query=query_content,
                n_results=n_results,
                model=self.model_name,
            )
            if outer_span.trace_id not in self._trace_ids:
                self._trace_ids.append(outer_span.trace_id)
        else:
            new_trace_id = str(uuid.uuid4())
            outer_span = Span(
                operation_name,
                trace_id=new_trace_id,
                query=query_content,
                n_results=n_results,
                model=self.model_name,
            )
            if outer_span.trace_id not in self._trace_ids:
                self._trace_ids.append(outer_span.trace_id)

        with outer_span:
            messages, processed_messages = self.compose_rag_messages(
                query, n_results, outer_span
            )

            def finalize(final) -> None:
                """Update span, log, and optional memory docstore after RAG streaming.

                :param final: Assembled response from LiteLLM ``stream_chunk_builder``.
                """
                tool_calls = extract_tool_calls(final)
                content = extract_content(final)
                response_message = AIMessage(content=content, tool_calls=tool_calls)
                outer_span["response_length"] = len(content) if content else 0
                outer_span["response"] = content[:500] if content else None
                outer_span["tool_calls_count"] = len(tool_calls) if tool_calls else 0
                sqlite_log(self, messages + [response_message])
                if self.memory:
                    self.memory.append(response_message.content)

            async for delta in stream_tokens_for_messages(
                self, processed_messages, finalize=finalize
            ):
                yield delta

    async def __call__(
        self,
        query: Union[str, HumanMessage, BaseMessage],
        n_results: int = 20,
    ) -> AIMessage:
        """Async RAG query; delegates to the sync implementation on a worker thread."""
        return await asyncio.to_thread(super().__call__, query, n_results)


class AsyncToolBot(ToolBot):
    """Async :class:`~llamabot.bot.toolbot.ToolBot`.

    :meth:`__call__` uses LiteLLM ``acompletion`` and :func:`~llamabot.bot.simplebot.async_stream_chunks`
    (no :func:`asyncio.to_thread` around the completion). For sync tool selection on a worker
    thread, use :class:`ToolBot` or ``await asyncio.to_thread(bot, ...)``.
    """

    async def stream_async(
        self,
        *messages: Union[str, BaseMessage, list[Union[str, BaseMessage]], Callable],
        execution_history: Optional[List[Dict]] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream assistant text for tool selection (same inputs as :meth:`~llamabot.bot.toolbot.ToolBot.__call__`)."""
        message_list, user_messages = self.compose_tool_messages(
            *messages, execution_history=execution_history
        )
        logger.debug("Message list: {}", message_list)

        def finalize(final) -> None:
            """Append tool-call results to chat memory after streaming completes.

            :param final: Assembled response from LiteLLM ``stream_chunk_builder``.
            """
            tool_calls = extract_tool_calls(final)
            if user_messages:
                self.chat_memory.append(user_messages[0])
                self.chat_memory.append(AIMessage(content=str(tool_calls)))

        async for delta in stream_tokens_for_messages(
            self, message_list, finalize=finalize
        ):
            yield delta

    async def __call__(
        self,
        *messages: Union[str, BaseMessage, list[Union[str, BaseMessage]], Callable],
        execution_history: Optional[List[Dict]] = None,
    ) -> list:
        """Async tool selection via ``acompletion`` (same contract as :meth:`ToolBot.__call__`)."""
        message_list, user_messages = self.compose_tool_messages(
            *messages, execution_history=execution_history
        )

        stream = self.stream_target != "none"
        logger.debug("Message list: {}", message_list)
        response = await make_async_response(self, message_list, stream=stream)
        final = await async_stream_chunks(self, response, target=self.stream_target)
        logger.debug("Response: {}", final)

        try:
            if hasattr(final, "choices") and hasattr(final.choices, "__len__"):
                choices_len = len(final.choices)
                if choices_len > 0:
                    message = final.choices[0].message
                    tool_calls_attr = getattr(message, "tool_calls", None)
                    content_attr = getattr(message, "content", None)
                    logger.debug("Response message.tool_calls: {}", tool_calls_attr)
                    logger.debug("Response message.content: {}", content_attr)
        except (TypeError, AttributeError):
            pass

        tool_calls = extract_tool_calls(final)

        try:
            if (
                tool_calls
                and hasattr(final, "choices")
                and hasattr(final.choices, "__len__")
            ):
                choices_len = len(final.choices)
                if choices_len > 0:
                    message = final.choices[0].message
                    if message.tool_calls is None and message.content:
                        logger.debug(
                            "Parsed tool calls from JSON content for model {}: {}",
                            self.model_name,
                            [tc.function.name for tc in tool_calls],
                        )
        except (TypeError, AttributeError):
            pass

        if user_messages:
            self.chat_memory.append(user_messages[0])
            self.chat_memory.append(AIMessage(content=str(tool_calls)))

        return tool_calls


class AsyncStructuredBot(StructuredBot):
    """Async :class:`~llamabot.bot.structuredbot.StructuredBot`."""

    async def stream_async(
        self,
        *user_messages: Union[str, BaseMessage, list[Union[str, BaseMessage]]],
    ) -> AsyncGenerator[str, None]:
        """Stream one structured completion attempt (no validation on partial chunks)."""
        full_messages, messages = self.compose_structured_first_attempt_messages(
            *user_messages
        )
        query_content = " ".join(
            [msg.content if hasattr(msg, "content") else str(msg) for msg in messages]
        )

        operation_name = get_caller_variable_name(self)
        if operation_name is None:
            operation_name = "structuredbot_stream_async"

        current_span = get_current_span()
        if current_span:
            model_name = (
                self.pydantic_model.__name__
                if inspect.isclass(self.pydantic_model)
                else type(self.pydantic_model).__name__
            )
            outer_span = Span(
                operation_name,
                trace_id=current_span.trace_id,
                parent_span_id=current_span.span_id,
                query=query_content,
                model=self.model_name,
                pydantic_model=model_name,
            )
            if outer_span.trace_id not in self._trace_ids:
                self._trace_ids.append(outer_span.trace_id)
        else:
            model_name = (
                self.pydantic_model.__name__
                if inspect.isclass(self.pydantic_model)
                else type(self.pydantic_model).__name__
            )
            new_trace_id = str(uuid.uuid4())
            outer_span = Span(
                operation_name,
                trace_id=new_trace_id,
                query=query_content,
                model=self.model_name,
                pydantic_model=model_name,
            )
            if outer_span.trace_id not in self._trace_ids:
                self._trace_ids.append(outer_span.trace_id)

        with outer_span:

            def finalize(final) -> None:
                """Log the assembled structured JSON text after streaming completes.

                :param final: Assembled response from LiteLLM ``stream_chunk_builder``.
                """
                content = extract_content(final)
                last_response = AIMessage(content=content)
                sqlite_log(self, messages + [last_response])

            async for delta in stream_tokens_for_messages(
                self, full_messages, finalize=finalize
            ):
                yield delta

    async def __call__(
        self,
        *messages: Union[str, BaseMessage, list[Union[str, BaseMessage]]],
        num_attempts: int = 10,
        verbose: bool = False,
    ) -> BaseModel | None:
        """Async structured completion; delegates to the sync implementation on a worker thread."""
        return await asyncio.to_thread(
            super().__call__, *messages, num_attempts=num_attempts, verbose=verbose
        )
