"""Class definition for SimpleBot."""

import contextvars
import json
import uuid
from datetime import datetime
from types import NoneType
from typing import Generator, List, Optional, Union

from litellm import (
    ChatCompletionMessageToolCall,
    CustomStreamWrapper,
    Function,
    ModelResponse,
    stream_chunk_builder,
)
from loguru import logger
from pydantic import BaseModel

from llamabot.components.docstore import AbstractDocumentStore
from llamabot.components.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    to_basemessage,
)
from llamabot.config import default_language_model
from llamabot.recorder import (
    Span,
    build_hierarchy,
    generate_span_html,
    get_current_span,
    get_spans,
    span_to_dict,
    sqlite_log,
)

prompt_recorder_var = contextvars.ContextVar("prompt_recorder")


class SimpleBot:
    """Simple Bot that is primed with a system prompt, accepts a human message,
    and sends back a single response.

    This bot does not retain chat history by default, but can be configured
    to do so by passing in a chat memory component.

    :param system_prompt: The system prompt to use.
    :param temperature: The model temperature to use.
        See https://platform.openai.com/docs/api-reference/completions/create#completions/create-temperature
        for more information.
    :param memory: An optional chat memory component to use. If provided,
        the bot will retain chat history. For conversational memory, use
        `ChatMemory` (e.g., `lmb.ChatMemory()`). For RAG/document retrieval,
        use an `AbstractDocumentStore` (such as `BM25DocStore` or `LanceDBDocStore`).
    :param model_name: The name of the model to use.
    :param stream_target: The target to stream the response to.
        Should be one of ("stdout", "panel", "api").
    :param json_mode: Whether to print debug messages.
    :param api_key: The OpenAI API key to use.
    :param mock_response: A mock response to use, for testing purposes only.
    :param completion_kwargs: Additional keyword arguments to pass to the
        completion function of `litellm`.
    """

    def __init__(
        self,
        system_prompt: str,
        temperature=0.0,
        memory: Optional[AbstractDocumentStore] = None,
        model_name=default_language_model(),
        stream_target: str = "stdout",
        json_mode: bool = False,
        api_key: Optional[str] = None,
        mock_response: Optional[str] = None,
        **completion_kwargs,
    ):
        # Validate `stream_target.
        if stream_target not in ("stdout", "panel", "api", "none"):
            raise ValueError(
                f"stream_target must be one of ('stdout', 'panel', 'api', 'none'), got {stream_target}."
            )

        self.system_prompt = system_prompt
        if isinstance(system_prompt, str):
            self.system_prompt = SystemMessage(content=system_prompt)

        self.model_name = model_name
        self.stream_target = stream_target
        self.temperature = temperature
        self.memory = memory
        self.json_mode = json_mode
        self.api_key = api_key
        self.mock_response = mock_response
        self.completion_kwargs = completion_kwargs

        # Set special cases for for o1 models.
        if model_name in ["o1-preview", "o1-mini"]:
            self.system_prompt = HumanMessage(
                content=self.system_prompt.content,
            )
            self.temperature = 1.0
            self.stream_target = "none"

        # Single dictionary for all run metadata and metrics
        self.run_meta = {}

        # Track all trace_ids from this bot instance for span visualization
        self._trace_ids = []

    def __call__(
        self, *human_messages: Union[str, BaseMessage, list[Union[str, BaseMessage]]]
    ) -> Union[AIMessage, Generator]:
        """Call the SimpleBot.

        :param human_messages: One or more human messages to use, or lists of messages.
        :return: The response to the human messages, primed by the system prompt.
        """
        # Create outer span for the entire bot call
        query_content = " ".join(
            [
                msg.content if hasattr(msg, "content") else str(msg)
                for msg in human_messages
            ]
        )
        # Check if there's a current span - if so, create a child span
        current_span = get_current_span()
        if current_span:
            # Create child span using parent's trace_id
            outer_span = Span(
                "simplebot_call",
                trace_id=current_span.trace_id,
                parent_span_id=current_span.span_id,
                query=query_content,
                model=self.model_name,
                temperature=self.temperature,
            )
        else:
            # No current span - create a new trace
            new_trace_id = str(uuid.uuid4())
            outer_span = Span(
                "simplebot_call",
                trace_id=new_trace_id,
                query=query_content,
                model=self.model_name,
                temperature=self.temperature,
            )
            # Track trace_id for this bot instance (only for root spans)
            if outer_span.trace_id not in self._trace_ids:
                self._trace_ids.append(outer_span.trace_id)
        with outer_span:
            return self._call_with_spans(human_messages, outer_span)

    def _call_without_spans(
        self, human_messages: Union[str, BaseMessage, list[Union[str, BaseMessage]]]
    ) -> Union[AIMessage, Generator]:
        """Internal method for calling bot without spans."""
        # Reset run metadata for new call
        self.run_meta = {
            "start_time": datetime.now(),
            "message_counts": {"user": 0, "assistant": 0, "tool": 0},
            "tool_usage": {},
        }

        processed_messages = to_basemessage(human_messages)

        memory_messages = []
        if self.memory:
            memory_messages = self.memory.retrieve(
                query=f"From our conversation history, give me the most relevant information to the query, {[p.content for p in processed_messages]}"
            )

        messages = [self.system_prompt] + memory_messages + processed_messages

        # Count initial messages
        for msg in messages:
            if isinstance(msg, HumanMessage):
                self.run_meta["message_counts"]["user"] += 1
            elif isinstance(msg, AIMessage):
                self.run_meta["message_counts"]["assistant"] += 1

        stream = self.stream_target != "none"
        response = make_response(self, messages, stream)
        response = stream_chunks(response, target=self.stream_target)
        tool_calls = extract_tool_calls(response)
        content = extract_content(response)
        response_message = AIMessage(content=content, tool_calls=tool_calls)

        # Update message counts
        self.run_meta["message_counts"]["assistant"] += 1

        # Record tool usage if any
        if tool_calls:
            for call in tool_calls:
                tool_name = call.function.name
                if tool_name not in self.run_meta["tool_usage"]:
                    self.run_meta["tool_usage"][tool_name] = {
                        "calls": 0,
                        "success": 0,
                        "failures": 0,
                    }
                self.run_meta["tool_usage"][tool_name]["calls"] += 1

        sqlite_log(self, messages + [response_message])
        if self.memory:
            self.memory.append(processed_messages[-1])
            self.memory.append(response_message)

        # Record end time
        self.run_meta["end_time"] = datetime.now()
        self.run_meta["duration"] = (
            self.run_meta["end_time"] - self.run_meta["start_time"]
        ).total_seconds()

        return response_message

    def _call_with_spans(
        self,
        human_messages: Union[str, BaseMessage, list[Union[str, BaseMessage]]],
        outer_span: Span,
    ) -> Union[AIMessage, Generator]:
        """Internal method for calling bot with spans."""
        processed_messages = to_basemessage(human_messages)

        memory_messages = []
        if self.memory:
            memory_messages = self.memory.retrieve(
                query=f"From our conversation history, give me the most relevant information to the query, {[p.content for p in processed_messages]}"
            )

        messages = [self.system_prompt] + memory_messages + processed_messages

        # Count initial messages and record in span
        user_count = sum(1 for msg in messages if isinstance(msg, HumanMessage))
        assistant_count = sum(1 for msg in messages if isinstance(msg, AIMessage))
        outer_span["input_message_count"] = len(messages)
        outer_span["input_user_messages"] = user_count
        outer_span["input_assistant_messages"] = assistant_count

        # Create span for LLM request
        with outer_span.span("llm_request", model=self.model_name) as llm_request_span:
            llm_request_span["message_count"] = len(messages)
            llm_request_span["temperature"] = self.temperature

            stream = self.stream_target != "none"
            response = make_response(self, messages, stream)
            response = stream_chunks(response, target=self.stream_target)
            tool_calls = extract_tool_calls(response)
            content = extract_content(response)
            response_message = AIMessage(content=content, tool_calls=tool_calls)

        # Create span for LLM response processing
        with outer_span.span("llm_response") as llm_response_span:
            llm_response_span["response_length"] = len(content) if content else 0
            llm_response_span["tool_calls_count"] = len(tool_calls) if tool_calls else 0

            # Record tool usage in span if any
            if tool_calls:
                tool_names = [call.function.name for call in tool_calls]
                outer_span["tool_calls"] = ", ".join(tool_names)
                outer_span["tool_calls_count"] = len(tool_calls)

            sqlite_log(self, messages + [response_message])
            if self.memory:
                self.memory.append(processed_messages[-1])
                self.memory.append(response_message)

        return response_message

    def display_spans(self) -> str:
        """Display all spans from all bot calls as HTML.

        Queries spans associated with all trace_ids from this bot instance
        and generates an HTML visualization showing all spans from all calls.

        :return: HTML string for displaying spans in marimo notebooks
        """
        if not self._trace_ids:
            return '<div style="padding: 1rem; color: #2E3440;">No spans recorded for this bot instance yet.</div>'

        # Collect all spans from all trace_ids for this bot instance
        all_spans_objects = []
        for trace_id in self._trace_ids:
            spans = get_spans(trace_id=trace_id)
            # SpanList is iterable, so we can extend with it
            all_spans_objects.extend(spans)

        if not all_spans_objects:
            return '<div style="padding: 1rem; color: #2E3440;">No spans found in database for this bot instance.</div>'

        # Convert Span objects to dictionaries for visualization
        all_spans = [span_to_dict(s) for s in all_spans_objects]

        # Find root spans (spans with no parent) to use as current span
        # Use the most recent root span (last one in the list)
        root_spans = [s for s in all_spans if s.get("parent_span_id") is None]
        if root_spans:
            # Use the last root span (most recent) as the current span for highlighting
            current_span_dict = root_spans[-1]
            current_span_id = current_span_dict["span_id"]
        else:
            # Fallback to last span if no root spans found
            current_span_dict = all_spans[-1]
            current_span_id = current_span_dict["span_id"]

        # Build hierarchical structure
        trace_tree = build_hierarchy(all_spans)

        # Generate HTML visualization
        return generate_span_html(
            span_dict=current_span_dict,
            all_spans=all_spans,
            trace_tree=trace_tree,
            current_span_id=current_span_id,
        )

    def _repr_html_(self) -> str:
        """Return HTML representation for marimo display.

        When a SimpleBot object is the last expression in a marimo cell,
        this method is automatically called to display the spans visualization
        from the most recent bot call.

        :return: HTML string for displaying spans
        """
        return self.display_spans()


def make_response(
    bot: SimpleBot, messages: list[BaseMessage], stream: bool = True
) -> ModelResponse | CustomStreamWrapper:
    """Make a response from the given messages.

    :param bot: A SimpleBot
    :param messages: A list of Messages.
    :return: A response object.
    """
    from litellm import completion

    messages_dumped: list[dict] = [
        {
            k: v
            for k, v in m.model_dump(model_name=bot.model_name).items()
            if k in ("role", "content")
        }
        for m in messages
    ]
    completion_kwargs = dict(
        model=bot.model_name,
        messages=messages_dumped,
        temperature=bot.temperature,
        stream=stream,
    )
    completion_kwargs.update(bot.completion_kwargs)

    if bot.mock_response:
        completion_kwargs["mock_response"] = bot.mock_response
    if bot.json_mode:
        # Check if bot has pydantic_model attribute and it's a BaseModel
        if not hasattr(bot, "pydantic_model"):
            raise ValueError(
                "Please set a pydantic_model for this bot to use JSON mode!"
            )
        if not issubclass(getattr(bot, "pydantic_model"), BaseModel):
            raise ValueError("pydantic_model must be a Pydantic BaseModel class")

        completion_kwargs["response_format"] = getattr(bot, "pydantic_model")
    if bot.api_key:
        completion_kwargs["api_key"] = bot.api_key
    if hasattr(bot, "tools"):
        logger.debug(f"Passing in tools: {bot.tools}")
        completion_kwargs["tools"] = bot.tools
        # Allow bots to specify their own tool_choice, default to "auto"
        tool_choice = getattr(bot, "tool_choice", "auto")
        completion_kwargs["tool_choice"] = tool_choice

    logger.debug("Completion kwargs: {}", completion_kwargs)
    return completion(**completion_kwargs)


def stream_chunks(
    response: ModelResponse | CustomStreamWrapper, target="stdout"
) -> ModelResponse | Generator:
    """Stream the response from a `completion` call.

    This will work whether or not the response is actually streamed or not.

    :param response: The response from a `completion` call.
    :param target: The target to stream the response to.
    :return: The response from the `completion` call.
    """
    # Pass through if it's already a ModelResponse
    if isinstance(response, ModelResponse):
        return response

    # Create a separate generator function for the streaming part
    def generate_stream():
        """Generate a stream of chunks from the response.

        This function is inside to prevent the outer `_stream_chunks` function
        from being a generator.

        :param response: The response from a `completion` call.
        :return: A generator of chunks.
        """
        chunks = []
        response_message = ""  # only used for target == 'panel'
        for chunk in response:
            chunks.append(chunk)
            delta = chunk.choices[0].delta["content"]
            if delta is not None:
                if target == "panel":
                    response_message += delta
                    yield response_message
                elif target == "api":
                    yield delta
        return stream_chunk_builder(chunks)

    # If we need to yield, return the generator
    if target in ["panel", "api"]:
        return generate_stream()

    # Otherwise, assume it's a generator of chunks
    chunks = []
    for chunk in response:
        chunks.append(chunk)
        delta = chunk.choices[0].delta["content"]
        if delta is not None:
            if target == "stdout":
                print(delta, end="")
    return stream_chunk_builder(chunks)


def serialize_tool_arguments(args: Union[dict, str, None]) -> str:
    """Safely serialize tool arguments to JSON string.

    Handles cases where arguments might already be a JSON string,
    a dict, or None.

    :param args: Arguments that may be a dict, JSON string, or None
    :return: JSON string representation of arguments
    """
    if args is None:
        return "{}"
    if isinstance(args, str):
        # If it's already a string, validate it's valid JSON
        try:
            json.loads(args)  # Validate it's valid JSON
            return args  # Return as-is if valid
        except json.JSONDecodeError:
            # If not valid JSON, treat as a string argument value
            return json.dumps(args)
    # If it's a dict or other object, serialize it
    return json.dumps(args)


def extract_tool_calls(response: ModelResponse) -> list[ChatCompletionMessageToolCall]:
    """Extract the tool calls from the response.

    Handles both structured tool_calls (OpenAI-style) and JSON in content field
    (Ollama-style models that return tool calls as JSON strings).

    :param response: The response from a `completion` call.
    :return: A list of tool calls.
    """
    message = response.choices[0].message
    tool_calls: List[ChatCompletionMessageToolCall] | NoneType = message.tool_calls

    # If structured tool_calls exist, return them
    if tool_calls is not None:
        return tool_calls

    # Fallback: Check if content contains JSON tool call
    # Some models (e.g., Ollama) return tool calls as JSON in content field
    if message.content:
        try:
            # Try to parse JSON from content
            content_json = json.loads(message.content.strip())

            # Check if it's a single tool call object
            if isinstance(content_json, dict) and "name" in content_json:
                # Single tool call: {"name": "tool_name", "arguments": {...}}
                tool_name = content_json.get("name", "unknown")
                function = Function(
                    name=tool_name,
                    arguments=serialize_tool_arguments(content_json.get("arguments")),
                )
                tool_call = ChatCompletionMessageToolCall(
                    function=function, id=f"call_{tool_name}", type="function"
                )
                return [tool_call]

            # Check if it's a list of tool calls
            elif isinstance(content_json, list) and len(content_json) > 0:
                parsed_tool_calls = []
                for idx, tc in enumerate(content_json):
                    if isinstance(tc, dict) and "name" in tc:
                        tool_name = tc.get("name", "unknown")
                        function = Function(
                            name=tool_name,
                            arguments=serialize_tool_arguments(tc.get("arguments")),
                        )
                        tool_call = ChatCompletionMessageToolCall(
                            function=function,
                            id=f"call_{tool_name}_{idx}",
                            type="function",
                        )
                        parsed_tool_calls.append(tool_call)
                if parsed_tool_calls:
                    return parsed_tool_calls

        except json.JSONDecodeError:
            # Content is not valid JSON - silently return empty list
            pass
        except (KeyError, TypeError):
            # Unexpected structure in JSON - log and return empty list
            # This handles cases where JSON structure doesn't match expected format
            pass

    return []


def extract_content(response: ModelResponse) -> str:
    """Extract the content from the response.

    :param response: The response from a `completion` call.
    :return: The content of the response.
    """
    content: str | NoneType = response.choices[0].message.content
    if content is None:
        return ""
    return content
