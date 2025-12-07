"""Implementation source code for PydanticBot.

Courtesey of Elliot Salisbury (@ElliotSalisbury).

Highly inspired by instructor by jxnl (https://github.com/jxnl/instructor).
"""

import json
import uuid
from datetime import datetime
from typing import Union

from litellm import get_supported_openai_params, supports_response_schema
from loguru import logger
from pydantic import BaseModel, ValidationError

from llamabot.bot.simplebot import (
    SimpleBot,
    extract_content,
    make_response,
    stream_chunks,
)
from llamabot.components.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    to_basemessage,
)
from llamabot.config import default_language_model
from llamabot.prompt_manager import prompt
from llamabot.recorder import (
    Span,
    build_hierarchy,
    generate_span_html,
    get_current_span,
    get_spans,
    span_to_dict,
    sqlite_log,
)


@prompt(role="user")
def bot_task() -> BaseMessage:
    """Your task is to return the data in a json object
    that matches the provided json_schema:
    """


class StructuredBot(SimpleBot):
    """StructuredBot is given a Pydantic Model and expects the LLM to return
    a JSON structure that conforms to the model schema.
    It will validate the returned json against the pydantic model,
    prompting the LLM to fix any of the validation errors if it does not validate,
    and then explicitly return an instance of that model.

    StructuredBot streams only to stdout; other modes of streaming are not supported.

    This is distinct from SimpleBot's JSON-mode behaviour in the following ways:

    1. JSON mode ensures a valid JSON is returned, but it does not guarantee it's in the schema you requested.
    2. We can customize the validation rules to our needs. For example: checking for hallucinations.
    3. Explicitly controllable number of retries.
    4. Directly returning an instance of the Pydantic model rather than just a plain JSON string.
    """

    def __init__(
        self,
        system_prompt: Union[str, SystemMessage],
        pydantic_model: BaseModel,
        model_name: str = default_language_model(),
        stream_target: str = "stdout",
        allow_failed_validation: bool = False,
        **completion_kwargs,
    ):
        params = get_supported_openai_params(model=model_name)
        # Special case for ollama_chat - it supports structured outputs
        if "ollama_chat" in model_name:
            pass  # Ollama chat supports structured outputs
        elif not (
            "response_format" in params and supports_response_schema(model=model_name)
        ):
            raise ValueError(
                f"Model {model_name} does not support structured responses. "
                "Please use a model that supports both `response_format` and `response_schema`; "
                "`gpt-4`, `anthropic/claude-3-5-sonnet`, "
                "and gemini/gemini-1.5-pro-latest are specific examples, "
                "just make sure you have the appropriate API key for the model."
            )

        super().__init__(
            system_prompt,
            stream_target=stream_target,
            json_mode=True,
            model_name=model_name,
            **completion_kwargs,
        )

        self.pydantic_model = pydantic_model
        self.allow_failed_validation = allow_failed_validation

        # Track all trace_ids from this bot instance for span visualization
        self._trace_ids = []

    def get_validation_error_message(self, exception: ValidationError) -> HumanMessage:
        """Return a validation error message to feed back to the bot.

        :param exception: The raised ValidationError from pydantic.
        """
        exception_msg = exception.json()
        return HumanMessage(
            content=f"There was a validation error: {exception_msg} try again."
        )

    def __call__(
        self,
        *messages: Union[str, BaseMessage, list[Union[str, BaseMessage]]],
        num_attempts: int = 10,
        verbose: bool = False,
    ) -> BaseModel | None:
        """Process the input messages and return an instance of the Pydantic model.

        :param messages: One or more messages to process. Can be strings or BaseMessage objects.
        :param num_attempts: Number of attempts to try getting a valid response.
        :param verbose: Whether to show verbose output.
        :return: An instance of the specified Pydantic model.
        """
        # Compose the full message list
        messages = to_basemessage(messages)
        query_content = " ".join(
            [msg.content if hasattr(msg, "content") else str(msg) for msg in messages]
        )

        # Check if there's a current span - if so, create a child span
        current_span = get_current_span()
        if current_span:
            # Create child span using parent's trace_id
            outer_span = Span(
                "structuredbot_call",
                trace_id=current_span.trace_id,
                parent_span_id=current_span.span_id,
                query=query_content,
                model=self.model_name,
                pydantic_model=self.pydantic_model.__name__,
                num_attempts=num_attempts,
            )
        else:
            # No current span - create a new trace
            new_trace_id = str(uuid.uuid4())
            outer_span = Span(
                "structuredbot_call",
                trace_id=new_trace_id,
                query=query_content,
                model=self.model_name,
                pydantic_model=self.pydantic_model.__name__,
                num_attempts=num_attempts,
            )
            # Track trace_id for this bot instance (only for root spans)
            if outer_span.trace_id not in self._trace_ids:
                self._trace_ids.append(outer_span.trace_id)
        with outer_span:
            return self._call_with_spans(messages, num_attempts, verbose, outer_span)

    def _call_without_spans(
        self,
        messages: list[BaseMessage],
        num_attempts: int,
        verbose: bool,
    ) -> BaseModel | None:
        """Internal method for calling bot without spans."""
        # Initialize run metadata
        self.run_meta = {
            "start_time": datetime.now(),
            "validation_attempts": 0,
            "validation_success": False,
            "schema_complexity": {
                "fields": len(self.pydantic_model.model_fields),
                "nested_models": len(
                    [
                        f
                        for f in self.pydantic_model.model_fields.values()
                        if hasattr(f.annotation, "__origin__")
                        and f.annotation.__origin__ is BaseModel
                    ]
                ),
            },
        }

        full_messages = [self.system_prompt] + messages

        last_response = None
        last_codeblock = None
        # we'll attempt to get the response from the model and validate it
        for attempt in range(num_attempts):
            try:
                self.run_meta["validation_attempts"] += 1
                validation_start = datetime.now()

                # Create full messages list for this attempt
                for message in full_messages:
                    if isinstance(message, str):
                        print(message)

                # Get response using the same pattern as SimpleBot
                response = make_response(
                    self, full_messages, self.stream_target != "none"
                )
                response = stream_chunks(response, target=self.stream_target)

                # Extract content from the response
                content = extract_content(response)

                # parse the response, and validate it against the pydantic model
                last_response = AIMessage(content=content)
                sqlite_log(self, messages + [last_response])

                last_codeblock = json.loads(last_response.content)
                obj = self.pydantic_model.model_validate(last_codeblock)

                # Record successful validation
                self.run_meta["validation_success"] = True
                self.run_meta["validation_time"] = (
                    datetime.now() - validation_start
                ).total_seconds()
                self.run_meta["end_time"] = datetime.now()
                self.run_meta["duration"] = (
                    self.run_meta["end_time"] - self.run_meta["start_time"]
                ).total_seconds()

                return obj

            except ValidationError as e:
                # Record validation error
                self.run_meta["last_validation_error"] = str(e)

                # we're on our last try
                if attempt == num_attempts - 1:
                    if self.allow_failed_validation and last_codeblock is not None:
                        # If we allow failed validation, return the last attempt
                        self.run_meta["end_time"] = datetime.now()
                        self.run_meta["duration"] = (
                            self.run_meta["end_time"] - self.run_meta["start_time"]
                        ).total_seconds()
                        return self.pydantic_model.model_construct(**last_codeblock)
                    raise e

                # Otherwise, if we failed, give the LLM the validation error and try again.
                if verbose:
                    logger.info(e)
                full_messages.extend(
                    [
                        AIMessage(content=last_response.content),
                        self.get_validation_error_message(e),
                    ]
                )

    def _call_with_spans(
        self,
        messages: list[BaseMessage],
        num_attempts: int,
        verbose: bool,
        outer_span: Span,
    ) -> BaseModel | None:
        """Internal method for calling bot with spans."""
        # Record schema complexity in span
        schema_fields = len(self.pydantic_model.model_fields)
        nested_models = len(
            [
                f
                for f in self.pydantic_model.model_fields.values()
                if hasattr(f.annotation, "__origin__")
                and f.annotation.__origin__ is BaseModel
            ]
        )
        outer_span["schema_fields"] = schema_fields
        outer_span["schema_nested_models"] = nested_models

        full_messages = [self.system_prompt] + messages

        last_response = None
        last_codeblock = None
        validation_attempts = 0
        # we'll attempt to get the response from the model and validate it
        for attempt in range(num_attempts):
            try:
                validation_attempts += 1
                validation_start = datetime.now()

                # Create full messages list for this attempt
                for message in full_messages:
                    if isinstance(message, str):
                        print(message)

                # Get response using the same pattern as SimpleBot
                response = make_response(
                    self, full_messages, self.stream_target != "none"
                )
                response = stream_chunks(response, target=self.stream_target)

                # Extract content from the response
                content = extract_content(response)

                # parse the response, and validate it against the pydantic model
                last_response = AIMessage(content=content)
                sqlite_log(self, messages + [last_response])

                last_codeblock = json.loads(last_response.content)
                obj = self.pydantic_model.model_validate(last_codeblock)

                # Record successful validation in span
                validation_time = (datetime.now() - validation_start).total_seconds()
                outer_span["validation_attempts"] = validation_attempts
                outer_span["validation_success"] = True
                outer_span["validation_time"] = validation_time

                return obj

            except ValidationError as e:
                # Record validation error in span
                outer_span["validation_attempts"] = validation_attempts
                outer_span["last_validation_error"] = str(e)

                # we're on our last try
                if attempt == num_attempts - 1:
                    if self.allow_failed_validation and last_codeblock is not None:
                        outer_span["validation_success"] = False
                        outer_span["allow_failed_validation"] = True
                        return self.pydantic_model.model_construct(**last_codeblock)
                    outer_span["validation_success"] = False
                    raise e

                # Otherwise, if we failed, give the LLM the validation error and try again.
                if verbose:
                    logger.info(e)
                full_messages.extend(
                    [
                        AIMessage(content=last_response.content),
                        self.get_validation_error_message(e),
                    ]
                )

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

        When a StructuredBot object is the last expression in a marimo cell,
        this method is automatically called to display the spans visualization
        from all bot calls.

        :return: HTML string for displaying spans
        """
        return self.display_spans()
