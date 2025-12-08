"""Implementation source code for PydanticBot.

Courtesey of Elliot Salisbury (@ElliotSalisbury).

Highly inspired by instructor by jxnl (https://github.com/jxnl/instructor).
"""

import inspect
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
    get_caller_variable_name,
    get_current_span,
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

        # Try to get the variable name from the calling frame
        operation_name = get_caller_variable_name(self)
        if operation_name is None:
            operation_name = "structuredbot_call"

        # Check if there's a current span - if so, create a child span
        current_span = get_current_span()
        if current_span:
            # Create child span using parent's trace_id
            # Get model class name - pydantic_model is a class, not an instance
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
                num_attempts=num_attempts,
            )
            # Track trace_id for this bot instance (even for child spans)
            if outer_span.trace_id not in self._trace_ids:
                self._trace_ids.append(outer_span.trace_id)
        else:
            # No current span - create a new trace
            # Get model class name - pydantic_model is a class, not an instance
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
                num_attempts=num_attempts,
            )
            # Track trace_id for this bot instance
            if outer_span.trace_id not in self._trace_ids:
                self._trace_ids.append(outer_span.trace_id)
        with outer_span:
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

            # Extract query content from messages for span
            query_content = " ".join(
                [
                    msg.content if hasattr(msg, "content") else str(msg)
                    for msg in messages
                ]
            )
            outer_span["query"] = query_content[:500]  # Truncate for storage
            outer_span["temperature"] = self.temperature

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
                    validation_time = (
                        datetime.now() - validation_start
                    ).total_seconds()
                    outer_span["validation_attempts"] = validation_attempts
                    outer_span["validation_success"] = True
                    outer_span["validation_time"] = validation_time
                    # Record response content
                    outer_span["response"] = (
                        content[:500] if content else None
                    )  # Truncate for storage
                    outer_span["response_length"] = len(content) if content else 0

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
        return self.spans._repr_html_()

    def format_field_type(self, field_info: dict, schema: dict) -> str:
        """Format a field type from JSON schema to readable string.

        Handles various JSON schema type formats including arrays, unions,
        optional fields, and nested model references.

        :param field_info: Field schema information dictionary
        :param schema: Complete schema dictionary (for $ref lookups)
        :return: Formatted type string (e.g., "array of string", "string | null")
        """
        # Handle $ref (nested models)
        if "$ref" in field_info:
            ref_path = field_info["$ref"].split("/")[-1]
            return ref_path

        # Handle anyOf (unions)
        if "anyOf" in field_info:
            types = []
            for option in field_info["anyOf"]:
                if option.get("type") == "null":
                    types.append("null")
                elif "$ref" in option:
                    ref_name = option["$ref"].split("/")[-1]
                    types.append(ref_name)
                elif "type" in option:
                    types.append(option["type"])
            return " | ".join(types)

        # Handle arrays
        if field_info.get("type") == "array":
            if "items" in field_info:
                items = field_info["items"]
                if "$ref" in items:
                    item_type = items["$ref"].split("/")[-1]
                else:
                    item_type = items.get("type", "any")
                return f"array of {item_type}"
            return "array"

        # Simple type
        return field_info.get("type", "any")

    def render_field(
        self, name: str, info: dict, required: bool, schema: dict, indent_level: int = 0
    ) -> str:
        """Render a single Pydantic model field as HTML.

        :param name: Field name
        :param info: Field schema information
        :param required: Whether the field is required
        :param schema: Complete schema dictionary (for $ref lookups)
        :param indent_level: Indentation level for nested fields
        :return: HTML string for the field
        """
        field_type = self.format_field_type(info, schema)
        required_badge = (
            '<span style="color: #BF616A; font-size: 0.85rem; font-weight: 600;">required</span>'
            if required
            else '<span style="color: #81A1C1; font-size: 0.85rem;">optional</span>'
        )

        # Extract default value
        default_html = ""
        if "default" in info:
            default_val = str(info["default"])
            default_html = f'<div style="color: #A3BE8C; font-size: 0.85rem; font-family: monospace; margin-top: 0.25rem;">Default: {default_val}</div>'

        # Extract description
        description_html = ""
        if "description" in info:
            description_html = f'<div style="color: #4C566A; font-size: 0.9rem; margin-top: 0.25rem; font-style: italic;">{info["description"]}</div>'

        indent_style = (
            f"margin-left: {indent_level * 1.5}rem;" if indent_level > 0 else ""
        )

        return f"""
        <div style="background: white; padding: 0.75rem; margin-bottom: 0.5rem; border-radius: 4px; border-left: 3px solid #A3BE8C; {indent_style}">
            <div>
                <span style="font-weight: 600; color: #2E3440; font-family: monospace;">{name}</span>:
                <span style="color: #5E81AC; font-family: monospace; font-size: 0.9rem;">{field_type}</span>
                ({required_badge})
            </div>
            {default_html}
            {description_html}
        </div>
        """

    def generate_schema_html(self) -> str:
        """Generate HTML representation of the Pydantic model schema.

        Extracts schema from self.pydantic_model and creates a beautiful
        HTML rendering showing all fields with their types, requirements,
        defaults, and descriptions.

        :return: HTML string showing the Pydantic model schema
        """
        schema = self.pydantic_model.model_json_schema()
        model_name = schema.get("title", self.pydantic_model.__name__)
        properties = schema.get("properties", {})
        required_fields = set(schema.get("required", []))

        # Render all fields
        fields_html = []
        for field_name, field_info in properties.items():
            is_required = field_name in required_fields
            field_html = self.render_field(field_name, field_info, is_required, schema)
            fields_html.append(field_html)

        html = f"""
        <div style="margin-top: 1rem;">
            <div style="color: #2E3440; font-size: 1.05rem; font-weight: 600; border-bottom: 2px solid #5E81AC; padding-bottom: 0.5rem; margin-bottom: 0.75rem;">
                Pydantic Model Schema
            </div>
            <div style="margin-bottom: 0.5rem; color: #2E3440;">
                <span style="font-weight: 600;">Model:</span> <span style="font-family: monospace; color: #5E81AC;">{model_name}</span>
            </div>
            <div style="margin-top: 0.75rem;">
                <div style="font-weight: 600; color: #2E3440; margin-bottom: 0.5rem;">Fields:</div>
                {''.join(fields_html)}
            </div>
        </div>
        """
        return html

    def _repr_html_(self) -> str:
        """Return HTML representation for marimo display.

        When a StructuredBot object is the last expression in a marimo cell,
        this method is automatically called to display the bot's configuration
        and Pydantic model schema.

        :return: HTML string for displaying bot configuration and schema
        """
        config_html = self.generate_config_html()
        schema_html = self.generate_schema_html()

        # Combine both, inserting schema_html before the closing divs of config
        # Extract the content part of config_html (before final closing tags)
        config_parts = config_html.rsplit("</div>", 2)
        combined_html = config_parts[0] + schema_html + "</div>" + "</div>"

        # Update the header to say "StructuredBot"
        combined_html = combined_html.replace(
            "SimpleBot Configuration", "StructuredBot Configuration"
        )

        return combined_html
