"""Class definition for SimpleBot."""

import contextvars
from types import NoneType
from typing import Generator, List, Optional, Union
from datetime import datetime

from llamabot.recorder import sqlite_log
from loguru import logger


from llamabot.components.docstore import AbstractDocumentStore
from llamabot.components.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
    to_basemessage,
)
from llamabot.config import default_language_model
from pydantic import BaseModel
from litellm import (
    ChatCompletionMessageToolCall,
    CustomStreamWrapper,
    ModelResponse,
    stream_chunk_builder,
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

    def __call__(
        self, *human_messages: Union[str, BaseMessage, list[Union[str, BaseMessage]]]
    ) -> Union[AIMessage, Generator]:
        """Call the SimpleBot.

        :param human_messages: One or more human messages to use, or lists of messages.
        :return: The response to the human messages, primed by the system prompt.
        """
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
        {k: v for k, v in m.model_dump().items() if k in ("role", "content")}
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


def extract_tool_calls(response: ModelResponse) -> list[ChatCompletionMessageToolCall]:
    """Extract the tool calls from the response.

    :param response: The response from a `completion` call.
    :return: A list of tool calls.
    """
    tool_calls: List[ChatCompletionMessageToolCall] | NoneType = response.choices[
        0
    ].message.tool_calls
    if tool_calls is None:
        return []
    return tool_calls


def extract_content(response: ModelResponse) -> str:
    """Extract the content from the response.

    :param response: The response from a `completion` call.
    :return: The content of the response.
    """
    content: str | NoneType = response.choices[0].message.content
    if content is None:
        return ""
    return content
