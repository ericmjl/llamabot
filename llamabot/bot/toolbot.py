"""ToolBot - A single-turn bot that can execute tools."""

from typing import Callable, List, Optional, Union
from loguru import logger

from llamabot.components.tools import today_date, respond_to_user
from llamabot.components.chat_memory import ChatMemory
from llamabot.components.messages import AIMessage, BaseMessage, to_basemessage
from llamabot.bot.simplebot import (
    SimpleBot,
    extract_content,
    extract_tool_calls,
    make_response,
    stream_chunks,
)
from litellm import stream_chunk_builder


class ToolBot(SimpleBot):
    """A single-turn bot that can execute tools.

    This bot is designed to analyze user requests and determine the most appropriate
    tool or function to execute. It's a generalization of other bot types, focusing
    on tool selection and execution rather than multi-turn conversation.

    :param system_prompt: The system prompt to use
    :param model_name: The name of the model to use
    :param tools: Optional list of additional tools to include
    :param chat_memory: Chat memory component for context retrieval
    :param completion_kwargs: Additional keyword arguments for completion
    """

    def __init__(
        self,
        system_prompt: str,
        model_name: str,
        tools: Optional[List[Callable]] = None,
        chat_memory: Optional[ChatMemory] = None,
        **completion_kwargs,
    ):
        super().__init__(
            system_prompt=system_prompt,
            model_name=model_name,
            **completion_kwargs,
        )

        # Initialize with core tools
        all_tools = [today_date, respond_to_user]
        if tools is not None:
            all_tools.extend([f for f in tools])

        self.tools = [f.json_schema for f in all_tools]
        self.name_to_tool_map = {f.__name__: f for f in all_tools}
        self.chat_memory = chat_memory or ChatMemory()

    def __call__(
        self, *human_messages: Union[str, BaseMessage, list[Union[str, BaseMessage]]]
    ) -> list:
        """Process a message and return tool calls.

        :param human_messages: One or more human messages to process
        :return: List of tool calls to execute
        """
        # Convert message to BaseMessage format
        processed_messages = to_basemessage(human_messages)

        # Get memory messages if available
        memory_messages = []
        if self.chat_memory:
            memory_messages = self.chat_memory.retrieve(
                query=f"From our conversation history, give me the most relevant information to the query, {[p.content for p in processed_messages]}"
            )

        # Build message list - cast to list[BaseMessage] to satisfy type checker
        messages = [self.system_prompt] + memory_messages + processed_messages
        messages = list(messages)  # Ensure it's a list

        # Execute the plan
        stream = self.stream_target != "none"
        logger.debug("Message list: {}", messages)
        response = make_response(self, messages, stream=stream)
        response = stream_chunks(response, target=self.stream_target)
        logger.debug("Response: {}", response)

        # Handle streaming response
        if hasattr(response, "__iter__") and not hasattr(response, "choices"):
            # It's a generator, we need to consume it
            chunks = []
            for chunk in response:
                chunks.append(chunk)
            response = stream_chunk_builder(chunks)

        tool_calls = extract_tool_calls(response)
        content = extract_content(response)

        # Store in chat memory
        if self.chat_memory:
            self.chat_memory.append(processed_messages[-1], AIMessage(content=content))

        return tool_calls
