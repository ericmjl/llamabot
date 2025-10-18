"""ToolBot - A single-turn bot that can execute tools via MCP."""

import json
import asyncio
from typing import Callable, List, Optional, Union
from loguru import logger

from fastmcp import Client

# Note: today_date and respond_to_user are now handled via MCP
from llamabot.components.chat_memory import ChatMemory
from llamabot.components.messages import AIMessage, BaseMessage
from llamabot.bot.simplebot import (
    SimpleBot,
    extract_tool_calls,
    make_response,
    stream_chunks,
)
from llamabot.prompt_manager import prompt


@prompt("system")
def toolbot_sysprompt(globals_dict: dict = {}) -> str:
    """
    You are a ToolBot, an intelligent agent designed to analyze user requests and determine the most appropriate tool or function to execute.

    Your primary responsibilities:
    1. **Analyze the user's request** to understand what they want to accomplish
    2. **Select the most appropriate tool** from your available function toolkit
    3. **Extract or infer the necessary arguments** for the selected function
    4. **Return a single function call** with the proper arguments to execute

    ## Available Tools:
    You have access to tools through function calling. Each tool has detailed documentation in its docstring that explains:
    - When to use the tool
    - What parameters it expects
    - What it returns
    - Usage examples and guidelines

    ## Decision Process:
    When you receive a user request:
    - Break down what the user is asking for
    - Identify the core action or information needed
    - Map this to one of your available tools by reading their docstrings
    - Determine the required parameters/arguments
    - Make the function call with appropriate arguments

    Remember: You are a function selector and executor. Read the tool docstrings carefully to understand when and how to use each tool effectively.

    ## Available Global Variables:

    The available dataframes are:

    {% for k, v in globals_dict.items() %}
        {% if v is defined and v is not none and v.__class__.__name__ == 'DataFrame' %}
    - {{ k }}: DataFrame with shape {{ v.shape }} and columns {{ v.columns | list }}
        {% endif %}
    {% endfor %}

    The available callables are:

    {% for k, v in globals_dict.items() %}
        {% if v is defined and v is not none and v.__call__ is not none %}
    - {{ k }}: {{ v.__class__.__name__ }}
        {% endif %}
    {% endfor %}

    The available other variables are:

    {% for k, v in globals_dict.items() %}
        {% if v is defined and v is not none and v.__call__ is none and v.__class__.__name__ != 'DataFrame' %}
    - {{ k }}: {{ v.__class__.__name__ }}
        {% endif %}
    {% endfor %}
    """


class ToolBot(SimpleBot):
    """A single-turn bot that can execute tools via MCP.

    This bot is designed to analyze user requests and determine the most appropriate
    tool or function to execute. It uses MCP clients to access both local and remote
    tools through a unified interface.

    :param system_prompt: The system prompt to use
    :param model_name: The name of the model to use
    :param mcp_clients: Optional list of FastMCP Client instances
    :param chat_memory: Chat memory component for context retrieval
    :param completion_kwargs: Additional keyword arguments for completion
    """

    def __init__(
        self,
        system_prompt: str,
        model_name: str,
        mcp_clients: Optional[List[Client]] = None,
        chat_memory: Optional[ChatMemory] = None,
        **completion_kwargs,
    ):
        super().__init__(
            system_prompt=system_prompt,
            model_name=model_name,
            **completion_kwargs,
        )

        self.mcp_clients = mcp_clients or []
        self._tool_schemas = None  # Lazy load
        self.chat_memory = chat_memory or ChatMemory()

    def _run_async_in_sync(self, coro):
        """Run async coroutine in a sync context, handling both regular and notebook environments."""
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # If we're in a running loop (like Jupyter), use nest_asyncio
            import nest_asyncio

            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        except RuntimeError:
            # No event loop running, safe to use asyncio.run()
            return asyncio.run(coro)

    async def _load_tool_schemas(self):
        """Load tool schemas from all MCP clients.

        :return: List of tool schemas for LLM function calling
        """
        if self._tool_schemas is not None:
            return self._tool_schemas

        schemas = []
        for client in self.mcp_clients:
            try:
                async with client:
                    tools = await client.list_tools()
                    for tool in tools:
                        schemas.append(
                            {
                                "type": "function",
                                "function": {
                                    "name": tool.name,
                                    "description": tool.description,
                                    "parameters": tool.inputSchema,
                                },
                            }
                        )
            except Exception as e:
                logger.warning(f"Failed to load tools from MCP client: {e}")
                continue

        self._tool_schemas = schemas
        return schemas

    async def execute_tool_call(self, tool_call):
        """Execute a tool call via MCP clients.

        :param tool_call: The tool call to execute
        :return: The result of the tool call
        """
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        # Try each client until we find the tool
        for client in self.mcp_clients:
            try:
                async with client:
                    result = await client.call_tool(tool_name, arguments)
                    # Extract the actual result from CallToolResult
                    if hasattr(result, "data"):
                        return result.data
                    elif hasattr(result, "content") and result.content:
                        return result.content[0].text if result.content else str(result)
                    else:
                        return str(result)
            except Exception as e:
                logger.debug(f"Tool {tool_name} not found on client: {e}")
                # Tool not found on this client, try next
                continue

        raise ValueError(f"Tool {tool_name} not found on any MCP server")

    def __call__(
        self,
        *messages: Union[str, BaseMessage, list[Union[str, BaseMessage]], Callable],
    ):
        """Process messages and return tool calls.

        :param messages: One or more messages to process. Can be strings, BaseMessage objects, or callable functions.
        :return: List of tool calls to execute
        """
        # Handle async operations in a notebook-friendly way
        return self._run_async_in_sync(self._async_call(messages))

    async def _async_call(self, messages):
        """Async implementation of tool call processing.

        :param messages: Messages to process
        :return: List of tool calls to execute
        """
        from llamabot.components.messages import to_basemessage, HumanMessage

        # Handle callable functions by calling them and converting to strings
        processed_messages = []
        for msg in messages:
            if callable(msg):
                # Call the function and validate it returns a string
                result = msg()
                if not isinstance(result, str):
                    raise ValueError("Callable function must return a string")
                processed_messages.append(HumanMessage(content=result))
            else:
                processed_messages.append(msg)

        # Convert messages to BaseMessage objects using the same utility as other bots
        user_messages = to_basemessage(processed_messages)

        # Build message list: system prompt, chat memory, then user messages
        message_list = [self.system_prompt]
        if self.chat_memory and user_messages:
            # Use the first message content for chat memory retrieval
            message_list.extend(self.chat_memory.retrieve(user_messages[0].content))
        message_list.extend(user_messages)

        # Load tool schemas from MCP clients
        self.tools = await self._load_tool_schemas()

        # Execute the plan
        stream = self.stream_target != "none"
        logger.debug("Message list: {}", message_list)
        response = make_response(self, message_list, stream=stream)
        response = stream_chunks(response, target=self.stream_target)
        logger.debug("Response: {}", response)
        tool_calls = extract_tool_calls(response)

        if user_messages:
            self.chat_memory.append(
                user_messages[0], AIMessage(content=str(tool_calls))
            )

        return tool_calls
