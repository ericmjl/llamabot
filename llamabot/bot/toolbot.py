"""ToolBot - A single-turn bot that can execute tools."""

from typing import Callable, Dict, List, Optional, Union
from loguru import logger

from llamabot.components.chat_memory import ChatMemory
from llamabot.components.messages import AIMessage, BaseMessage
from llamabot.components.tools import DEFAULT_TOOLS
from llamabot.bot.simplebot import (
    SimpleBot,
    extract_tool_calls,
    make_response,
    stream_chunks,
)
from llamabot.prompt_manager import prompt


@prompt("system")
def toolbot_sysprompt(globals_dict: dict = {}, categorized_vars: dict = None) -> str:
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

    {% for name, class_name in categorized_vars.dataframes %}
    - {{ name }}: {{ class_name }}
    {% endfor %}

    The available callables are:

    {% for name, class_name in categorized_vars.callables %}
    - {{ name }}: {{ class_name }}
    {% endfor %}

    The available other variables are:

    {% for name, class_name in categorized_vars.other %}
    - {{ name }}: {{ class_name }}
    {% endfor %}
    """


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

        # Combine default and user-provided tools
        user_tools = tools or []
        all_tools = DEFAULT_TOOLS + user_tools

        self.tools = [f.json_schema for f in all_tools]
        self.name_to_tool_map = {f.__name__: f for f in all_tools}
        self.chat_memory = chat_memory or ChatMemory()

    def __call__(
        self,
        *messages: Union[str, BaseMessage, list[Union[str, BaseMessage]], Callable],
        execution_history: Optional[List[Dict]] = None,
    ):
        """Process messages and return tool calls.

        :param messages: One or more messages to process. Can be strings, BaseMessage objects, or callable functions.
        :param execution_history: Optional list of previously executed tool calls for context
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

        # Build message list: system prompt, chat memory, execution history, then user messages
        message_list = [self.system_prompt]
        if self.chat_memory and user_messages:
            # Use the first message content for chat memory retrieval
            message_list.extend(self.chat_memory.retrieve(user_messages[0].content))

        # Add execution history context if provided
        if execution_history:
            history_context = "Previously called tools:\n"
            for call in execution_history[-5:]:  # Show last 5 tool calls
                tool_name = call.get("tool_name", "unknown")
                args = call.get("args", {})
                result = call.get("result", "")
                was_cached = call.get("was_cached", False)
                cache_indicator = " (cached)" if was_cached else ""
                history_context += (
                    f"- {tool_name}({args}) -> {result}{cache_indicator}\n"
                )

            from llamabot.components.messages import SystemMessage

            history_message = SystemMessage(content=history_context)
            message_list.append(history_message)

        message_list.extend(user_messages)

        # Execute the plan
        stream = self.stream_target != "none"
        logger.debug("Message list: {}", message_list)
        response = make_response(self, message_list, stream=stream)
        response = stream_chunks(response, target=self.stream_target)
        logger.debug("Response: {}", response)
        tool_calls = extract_tool_calls(response)

        if user_messages:
            self.chat_memory.append(user_messages[0])
            self.chat_memory.append(AIMessage(content=str(tool_calls)))

        return tool_calls
