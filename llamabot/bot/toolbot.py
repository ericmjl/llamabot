"""ToolBot - A single-turn bot that can execute tools."""

from typing import Callable, List, Optional, Union
from loguru import logger

from llamabot.components.tools import today_date, respond_to_user
from llamabot.components.messages import BaseMessage
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
    2. **Review conversation history** to see if tools have already been executed
    3. **Select the most appropriate tool** from your available function toolkit
    4. **Extract or infer the necessary arguments** for the selected function
    5. **Return a single function call** with the proper arguments to execute

    ## Available Tools:
    You have access to tools through function calling. Each tool has detailed documentation in its docstring that explains:
    - When to use the tool
    - What parameters it expects
    - What it returns
    - Usage examples and guidelines

    ## Decision Process:
    When you receive a user request:
    1. **FIRST: Review the conversation history** - Look for ObservationMessages that show previous tool executions and their results
    2. **CRITICAL: Check if the task is already complete** - If a tool has already been called successfully and provided the needed information, use `respond_to_user` instead of calling the same tool again
    3. **Break down what the user is asking for** - Understand the core request
    4. **Identify the core action or information needed** - What specific tool or response is required
    5. **Map this to one of your available tools** by reading their docstrings
    6. **Determine the required parameters/arguments** for the selected tool
    7. **Make the function call** with appropriate arguments

    **CRITICAL CHECK**: Before calling any tool, scan the conversation for ObservationMessages. If you see an ObservationMessage with a successful result from the same tool, use `respond_to_user` instead of calling the tool again.

    **IMPORTANT**: Before calling any tool, ask yourself: "Has this exact tool already been called successfully in this conversation?" If yes, use `respond_to_user` instead.

    ## Important Guidelines:
    - **CRITICAL: Avoid redundant tool calls**: If you can see from ObservationMessages that a tool has already been executed successfully, DO NOT call it again unless the user is asking for something completely different
    - **Use respond_to_user when appropriate**: If the user's request has been fulfilled by previous tool executions, use `respond_to_user` to provide a helpful response based on the available information
    - **Consider the conversation context**: Always review what has already been done before deciding on the next action

    ## Examples of Good Decision Making:

    **Example 1 - DO NOT repeat tool calls:**
    - User asks: "Extract statistical info from this text"
    - ToolBot calls: `extract_statistical_info` → gets successful result
    - User asks again: "Can you check if my experimental design makes sense?"
    - **CORRECT**: Use `respond_to_user` with the existing extracted information
    - **WRONG**: Call `extract_statistical_info` again

    **Example 2 - When to repeat tool calls:**
    - User asks: "Extract statistical info from this text"
    - ToolBot calls: `extract_statistical_info` → gets successful result
    - User asks: "Now extract info from this DIFFERENT text"
    - **CORRECT**: Call `extract_statistical_info` with the new text

    Remember: You are a function selector and executor. Read the tool docstrings carefully to understand when and how to use each tool effectively. Always consider the full conversation context, including previous tool executions and their results.

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
    """A single-turn bot that can execute tools.

    This bot is designed to analyze user requests and determine the most appropriate
    tool or function to execute. It's a generalization of other bot types, focusing
    on tool selection and execution rather than multi-turn conversation.

    :param system_prompt: The system prompt to use
    :param model_name: The name of the model to use
    :param tools: Optional list of additional tools to include
    :param completion_kwargs: Additional keyword arguments for completion
    """

    def __init__(
        self,
        system_prompt: str,
        model_name: str,
        tools: Optional[List[Callable]] = None,
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

    def __call__(
        self,
        *messages: Union[str, BaseMessage, list[Union[str, BaseMessage]], Callable],
    ):
        """Process messages and return tool calls.

        :param messages: One or more messages to process. Can be strings, BaseMessage objects, or callable functions.
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

        # Always use the full conversation context with our system prompt prepended
        message_list = [self.system_prompt] + user_messages

        # Execute the plan
        stream = self.stream_target != "none"
        logger.debug("Message list: {}", message_list)

        # Detailed logging of what ToolBot receives
        logger.debug("=== TOOLBOT INPUT DEBUG ===")
        logger.debug("Number of messages received: {}", len(processed_messages))
        logger.debug(
            "Processed messages types: {}",
            [type(msg).__name__ for msg in processed_messages],
        )
        logger.debug("User messages count: {}", len(user_messages))
        logger.debug("Final message_list length: {}", len(message_list))
        logger.debug(
            "Message list roles: {}",
            [getattr(msg, "role", "no-role") for msg in message_list],
        )

        # Log each message in detail
        for i, msg in enumerate(message_list):
            if hasattr(msg, "role"):
                logger.debug(
                    "Message {}: role='{}', content='{}'",
                    i,
                    msg.role,
                    (
                        str(msg.content)[:100] + "..."
                        if len(str(msg.content)) > 100
                        else str(msg.content)
                    ),
                )
            else:
                logger.debug(
                    "Message {}: type='{}', content='{}'",
                    i,
                    type(msg).__name__,
                    (
                        str(msg.content)[:100] + "..."
                        if len(str(msg.content)) > 100
                        else str(msg.content)
                    ),
                )

        logger.debug("=== END TOOLBOT INPUT DEBUG ===")
        response = make_response(self, message_list, stream=stream)
        response = stream_chunks(response, target=self.stream_target)
        logger.debug("Response: {}", response)
        tool_calls = extract_tool_calls(response)

        return tool_calls
