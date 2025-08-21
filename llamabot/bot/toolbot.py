"""ToolBot - A single-turn bot that can execute tools."""

from typing import Callable, List, Optional
from loguru import logger

from llamabot.components.tools import today_date, respond_to_user
from llamabot.components.chat_memory import ChatMemory
from llamabot.components.messages import AIMessage
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

    def __call__(self, message):
        """Process a message and return tool calls.

        :param message: The message to process
        :return: List of tool calls to execute
        """
        from llamabot.components.messages import user

        message = user(message)
        # Convert messages to a list of UserMessage objects
        message_list = [self.system_prompt]
        if self.chat_memory:
            message_list.extend(self.chat_memory.retrieve(message.content))
        message_list.extend([message])

        # Execute the plan
        stream = self.stream_target != "none"
        logger.debug("Message list: {}", message_list)
        response = make_response(self, message_list, stream=stream)
        response = stream_chunks(response, target=self.stream_target)
        logger.debug("Response: {}", response)
        tool_calls = extract_tool_calls(response)

        self.chat_memory.append(message, AIMessage(content=str(tool_calls)))

        return tool_calls
