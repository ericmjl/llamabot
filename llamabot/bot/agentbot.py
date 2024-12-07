"""A module implementing an agent-based bot that can execute a sequence of tools.

This module provides the AgentBot class, which combines language model capabilities
with the ability to execute a sequence of tools based on user input.
The bot uses a decision-making system to determine which tools to call
and in what order, making it suitable for complex, multi-step tasks.
"""

from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from llamabot import system, user
from llamabot.bot.simplebot import SimpleBot
from llamabot.bot.structuredbot import StructuredBot
from llamabot.components.messages import BaseMessage
from llamabot.config import default_language_model


class ToolsToCall(BaseModel):
    """Pydantic model representing a sequence of tools to be called by the agent.

    :param tool_names: List of tool names in the order they should be called.
    :param tool_arguments: Dictionary mapping tool names to their arguments.
        Arguments that are not known ahead of time should be None.
    """

    tool_names: List[str] = Field(..., description="The order in which to call tools.")
    tool_arguments: Dict[str, Any | None] = Field(
        ...,
        description="Arguments to pass to each tool. Use None for arguments that are not known ahead of time.",
    )


class AgentBot(SimpleBot):
    """A bot that uses an agent to process messages and execute tools.

    This bot extends SimpleBot by adding the ability to execute a sequence of tools
    based on user input. It uses a decision-making system (implemented as a StructuredBot)
    to determine which tools to call and in what order.

    Example:
        >>> def add_numbers(a: int, b: int) -> int:
        ...     '''Add two numbers together.'''
        ...     return a + b
        >>> def multiply_by_two(x: int) -> int:
        ...     '''Multiply a number by 2.'''
        ...     return x * 2
        >>> agent = AgentBot(
        ...     system_prompt="Do math calculations",
        ...     functions=[add_numbers, multiply_by_two]
        ... )
        >>> # Agent can now chain add_numbers() result into multiply_by_two()

    :param system_prompt: The system prompt that guides the bot's behavior
    :param temperature: Controls randomness in the model's output. Lower values make the output more deterministic
    :param model_name: Name of the language model to use
    :param stream_target: Where to stream the output ("stdout" by default)
    :param api_key: Optional API key for the language model service
    :param mock_response: Optional mock response for testing
    :param functions: List of callable tools that the agent can use. The tools can be chained together,
        with output from one tool being used as input for another tool. This allows for complex
        multi-step computations where intermediate results feed into subsequent tool calls.
        Each function should have type hints and docstrings to help the agent understand
        how to use them correctly. These functions should also be decorated with `@tool`
        to automatically expose their function signature to the agent.
    :param completion_kwargs: Additional keyword arguments for the completion API
    """

    def __init__(
        self,
        system_prompt: str,
        temperature=0.0,
        model_name=default_language_model(),
        stream_target: str = "stdout",
        api_key: Optional[str] = None,
        mock_response: Optional[str] = None,
        functions: Optional[list[Callable]] = None,
        **completion_kwargs,
    ):
        super().__init__(
            system_prompt=system_prompt,
            temperature=temperature,
            model_name=model_name,
            stream_target=stream_target,
            api_key=api_key,
            mock_response=mock_response,
            **completion_kwargs,
        )

        self.decision_bot = StructuredBot(
            pydantic_model=ToolsToCall,
            system_prompt=system(
                "Given the following message and available tools, "
                "pick the tool(s) that you need to call on "
                "and the arguments that you need for those. "
                "Any arguments that you do not know ahead of time, just leave blank. "
                "Not all tools need to be called."
            ),
            model_name="gpt-4o-mini",
        )

        self.functions = functions
        self.tools = {func.__name__: func for func in self.functions}

    def __call__(
        self, *messages: Union[str, BaseMessage, List[Union[str, BaseMessage]]]
    ) -> str:
        """Process messages and execute the appropriate sequence of tools.

        This method takes one or more messages, determines which tools to call using
        the decision bot, and executes them in sequence. The output of each tool
        can be used as input for subsequent tools if needed.

        :param messages: One or more messages to process. Can be strings, BaseMessages,
            or lists of either
        :return: The result of the final tool execution in the sequence
        """
        functions_to_call = self.decision_bot(
            user(
                *messages,
                *[func.json_schema for func in self.functions],
            )
        )

        result = None
        for function in functions_to_call.tool_names:
            args = functions_to_call.tool_arguments[function].copy()

            # If this isn't the first function and we have a previous result,
            # look for any None/empty arguments to fill with the previous result
            if result is not None:
                for arg_name, arg_value in args.items():
                    if arg_value is None:
                        args[arg_name] = result

            result = self.tools[function](**args)
            print(f"{function}: {result}")

        return result
