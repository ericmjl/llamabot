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
from llamabot.components.tools import tool


@tool
def agent_finish(message: Any) -> str:
    """Tool to indicate that the agent has finished processing the user's message."""

    try:
        return str(message)
    except Exception:
        return repr(message)


class ToolToCall(BaseModel):
    """Pydantic model representing a single tool to be called by the agent.

    :param tool_name: Name of the tool to call.
    :param tool_arguments: Dictionary mapping argument names to their values.
        Arguments that are not known ahead of time should be None.
    """

    tool_name: str = Field(..., description="The tool to call.")
    tool_arguments: Dict[str, Any | None] = Field(
        ...,
        description="Arguments to pass to the tool. Use None for arguments that are not known ahead of time.",
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
            pydantic_model=ToolToCall,
            system_prompt=system(
                "Given the following message and available tools, "
                "pick the next tool that you need to call "
                "and the arguments that you need for it. "
                "Any arguments that you do not know ahead of time, just leave blank. "
                "Only call tools that are relevant to the user's message. "
                "If the task is complete, use the agent_finish tool."
            ),
            model_name="gpt-4o",
        )

        functions = [agent_finish] + (functions or [])

        self.functions = functions
        self.tools = {func.__name__: func for func in self.functions}

    def __call__(
        self,
        *messages: Union[str, BaseMessage, List[Union[str, BaseMessage]]],
        max_iterations: int = 10,
    ) -> str:
        """Process messages and execute the appropriate sequence of tools.

        This method takes one or more messages, determines which tools to call using
        the decision bot, and executes them in sequence. The output of each tool
        can be used as input for subsequent tools if needed.

        :param messages: One or more messages to process. Can be strings, BaseMessages,
            or lists of either
        :param max_iterations: Maximum number of iterations to run before raising an error
        :return: The result of the final tool execution in the sequence
        :raises RuntimeError: If max iterations is reached without completion
        """
        results = []  # Keep track of all results
        execution_history = []  # Keep track of function calls and results
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            next_tool = self.decision_bot(
                user(
                    *messages,
                    *[func.json_schema for func in self.functions],
                    *execution_history,  # Include full execution history
                )
            )

            args = next_tool.tool_arguments.copy()

            # If we have previous results, look for any None arguments to fill
            if results:
                for arg_name, arg_value in args.items():
                    if arg_value is None:
                        args[arg_name] = results[-1]  # Use most recent result

            result = self.tools[next_tool.tool_name](**args)
            print(f"{next_tool.tool_name}: {result}")

            # Store both result and execution history
            results.append(result)
            execution_history.append(
                f"Called {next_tool.tool_name}({args}) -> {result}"
            )

            if next_tool.tool_name == "agent_finish":
                return result

        raise RuntimeError(f"Agent exceeded maximum iterations ({max_iterations})")
