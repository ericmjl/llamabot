"""A module implementing an agent-based bot that can execute a sequence of tools.

This module provides the AgentBot class, which combines language model capabilities
with the ability to execute a sequence of tools based on user input.
The bot uses a decision-making system to determine which tools to call
and in what order, making it suitable for complex, multi-step tasks.
"""

from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime
import hashlib
import json

from pydantic import BaseModel, Field

from llamabot import system, user
from llamabot.bot.structuredbot import StructuredBot
from llamabot.components.messages import AIMessage, BaseMessage
from llamabot.config import default_language_model
from llamabot.components.tools import tool


@tool
def agent_finish(message: Any) -> str:
    """Tool to indicate that the agent has finished processing the user's message."""

    try:
        return str(message)
    except Exception:
        return repr(message)


@tool
def return_error(message: Any) -> str:
    """Tool to indicate that the agent has encountered an error.

    :param message: The error message or exception to raise
    :raises Exception: Always raises the provided error message as an exception
    """
    raise Exception(str(message))


class CachedResult(BaseModel):
    """Model for storing cached results from tool executions.

    :param tool_name: Name of the tool that generated the result
    :param tool_arguments: Dictionary of arguments used to call the tool
    :param result: The actual result value
    :param timestamp: When the result was generated
    :param hash_key: SHA256 hash of the stringified result
    """

    tool_name: str
    tool_arguments: Dict[str, Any]
    result: Any
    timestamp: datetime = Field(default_factory=datetime.now)
    hash_key: str


def hash_result(result: Any) -> str:
    """Generate a SHA256 hash for a result value.

    :param result: Any result value that can be converted to string
    :return: SHA256 hash of the stringified result
    """
    # Convert result to a consistent string representation
    result_str = json.dumps(result, sort_keys=True, default=str)
    return hashlib.sha256(result_str.encode()).hexdigest()


class ToolToCall(BaseModel):
    """Pydantic model representing a single tool to be called by the agent.

    :param tool_name: Name of the tool to call
    :param tool_arguments: Dictionary mapping argument names to their values
    :param use_cached_results: Dictionary mapping argument names to hash keys
        that should be used as input for those arguments
    """

    tool_name: str = Field(..., description="The tool to call.")
    tool_arguments: Dict[str, Any | None] = Field(
        ...,
        description="Arguments to pass to the tool. Use None for arguments that should come from cache.",
    )
    use_cached_results: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of argument names to result hash keys to use as input.",
    )


# NOTE: This is a bit of a breaking pattern by not inheriting from any classes.
# Should probably inherit from StructuredBot.
class AgentBot:
    """A bot that uses an agent to process messages and execute tools.

    Additional attributes:
    :param memory: Dictionary storing cached results from tool executions,
                  indexed by SHA256 hash of the result
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
        self.memory: Dict[str, CachedResult] = {}

        # Update decision bot prompt to include information about using cached results
        self.decision_bot = StructuredBot(
            pydantic_model=ToolToCall,
            system_prompt=system(
                system_prompt,
                "Given the following message and available tools, "
                "pick the next tool that you need to call "
                "and the arguments that you need for it. "
                "You have access to previously cached results - use them when appropriate "
                "by specifying their hash key in use_cached_results. "
                "Any arguments that you do not know ahead of time, just leave blank. "
                "Only call tools that are relevant to the user's message. "
                "If the task is complete, use the agent_finish tool.",
            ),
            model_name=model_name,
            temperature=temperature,
            stream_target=stream_target,
            api_key=api_key,
            mock_response=mock_response,
            **completion_kwargs,
        )

        functions = [agent_finish, return_error] + (functions or [])
        self.functions = functions
        self.tools = {func.__name__: func for func in self.functions}

    def _store_result(
        self, tool_name: str, result: Any, tool_arguments: Dict[str, Any]
    ) -> str:
        """Store a tool execution result in memory.

        :param tool_name: Name of the tool that generated the result
        :param tool_arguments: Dictionary of arguments used to call the tool
        :param result: The result to store
        :return: The hash key under which the result is stored
        """
        result_hash = hash_result(result)

        # If this exact result is already cached, return its hash
        if result_hash in self.memory:
            return result_hash

        # Store new result indexed by its hash
        self.memory[result_hash] = CachedResult(
            tool_name=tool_name,
            tool_arguments=tool_arguments,
            result=result,
            hash_key=result_hash,
        )
        return result_hash

    def __call__(
        self,
        *messages: Union[str, BaseMessage, List[Union[str, BaseMessage]]],
        max_iterations: int = 10,
    ) -> AIMessage:
        """Process messages and execute the appropriate sequence of tools.

        :param messages: One or more messages to process
        :param max_iterations: Maximum number of iterations to run
        :return: The final response as an AIMessage
        :raises RuntimeError: If max iterations is reached
        """
        results = []
        execution_history = []
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            next_tool = self.decision_bot(
                user(
                    "Here are the previous messages and the execution history. "
                    "Use this to decide which tool to call next or raise an error if you need to stop. ",
                    *messages,
                    "Here are the available tools: ",
                    *[func.json_schema for func in self.functions],
                    "Here are the cached results you can use: ",
                    *[
                        f"{k}: {v.tool_name} result from {v.timestamp}"
                        for k, v in self.memory.items()
                    ],
                    "Here is the execution history: ",
                    *execution_history,
                )
            )

            # Prepare arguments, substituting cached results where specified
            args = next_tool.tool_arguments.copy()
            for arg_name, result_id in next_tool.use_cached_results.items():
                if result_id not in self.memory:
                    raise ValueError(f"Cached result {result_id} not found")
                args[arg_name] = self.memory[result_id].result

            try:
                tool_name = next_tool.tool_name
                result = self.tools[tool_name](**args)
                result_id = self._store_result(tool_name, result, args)

                # Log result - only show tool name, args, and result ID for conciseness
                result_str = f"{tool_name}({args}) -> {result_id}"
                print(result_str)
                results.append(result)
                execution_history.append(result_str)

                # Handle special tool cases
                if tool_name == "agent_finish":
                    return AIMessage(content=str(result))
                if tool_name == "return_error":
                    raise Exception(result)

            except Exception as e:
                error_msg = f"Error calling {next_tool.tool_name}: {str(e)}"
                print(error_msg)
                execution_history.append(error_msg)
                if next_tool.tool_name == "return_error":
                    raise e
                continue

        raise RuntimeError(f"Agent exceeded maximum iterations ({max_iterations})")
