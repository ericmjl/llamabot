"""A module implementing an agent-based bot that can execute a sequence of tools.

This module provides the AgentBot class, which combines language model capabilities
with the ability to execute a sequence of tools based on user input.
The bot uses a decision-making system to determine which tools to call
and in what order, making it suitable for complex, multi-step tasks.
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field

from llamabot.bot.structuredbot import StructuredBot
from llamabot.components.messages import AIMessage, BaseMessage, system, user
from llamabot.components.sandbox import ScriptExecutor, ScriptMetadata
from llamabot.components.tools import tool
from llamabot.config import default_language_model


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


@tool
def write_and_execute_script(
    code: str,
    python_version: str = ">=3.11",
    dependencies: List[str] = Field(
        ...,
        description="List of pip-installable dependencies that go into the embedded script metadata.",
    ),
    timeout: int = 30,
) -> Dict[str, Any]:
    """Write and execute a Python script in a secure sandbox.

    :param code: The Python code to execute
    :param python_version: Python version requirement
    :param dependencies: List of pip dependencies
    :param timeout: Execution timeout in seconds
    :return: Script execution results
    """
    # Create metadata
    metadata = ScriptMetadata(
        requires_python=python_version,
        dependencies=dependencies or [],
        auth=str(uuid4()),  # Generate unique ID for this execution
        timestamp=datetime.now(),
    )

    # Initialize executor
    executor = ScriptExecutor()

    # Write and run script
    script_path = executor.write_script(code, metadata)
    return executor.run_script(script_path, timeout)


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
    :return: First 8 characters of the SHA256 hash of the stringified result
    """
    # Convert result to a consistent string representation
    result_str = json.dumps(result, sort_keys=True, default=str)
    return hashlib.sha256(result_str.encode()).hexdigest()[:8]


class ToolArguments(BaseModel):
    """Pydantic model for tool arguments.

    :param name: Name of the argument
    :param value: Value of the argument, can be None if coming from cache
    """

    name: str = Field(..., description="Name of the argument.")
    value: Union[int, str, float, bool, None] = Field(
        description="Argument value. None if coming from cache."
    )


class CachedArguments(BaseModel):
    """Pydantic model for cached argument references.

    :param arg_name: Name of the argument to populate from cache
    :param hash_key: Hash key of the cached result to use
    """

    arg_name: str = Field(
        ..., description="Name of the argument to populate from cache."
    )
    hash_key: str = Field(
        ...,
        description="Hash key of the cached result to use. It is a sha256 hash of the result.",
    )


class ToolToCall(BaseModel):
    """Pydantic model representing a single tool to be called by the agent.

    :param tool_name: Name of the tool to call
    :param tool_args: List of tool arguments and their values
    :param use_cached_results: List of argument names and hash keys
        that should be used as input for those arguments
    """

    tool_name: str = Field(..., description="The tool to call.")
    tool_args: list[ToolArguments] = Field(
        ...,
        description="Arguments to pass to the tool. Use None for arguments that should come from cache.",
    )
    use_cached_results: list[CachedArguments] = Field(
        ...,
        description="List of argument names and result hash keys to use as input.",
    )


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

        functions = [agent_finish, return_error, write_and_execute_script] + (
            functions or []
        )
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

        # Skip caching for write_and_execute_script results
        if tool_name == "write_and_execute_script":
            return result_hash

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
                    *[json.dumps(func.json_schema) for func in self.functions],
                    "Here are the cached results you can use: ",
                    *[
                        f"{k}: {v.tool_name} result from {v.timestamp}"
                        for k, v in self.memory.items()
                    ],
                    "Here is the execution history: ",
                    *execution_history,
                )
            )

            # Prepare arguments from tool_args
            args = {}
            for arg in next_tool.tool_args:
                if arg.value is not None:
                    args[arg.name] = arg.value

            try:
                # Add cached results where specified
                for cached_arg in next_tool.use_cached_results:
                    if cached_arg.hash_key not in self.memory:
                        raise ValueError(
                            f"Cached result {cached_arg.hash_key} not found"
                        )
                    args[cached_arg.arg_name] = self.memory[cached_arg.hash_key].result

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
