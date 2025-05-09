"""A module implementing an agent-based bot that can execute a sequence of tools.

This module provides the AgentBot class, which combines language model capabilities
with the ability to execute a sequence of tools based on user input.
The bot uses a decision-making system to determine which tools to call
and in what order, making it suitable for complex, multi-step tasks.
"""

import hashlib
import json
from typing import Any, Callable, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from llamabot.bot.simplebot import (
    SimpleBot,
    _extract_content,
    _extract_tool_calls,
    _make_response,
    _stream_chunks,
)
from llamabot.components.messages import (
    AIMessage,
    BaseMessage,
    user,
    to_basemessage,
)
from llamabot.config import default_language_model


def hash_result(result: Any) -> str:
    """Generate a SHA256 hash for a result value.

    :param result: Any result value that can be converted to string
    :return: First 8 characters of the SHA256 hash of the stringified result
    """
    # Convert result to a consistent string representation
    result_str = json.dumps(result, sort_keys=True, default=str)
    return hashlib.sha256(result_str.encode()).hexdigest()[:8]


class AgentBot(SimpleBot):
    """An AgentBot that is capable of executing tools to solve a problem."""

    def __init__(
        self,
        system_prompt: str,
        temperature=0.0,
        model_name=default_language_model(),
        stream_target: str = "stdout",
        api_key: Optional[str] = None,
        mock_response: Optional[str] = None,
        tools: Optional[list[Callable]] = None,
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
        if tools is None:
            self.tools = []
            self.name_to_tool_map = {}
        else:
            self.tools = [
                {"type": "function", "function": f.json_schema} for f in tools
            ]
            self.name_to_tool_map = {f.__name__: f for f in tools}

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
        iteration = 0

        messages = [self.system_prompt] + list([user(m) for m in messages])

        while iteration < max_iterations:
            iteration += 1
            processed_messages = to_basemessage(messages)
            print(processed_messages)
            response = _make_response(self, processed_messages)
            response = _stream_chunks(response, target=self.stream_target)
            tool_calls = _extract_tool_calls(response)
            content = _extract_content(response)
            response_message = AIMessage(content=content, tool_calls=tool_calls)

            if tool_calls:
                # Create a function to execute a tool call
                def execute_tool_call(tool_call):
                    """Execute a tool call.

                    :param tool_call: The tool call to execute
                    :return: The result of the tool call
                    """
                    func_name = tool_call.function.name
                    func = self.name_to_tool_map.get(func_name)
                    if not func:
                        return f"Error: Function {func_name} not found"
                    func_args = json.loads(tool_call.function.arguments)
                    return func(**func_args)

                # Print the names of the functions that are to be called
                print("Calling functions:")
                for call in tool_calls:
                    print(f"Calling: {call.function.name}(**{call.function.arguments})")

                # Execute all tool calls in parallel using threads
                with ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(execute_tool_call, call) for call in tool_calls
                    ]
                    results = [future.result() for future in as_completed(futures)]
                messages.append((str(results)))
            else:
                return response_message.content

        raise RuntimeError(f"Agent exceeded maximum iterations ({max_iterations})")
