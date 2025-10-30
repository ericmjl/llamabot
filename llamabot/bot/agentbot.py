"""A module implementing an agent-based bot that can execute a sequence of tools.

This module provides the AgentBot class, which combines language model capabilities
with the ability to execute a sequence of tools based on user input.
The bot uses a decision-making system to determine which tools to call
and in what order, making it suitable for complex, multi-step tasks.
"""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

from loguru import logger

from llamabot.bot.simplebot import (
    SimpleBot,
    extract_content,
    extract_tool_calls,
)
from llamabot.components.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)
from llamabot.components.docstore import AbstractDocumentStore
from llamabot.components.chat_memory import ChatMemory
from llamabot.components.tools import respond_to_user, today_date
from llamabot.config import default_language_model
from llamabot.experiments import metric
from llamabot.prompt_manager import prompt
from llamabot.recorder import sqlite_log
from litellm import completion


def hash_result(result: Any) -> str:
    """Generate a SHA256 hash for a result value.

    :param result: Any result value that can be converted to string
    :return: First 8 characters of the SHA256 hash of the stringified result
    """
    # Convert result to a consistent string representation
    result_str = json.dumps(result, sort_keys=True, default=str)
    return hashlib.sha256(result_str.encode()).hexdigest()[:8]


@prompt("system")
def default_agentbot_system_prompt() -> str:
    """
    ## role

    You are a ReAct (Reasoning and Acting) agent that solves problems through explicit reasoning cycles.
    You must follow the Thought-Action-Observation pattern for every step of your reasoning.

    ## ReAct Pattern

    You must follow this exact cycle for each reasoning step:

    1. **Thought**: Analyze the current situation and plan your next action
    2. **Action**: Execute a tool or function call based on your reasoning
    3. **Observation**: Process the results and update your understanding

    This cycle repeats until you have enough information to provide a complete answer.

    ## Output Format

    You MUST format your responses as follows:

    **Thought**: [Your reasoning about what to do next]

    [Then either call a tool or provide your final answer]

    ## Tool Usage

    - Use available tools to gather information or perform actions
    - When you have enough information to answer the user's question, use the `respond_to_user` tool
    - Do not use `respond_to_user` until you are confident you have all necessary information
    - Always explain your reasoning in the "Thought" section before taking action

    ## Guidelines

    - Be explicit about your reasoning process
    - Use tools when you need external information or capabilities
    - Learn from observations to improve your next actions
    - Provide clear, helpful final answers
    - Never perform actions that could harm systems or violate policies

    ## Example

    User: "What's the weather like today?"

    Thought: I need to get current weather information. I should search for today's weather.

    [Tool call to search for weather]

    Observation: The search results show it's 72°F and sunny.

    Thought: I now have the weather information needed to answer the user's question.

    [Tool call to respond_to_user with the weather information]
    """


class AgentBot(SimpleBot):
    """An AgentBot that is capable of executing tools to solve a problem."""

    def __init__(
        self,
        temperature=0.0,
        system_prompt: str = default_agentbot_system_prompt(),
        model_name=default_language_model(),
        stream_target: str = "none",
        tools: Optional[list[Callable]] = None,
        memory: Optional[Union[ChatMemory, AbstractDocumentStore]] = None,
        **completion_kwargs,
    ):
        super().__init__(
            system_prompt=system_prompt,
            temperature=temperature,
            model_name=model_name,
            stream_target=stream_target,
            memory=memory,
            **completion_kwargs,
        )

        # AgentBot will use native tool_calls in a single ReAct loop
        self.tool_choice = "auto"

        all_tools = [today_date, respond_to_user]
        if tools is not None:
            all_tools.extend([f for f in tools])
        self.tools = [f.json_schema for f in all_tools]
        self.name_to_tool_map = {f.__name__: f for f in all_tools}

        # No ToolBot dependency; primary model selects tools

        # Initialize tool call caching system
        self.tool_call_cache: Dict[str, Any] = {}
        self.execution_history: List[Dict] = []

    def __call__(
        self,
        *messages: Union[str, BaseMessage, List[Union[str, BaseMessage]]],
        max_iterations: int = 10,
    ) -> AIMessage:
        """Process messages using the ReAct (Reasoning and Acting) pattern.

        :param messages: One or more messages to process
        :param max_iterations: Maximum number of ReAct cycles to run
        :return: The final response as an AIMessage
        :raises RuntimeError: If max iterations is reached
        """
        # Initialize run metadata
        self.run_meta = {
            "start_time": datetime.now(),
            "max_iterations": max_iterations,
            "current_iteration": 0,
            "tool_usage": {},
            "message_counts": {"user": 0, "assistant": 0, "tool": 0},
            "tool_calls_cached": 0,
            "tool_calls_executed": 0,
            "validation_failures": 0,
        }

        # Convert messages to a list of BaseMessage objects
        message_list = [self.system_prompt]
        for msg in messages:
            if isinstance(msg, str):
                message_list.append(HumanMessage(content=msg))
            elif isinstance(msg, list):
                for sub_msg in msg:
                    if isinstance(sub_msg, str):
                        message_list.append(HumanMessage(content=sub_msg))
                    else:
                        message_list.append(sub_msg)
            else:
                message_list.append(msg)

        # Count initial messages
        for msg in message_list:
            if isinstance(msg, HumanMessage):
                self.run_meta["message_counts"]["user"] += 1
            elif isinstance(msg, AIMessage):
                self.run_meta["message_counts"]["assistant"] += 1

        # Append user message to memory at start
        if self.memory:
            # Get the user message (should be last in initial message_list)
            user_messages = [
                msg for msg in message_list if isinstance(msg, HumanMessage)
            ]
            if user_messages:
                self.memory.append(user_messages[-1])

        # Retrieve from memory if available
        memory_messages = []
        if self.memory and message_list:
            # Get the user messages from the current call
            user_messages = [
                msg for msg in message_list if isinstance(msg, HumanMessage)
            ]
            if user_messages:
                memory_messages = self.memory.retrieve(
                    query=f"From our conversation history, give me the most relevant information to the query, {user_messages[-1].content}"
                )

        # Insert memory messages after system prompt
        if memory_messages:
            message_list = [message_list[0]] + memory_messages + message_list[1:]

        # ReAct iteration loop using native tool_calls
        # Maintain a separate raw messages list for provider function-calling protocol
        raw_messages: list[dict] = [
            {
                k: v
                for k, v in self.system_prompt.model_dump().items()
                if k in ("role", "content")
            }
        ]
        # Append initial user messages
        for msg in message_list[1:]:
            raw_messages.append(
                {k: v for k, v in msg.model_dump().items() if k in ("role", "content")}
            )

        for iteration in range(max_iterations):
            self.run_meta["current_iteration"] = iteration + 1
            logger.debug(f"Starting ReAct cycle {iteration + 1} of {max_iterations}")

            # Call model with tools, non-streaming, letting it decide tool_calls
            logger.debug("Message list for selection: {}", message_list)
            completion_kwargs = dict(
                model=self.model_name,
                messages=raw_messages,
                temperature=self.temperature,
                stream=False,
                tools=self.tools,
                tool_choice="auto",
            )
            completion_kwargs.update(self.completion_kwargs)
            response = completion(**completion_kwargs)
            logger.debug("Model response: {}", response)

            tool_calls = extract_tool_calls(response)
            content = extract_content(response)

            # Append assistant message to both tracked lists; include tool_calls in raw message
            message_list.append(AIMessage(content=content, tool_calls=tool_calls))
            assistant_msg = {
                "role": "assistant",
                "content": content if content else "",
            }
            if tool_calls:

                def _tc_dict(tc):
                    """Convert a tool call object to the provider tool_calls dict format."""
                    return {
                        "id": getattr(tc, "id", None),
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }

                assistant_msg["tool_calls"] = [_tc_dict(tc) for tc in tool_calls]
            raw_messages.append(assistant_msg)
            self.run_meta["message_counts"]["assistant"] += 1

            if tool_calls:
                logger.debug(
                    "Executing tools: {}", [call.function.name for call in tool_calls]
                )
                # If respond_to_user present, execute and finalize immediately
                respond_calls = [
                    c for c in tool_calls if c.function.name == "respond_to_user"
                ]
                if respond_calls:
                    start_time = datetime.now()
                    result = execute_tool_call(respond_calls[0], self.name_to_tool_map)
                    duration = (datetime.now() - start_time).total_seconds()
                    tool_name = respond_calls[0].function.name
                    if tool_name not in self.run_meta["tool_usage"]:
                        self.run_meta["tool_usage"][tool_name] = {
                            "calls": 0,
                            "success": 0,
                            "failures": 0,
                            "total_duration": 0.0,
                        }
                    self.run_meta["tool_usage"][tool_name]["calls"] += 1
                    self.run_meta["tool_usage"][tool_name]["success"] += 1
                    self.run_meta["tool_usage"][tool_name]["total_duration"] += duration

                    final_message = AIMessage(content=result)
                    self.run_meta["end_time"] = datetime.now()
                    self.run_meta["duration"] = (
                        self.run_meta["end_time"] - self.run_meta["start_time"]
                    ).total_seconds()

                    total_tool_calls = (
                        self.run_meta["tool_calls_cached"]
                        + self.run_meta["tool_calls_executed"]
                    )
                    if total_tool_calls > 0:
                        self.run_meta["cache_hit_rate"] = (
                            self.run_meta["tool_calls_cached"] / total_tool_calls
                        )
                    else:
                        self.run_meta["cache_hit_rate"] = 0.0

                    if self.memory:
                        self.memory.append(final_message)

                    sqlite_log(self, message_list + [final_message])
                    return final_message
                # Execute tool calls (in parallel)
                with ThreadPoolExecutor() as executor:
                    futures = {
                        executor.submit(self._execute_tool_with_cache, call): call
                        for call in tool_calls
                    }
                    for future in as_completed(futures):
                        call = futures[future]
                        start_time = datetime.now()
                        try:
                            result, was_cached = future.result()
                            duration = (datetime.now() - start_time).total_seconds()
                            tool_name = call.function.name
                            if tool_name not in self.run_meta["tool_usage"]:
                                self.run_meta["tool_usage"][tool_name] = {
                                    "calls": 0,
                                    "success": 0,
                                    "failures": 0,
                                    "total_duration": 0.0,
                                    "cached": 0,
                                }
                            self.run_meta["tool_usage"][tool_name]["calls"] += 1
                            if was_cached:
                                self.run_meta["tool_usage"][tool_name]["cached"] += 1
                                self.run_meta["tool_calls_cached"] += 1
                            else:
                                self.run_meta["tool_usage"][tool_name]["success"] += 1
                                self.run_meta["tool_calls_executed"] += 1
                            self.run_meta["tool_usage"][tool_name][
                                "total_duration"
                            ] += duration

                            # Append tool result message with tool_call_id for function-calling protocol
                            raw_messages.append(
                                {
                                    "role": "tool",
                                    "content": str(result),
                                    "tool_call_id": getattr(call, "id", None),
                                }
                            )
                            self.run_meta["message_counts"]["tool"] += 1

                        except Exception as e:
                            duration = (datetime.now() - start_time).total_seconds()
                            tool_name = call.function.name
                            if tool_name not in self.run_meta["tool_usage"]:
                                self.run_meta["tool_usage"][tool_name] = {
                                    "calls": 0,
                                    "success": 0,
                                    "failures": 0,
                                    "total_duration": 0.0,
                                    "cached": 0,
                                }
                            self.run_meta["tool_usage"][tool_name]["calls"] += 1
                            self.run_meta["tool_usage"][tool_name]["failures"] += 1
                            self.run_meta["tool_usage"][tool_name][
                                "total_duration"
                            ] += duration
                            raw_messages.append(
                                {
                                    "role": "tool",
                                    "content": f"Error: {str(e)}",
                                    "tool_call_id": getattr(call, "id", None),
                                }
                            )
                            self.run_meta["message_counts"]["tool"] += 1

                # Continue next cycle to let model incorporate observations
                continue

            # No tool calls; if we have content, finalize
            if content:
                final_message = AIMessage(content=content)
                self.run_meta["end_time"] = datetime.now()
                self.run_meta["duration"] = (
                    self.run_meta["end_time"] - self.run_meta["start_time"]
                ).total_seconds()

                total_tool_calls = (
                    self.run_meta["tool_calls_cached"]
                    + self.run_meta["tool_calls_executed"]
                )
                if total_tool_calls > 0:
                    self.run_meta["cache_hit_rate"] = (
                        self.run_meta["tool_calls_cached"] / total_tool_calls
                    )
                else:
                    self.run_meta["cache_hit_rate"] = 0.0

                if self.memory:
                    self.memory.append(final_message)

                sqlite_log(self, message_list + [final_message])
                return final_message

        # If we reach here, max iterations exceeded
        self.run_meta["end_time"] = datetime.now()
        self.run_meta["duration"] = (
            self.run_meta["end_time"] - self.run_meta["start_time"]
        ).total_seconds()

        total_tool_calls = (
            self.run_meta["tool_calls_cached"] + self.run_meta["tool_calls_executed"]
        )
        if total_tool_calls > 0:
            self.run_meta["cache_hit_rate"] = (
                self.run_meta["tool_calls_cached"] / total_tool_calls
            )
        else:
            self.run_meta["cache_hit_rate"] = 0.0

        raise RuntimeError(f"Agent exceeded maximum ReAct cycles ({max_iterations})")

        # If we reach here, max iterations exceeded
        self.run_meta["end_time"] = datetime.now()
        self.run_meta["duration"] = (
            self.run_meta["end_time"] - self.run_meta["start_time"]
        ).total_seconds()

        # Calculate cache hit rate
        total_tool_calls = (
            self.run_meta["tool_calls_cached"] + self.run_meta["tool_calls_executed"]
        )
        if total_tool_calls > 0:
            self.run_meta["cache_hit_rate"] = (
                self.run_meta["tool_calls_cached"] / total_tool_calls
            )
        else:
            self.run_meta["cache_hit_rate"] = 0.0

        raise RuntimeError(f"Agent exceeded maximum ReAct cycles ({max_iterations})")

    def _execute_tool_with_cache(self, tool_call) -> tuple[Any, bool]:
        """Execute a tool call with caching support.

        :param tool_call: The tool call to execute
        :return: (result, was_cached) tuple
        """
        tool_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        # Create cache key
        args_hash = hash_result(args)
        cache_key = f"{tool_name}:{args_hash}"

        # Check cache first
        if cache_key in self.tool_call_cache:
            logger.debug(f"Cache hit for {tool_name} with args {args_hash}")
            result = self.tool_call_cache[cache_key]

            # Add to execution history
            self.execution_history.append(
                {
                    "tool_name": tool_name,
                    "args": args,
                    "result": result,
                    "was_cached": True,
                    "timestamp": datetime.now(),
                }
            )

            return result, True

        # Not in cache, execute tool
        logger.debug(f"Cache miss for {tool_name}, executing tool")
        result = execute_tool_call(tool_call, self.name_to_tool_map)

        # Store in cache
        self.tool_call_cache[cache_key] = result

        # Add to execution history
        self.execution_history.append(
            {
                "tool_name": tool_name,
                "args": args,
                "result": result,
                "was_cached": False,
                "timestamp": datetime.now(),
            }
        )

        return result, False


def execute_tool_call(tool_call, name_to_tool_map: dict[str, Callable]) -> Any:
    """Execute a tool call.

    :param tool_call: The tool call to execute
    :return: The result of the tool call
    """
    func_name = tool_call.function.name
    func = name_to_tool_map.get(func_name)
    if not func:
        return f"Error: Function {func_name} not found"
    func_args = json.loads(tool_call.function.arguments)
    try:
        logger.debug(f"Executing tool call: {func_name} with arguments: {func_args}")
        return func(**func_args)
    except Exception as e:
        result = (
            f"Error executing tool call: {e}! "
            f"Arguments were: {func_args}. "
            "If you get a timeout error, try bumping up the timeout parameter."
        )
        return result


@metric
def tool_usage_count(agent: AgentBot) -> int:
    """Return the total number of tool calls made by the agent.

    :param agent: The AgentBot instance
    :return: Total number of tool calls
    """
    return sum(usage["calls"] for usage in agent.run_meta["tool_usage"].values())
