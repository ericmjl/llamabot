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
    make_response,
    stream_chunks,
)
from llamabot.components.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ObservationMessage,
    ThoughtMessage,
)
from llamabot.components.docstore import AbstractDocumentStore
from llamabot.components.chat_memory import ChatMemory
from llamabot.components.tools import respond_to_user, today_date
from llamabot.config import default_language_model
from llamabot.experiments import metric
from llamabot.prompt_manager import prompt
from llamabot.recorder import sqlite_log
from llamabot.bot.toolbot import ToolBot, toolbot_sysprompt


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
        toolbot: Optional[ToolBot] = None,
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

        # Set tool_choice to "none" for the THOUGHT PHASE
        # The AgentBot should generate reasoning text, not call tools directly
        # Tools are called by the ToolBot in the ACTION PHASE
        self.tool_choice = "none"

        all_tools = [today_date, respond_to_user]
        if tools is not None:
            all_tools.extend([f for f in tools])
        self.tools = [f.json_schema for f in all_tools]
        self.name_to_tool_map = {f.__name__: f for f in all_tools}

        # Initialize ToolBot for tool selection
        if toolbot is None:
            self.toolbot = ToolBot(
                system_prompt=toolbot_sysprompt(globals_dict={}),
                model_name=model_name,
                tools=all_tools,
                **completion_kwargs,
            )
        else:
            self.toolbot = toolbot

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

        # ReAct iteration loop
        for iteration in range(max_iterations):
            self.run_meta["current_iteration"] = iteration + 1
            logger.debug(f"Starting ReAct cycle {iteration + 1} of {max_iterations}")

            # THOUGHT PHASE: Get reasoning from the agent
            stream = self.stream_target != "none"
            logger.debug("Message list for thought: {}", message_list)
            response = make_response(self, message_list, stream=stream)
            response = stream_chunks(response, target=self.stream_target)
            logger.debug("Thought response: {}", response)

            thought_content = extract_content(response)
            logger.debug("Thought content: {}", thought_content)

            # Add the thought to conversation
            thought_message = ThoughtMessage(content=thought_content)
            message_list.append(thought_message)
            self.run_meta["message_counts"]["assistant"] += 1

            # Append thought to memory immediately
            if self.memory:
                self.memory.append(thought_message)

            # ACTION PHASE: Use ToolBot to select and execute tools
            logger.debug("Using ToolBot for tool selection...")
            tool_calls = self.toolbot(
                *message_list, execution_history=self.execution_history
            )
            logger.debug("ToolBot selected: {}", tool_calls)

            # If no tools selected, automatically respond to user with the thought
            if not tool_calls:
                logger.debug(
                    "No tools selected, automatically responding to user with thought"
                )
                start_time = datetime.now()
                result = thought_content
                duration = (datetime.now() - start_time).total_seconds()

                # Record tool usage for respond_to_user
                tool_name = "respond_to_user"
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

                # Return final answer
                final_message = AIMessage(content=result)
                self.run_meta["end_time"] = datetime.now()
                self.run_meta["duration"] = (
                    self.run_meta["end_time"] - self.run_meta["start_time"]
                ).total_seconds()

                # Calculate cache hit rate
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

                # Append to memory if available
                if self.memory:
                    self.memory.append(final_message)

                sqlite_log(self, message_list + [final_message])
                return final_message

            if tool_calls:
                # Check for respond_to_user (final answer)
                respond_to_user_calls = [
                    call
                    for call in tool_calls
                    if call.function.name == "respond_to_user"
                ]
                if respond_to_user_calls:
                    logger.debug("Found respond_to_user, executing final answer")
                    start_time = datetime.now()
                    result = execute_tool_call(
                        respond_to_user_calls[0], self.name_to_tool_map
                    )
                    duration = (datetime.now() - start_time).total_seconds()

                    # Record tool usage
                    tool_name = respond_to_user_calls[0].function.name
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

                    # Return final answer
                    final_message = AIMessage(content=result)
                    self.run_meta["end_time"] = datetime.now()
                    self.run_meta["duration"] = (
                        self.run_meta["end_time"] - self.run_meta["start_time"]
                    ).total_seconds()

                    # Calculate cache hit rate
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

                    # Append to memory if available
                    if self.memory:
                        self.memory.append(final_message)

                    sqlite_log(self, message_list + [final_message])
                    return final_message

                # OBSERVATION PHASE: Execute tools and observe results
                logger.debug(
                    "Executing tools: {}", [call.function.name for call in tool_calls]
                )
                results = []
                session_tool_calls = []  # Track tools called in this session

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

                            # Record tool usage
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

                            # Track session tool calls for validation
                            session_tool_calls.append(
                                {
                                    "tool_name": tool_name,
                                    "args": json.loads(call.function.arguments),
                                    "result": result,
                                    "was_cached": was_cached,
                                }
                            )

                        except Exception as e:
                            duration = (datetime.now() - start_time).total_seconds()

                            # Record failed tool usage
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

                            result = f"Error: {str(e)}"

                        logger.debug("Tool result: {}", result)
                        results.append(result)

                # Add observation to conversation
                observation_content = (
                    f"Observation: {'; '.join(str(r) for r in results)}"
                )
                observation_message = ObservationMessage(content=observation_content)
                message_list.append(observation_message)
                self.run_meta["message_counts"]["tool"] += 1

                # Append observation to memory immediately
                if self.memory:
                    self.memory.append(observation_message)

                logger.debug("ReAct cycle completed. Results: {}", results)

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
