"""A module implementing an agent-based bot that can execute a sequence of tools.

This module provides the AgentBot class, which combines language model capabilities
with the ability to execute a sequence of tools based on user input.
The bot uses a decision-making system to determine which tools to call
and in what order, making it suitable for complex, multi-step tasks.
"""

import asyncio
import hashlib
import json

# Note: ThreadPoolExecutor and as_completed are no longer used with MCP approach
from datetime import datetime
from typing import Any, Callable, List, Optional, Union

from fastmcp import Client
from loguru import logger

from llamabot.bot.simplebot import (
    SimpleBot,
    extract_content,
    make_response,
    stream_chunks,
)
from llamabot.components.local_mcp_server import LocalMCPServer
from llamabot.components.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)
from llamabot.components.tools import respond_to_user, today_date
from llamabot.config import default_language_model
from llamabot.experiments import metric
from llamabot.prompt_manager import prompt
from llamabot.recorder import sqlite_log
from llamabot.bot.toolbot import ToolBot, toolbot_sysprompt


def run_async_in_sync(coro):
    """Run async coroutine in a sync context, handling both regular and notebook environments."""
    try:
        # Try to get the current event loop
        loop = asyncio.get_running_loop()
        # If we're in a running loop (like Jupyter), use nest_asyncio
        import nest_asyncio

        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop running, safe to use asyncio.run()
        return asyncio.run(coro)


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
        mcp_servers: Optional[List[str]] = None,
        toolbot: Optional[ToolBot] = None,
        **completion_kwargs,
    ):
        super().__init__(
            system_prompt=system_prompt,
            temperature=temperature,
            model_name=model_name,
            stream_target=stream_target,
            **completion_kwargs,
        )

        # Create local MCP server for local tools
        local_server = LocalMCPServer("local")
        all_tools = [today_date, respond_to_user]
        if tools is not None:
            all_tools.extend([f for f in tools])
        local_server.register_tools(all_tools)

        # Create MCP clients
        mcp_clients = []

        # Add local client
        local_client = Client(local_server.get_server())
        mcp_clients.append(local_client)

        # Add remote clients
        if mcp_servers:
            for server_url in mcp_servers:
                remote_client = Client(server_url)
                mcp_clients.append(remote_client)

        # Keep legacy tool schemas for backward compatibility
        self.tools = [f.json_schema for f in all_tools]
        self.name_to_tool_map = {f.__name__: f for f in all_tools}

        # Initialize ToolBot with MCP clients
        if toolbot is None:
            self.toolbot = ToolBot(
                system_prompt=toolbot_sysprompt(globals_dict={}),
                model_name=model_name,
                mcp_clients=mcp_clients,
                **completion_kwargs,
            )
        else:
            self.toolbot = toolbot

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
            thought_message = AIMessage(content=thought_content)
            message_list.append(thought_message)
            self.run_meta["message_counts"]["assistant"] += 1

            # ACTION PHASE: Use ToolBot to select and execute tools
            logger.debug("Using ToolBot for tool selection...")
            tool_calls = self.toolbot(*message_list)
            logger.debug("ToolBot selected: {}", tool_calls)

            # Check if agent provided a final answer without using tools
            if not tool_calls and thought_content and len(thought_content.strip()) > 0:
                # Check if the thought content looks like a final answer
                if any(
                    phrase in thought_content.lower()
                    for phrase in [
                        "equals",
                        "is",
                        "the answer is",
                        "the result is",
                        "final answer",
                        "answer:",
                        "result:",
                    ]
                ):
                    logger.debug("Agent provided final answer without tools")
                    final_message = AIMessage(content=thought_content)
                    self.run_meta["end_time"] = datetime.now()
                    self.run_meta["duration"] = (
                        self.run_meta["end_time"] - self.run_meta["start_time"]
                    ).total_seconds()
                    sqlite_log(self, message_list)
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
                    result = run_async_in_sync(
                        self.toolbot.execute_tool_call(respond_to_user_calls[0])
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
                    sqlite_log(self, message_list + [final_message])
                    return final_message

                # OBSERVATION PHASE: Execute tools via MCP and observe results
                logger.debug(
                    "Executing tools: {}", [call.function.name for call in tool_calls]
                )
                results = []

                # Execute tools via ToolBot's MCP execution
                for call in tool_calls:
                    start_time = datetime.now()
                    try:
                        result = run_async_in_sync(self.toolbot.execute_tool_call(call))
                        duration = (datetime.now() - start_time).total_seconds()

                        # Record successful tool usage
                        tool_name = call.function.name
                        if tool_name not in self.run_meta["tool_usage"]:
                            self.run_meta["tool_usage"][tool_name] = {
                                "calls": 0,
                                "success": 0,
                                "failures": 0,
                                "total_duration": 0.0,
                            }
                        self.run_meta["tool_usage"][tool_name]["calls"] += 1
                        self.run_meta["tool_usage"][tool_name]["success"] += 1
                        self.run_meta["tool_usage"][tool_name][
                            "total_duration"
                        ] += duration

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
                observation_message = HumanMessage(content=observation_content)
                message_list.append(observation_message)
                self.run_meta["message_counts"]["tool"] += 1

                logger.debug("ReAct cycle completed. Results: {}", results)

        # If we reach here, max iterations exceeded
        self.run_meta["end_time"] = datetime.now()
        self.run_meta["duration"] = (
            self.run_meta["end_time"] - self.run_meta["start_time"]
        ).total_seconds()
        raise RuntimeError(f"Agent exceeded maximum ReAct cycles ({max_iterations})")


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
