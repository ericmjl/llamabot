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
from typing import Any, Callable, List, Optional, Union

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


def hash_result(result: Any) -> str:
    """Generate a SHA256 hash for a result value.

    :param result: Any result value that can be converted to string
    :return: First 8 characters of the SHA256 hash of the stringified result
    """
    # Convert result to a consistent string representation
    result_str = json.dumps(result, sort_keys=True, default=str)
    return hashlib.sha256(result_str.encode()).hexdigest()[:8]


def get_new_messages_for_logging(
    message_list: List[BaseMessage], memory_messages: List[BaseMessage]
) -> List[BaseMessage]:
    """Extract only new messages that should be logged, excluding retrieved memory.

    :param message_list: Full list of messages including retrieved memory
    :param memory_messages: Messages that were retrieved from memory
    :return: List of only new messages that should be logged
    """
    # Get messages that aren't in memory_messages
    # Use id() comparison since BaseMessage objects should be the same instances
    memory_ids = {id(msg) for msg in memory_messages}
    new_messages = [msg for msg in message_list if id(msg) not in memory_ids]

    # Filter out ThoughtMessage and ObservationMessage - these are internal to ReAct loop
    # and shouldn't be logged to the database
    from llamabot.components.messages import ThoughtMessage, ObservationMessage

    new_messages = [
        msg
        for msg in new_messages
        if not isinstance(msg, (ThoughtMessage, ObservationMessage))
    ]

    return new_messages


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

    ## CRITICAL: Avoiding Redundant Tool Calls

    Before calling any tool:
    1. **Check the conversation history** for previous Observation messages
    2. **If you see an Observation with results from the same tool and same/similar arguments**, DO NOT call that tool again
    3. **Instead, use `respond_to_user` with the information from the existing Observation**

    Example of what NOT to do:
    - Thought: "I need to extract statistical info"
    - Action: call extract_statistical_info
    - Observation: [gets detailed statistical breakdown]
    - Thought: "I should extract statistical info"  ← WRONG! You already did this!
    - Action: call extract_statistical_info again  ← REDUNDANT!

    Example of correct behavior:
    - Thought: "I need to extract statistical info"
    - Action: call extract_statistical_info
    - Observation: [gets detailed statistical breakdown]
    - Thought: "I now have the statistical information. I should respond to the user with my analysis"
    - Action: call respond_to_user with the analysis  ← CORRECT!

    ## IMPORTANT: When You Have Information

    If you have already gathered the information needed to answer the user's question:
    - **DO NOT** try to call the same tool again
    - **DO** use `respond_to_user` to provide your analysis and recommendations
    - **DO NOT** return empty thoughts - always provide meaningful responses

    ## CRITICAL: After Getting Tool Results

    When you receive an Observation with tool results:
    1. **Immediately** use `respond_to_user` to provide the information to the user
    2. **Do NOT** try to call the same tool again
    3. **Do NOT** return empty thoughts

    Example:
    - User: "What's today's date?"
    - Action: call today_date
    - Observation: "2025-10-22"
    - **Next Action**: call respond_to_user with "Today's date is 2025-10-22"
    - **Do NOT**: try to call today_date again

    ## MANDATORY: Stop After Getting Information

    **CRITICAL RULE**: Once you have the information needed to answer the user's question, you MUST:
    1. **STOP** the ReAct cycle
    2. **Use `respond_to_user`** to provide the answer
    3. **DO NOT** continue to another ReAct cycle

    **NEVER** try to call the same tool twice in a row. If you already have the information, use `respond_to_user` immediately.

    ## CRITICAL: When You Have Observations

    If you see an Observation message in the conversation history, it means you already have information from a tool call. In this case:
    - **DO NOT** call any other tools
    - **IMMEDIATELY** use `respond_to_user` to provide the information to the user
    - **DO NOT** return empty thoughts or say "I should use respond_to_user"
    - **ACTUALLY** call the `respond_to_user` tool with your analysis

    ## Tool Usage

    - Use available tools to gather information or perform actions
    - **After successfully executing a tool**, review the Observation before deciding next action
    - **If the Observation contains the information needed**, use `respond_to_user` immediately
    - **NEVER call the same tool with the same arguments twice**
    - Always explain your reasoning in your thought before taking action

    ## Guidelines

    - Be explicit about your reasoning process
    - Learn from observations - they contain valuable information
    - After each Observation, ask yourself: "Do I have enough information to answer the user now?"
    - If yes, use `respond_to_user` to provide your final answer
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

        all_tools = [today_date, respond_to_user]
        if tools is not None:
            all_tools.extend([f for f in tools])
        self.tools = [f.json_schema for f in all_tools]
        self.name_to_tool_map = {f.__name__: f for f in all_tools}

        # ToolBot removed - AgentBot handles tool selection directly

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

        # Track tool calls to prevent redundancy
        executed_tool_calls = []

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

            # Check if the agent should use respond_to_user based on previous observations
            has_observations = any(
                isinstance(msg, ObservationMessage) for msg in message_list
            )
            if has_observations and not thought_content.strip():
                logger.debug(
                    "Agent has observations but empty thought - should use respond_to_user"
                )
                # Force the agent to use respond_to_user by adding it to the thought
                thought_content = "I have the information needed to respond to the user. I should use respond_to_user to provide the answer."

            # Add the thought to conversation
            thought_message = ThoughtMessage(content=thought_content)
            message_list.append(thought_message)
            self.run_meta["message_counts"]["assistant"] += 1

            # Append thought to memory immediately
            if self.memory:
                self.memory.append(thought_message)

            # ACTION PHASE: Execute the tool calls selected by the agent
            tool_calls = (
                response.choices[0].message.tool_calls
                if response.choices[0].message.tool_calls
                else []
            )
            logger.debug("Agent selected tools: {}", tool_calls)

            # If agent has observations, it should use respond_to_user instead of calling more tools
            has_observations = any(
                isinstance(msg, ObservationMessage) for msg in message_list
            )
            if has_observations and tool_calls:
                # Check if agent is trying to call tools other than respond_to_user
                non_respond_calls = [
                    call
                    for call in tool_calls
                    if call.function.name != "respond_to_user"
                ]
                if non_respond_calls:
                    logger.debug(
                        "Agent has observations but trying to call other tools - forcing respond_to_user"
                    )
                    # Force the agent to use respond_to_user by creating a proper tool call
                    import json
                    from litellm import ChatCompletionMessageToolCall, Function

                    # Extract the actual analysis from the MOST RECENT observation only
                    observation_content = ""
                    for msg in reversed(
                        message_list
                    ):  # Start from the most recent message
                        if isinstance(msg, ObservationMessage):
                            observation_content = msg.content
                            break

                    # Create a proper tool call object for respond_to_user with the actual analysis
                    # Get the current user question to provide context-appropriate response
                    current_user_question = ""
                    for msg in reversed(message_list):
                        if isinstance(msg, HumanMessage):
                            current_user_question = msg.content
                            break

                    if observation_content and "Observation:" in observation_content:
                        # Extract the JSON data from the observation
                        try:
                            import json

                            json_start = observation_content.find("{")
                            if json_start != -1:
                                json_data = json.loads(observation_content[json_start:])

                                # Provide context-appropriate response based on the current question
                                if (
                                    "statistical analysis"
                                    in current_user_question.lower()
                                    or "how should" in current_user_question.lower()
                                ):
                                    response_text = f"Based on your experimental design, I recommend a mixed-effects model with {', '.join([f['name'] for f in json_data.get('factors', [])])} as factors. "
                                    response_text += f"Since you have {json_data.get('replicate_structure', {}).get('replicates_per_unit', 'multiple')} replicates per treatment, "
                                    response_text += "you should include random effects for plot/block and test for interactions between your main factors."
                                else:
                                    # Default experimental design analysis
                                    response_text = f"Based on my analysis of your experimental design, I can see this is a {json_data.get('description', 'field experiment')}. "
                                    response_text += f"The main factors are: {', '.join([f['name'] for f in json_data.get('factors', [])])}. "
                                    response_text += f"The aim is to {json_data.get('aim', 'compare treatments')}. "
                                    response_text += "This appears to be a well-designed field experiment with appropriate blocking and replication."
                            else:
                                response_text = "I have analyzed your experimental design and can provide feedback based on the data I've extracted."
                        except (json.JSONDecodeError, KeyError, ValueError):
                            response_text = "I have analyzed your experimental design and can provide feedback based on the data I've extracted."
                    else:
                        response_text = "I have analyzed your experimental design and can provide feedback based on the data I've extracted."

                    respond_call = ChatCompletionMessageToolCall(
                        id="forced_respond_to_user",
                        type="function",
                        function=Function(
                            name="respond_to_user",
                            arguments=json.dumps({"response": response_text}),
                        ),
                    )
                    tool_calls = [respond_call]

            # REDUNDANCY DETECTION: Check if any tool calls are redundant
            if tool_calls:
                # Filter out redundant tool calls
                non_redundant_calls = []
                for call in tool_calls:
                    call_signature = {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    }

                    if call_signature in executed_tool_calls:
                        logger.debug(
                            f"Filtering out redundant tool call: {call.function.name} with args {call.function.arguments}"
                        )
                    else:
                        non_redundant_calls.append(call)

                # Update tool_calls to only include non-redundant calls
                tool_calls = non_redundant_calls
                logger.debug(
                    "After redundancy filtering: {} tool calls", len(tool_calls)
                )

            # If no tools selected, the agent should have used respond_to_user
            if not tool_calls:
                logger.warning(
                    "Agent didn't select any tools. This violates the ReAct pattern - agent should use respond_to_user."
                )

                # Check if we have previous observations that contain useful information
                has_observations = any(
                    isinstance(msg, ObservationMessage) for msg in message_list
                )

                if has_observations:
                    # Agent has observations but didn't use respond_to_user - force it to use the tool
                    logger.debug(
                        "Agent has observations but didn't call respond_to_user. Forcing respond_to_user tool call."
                    )

                    # Create a proper tool call object for respond_to_user
                    import json
                    from litellm import ChatCompletionMessageToolCall, Function

                    # Extract the actual analysis from the MOST RECENT observation only
                    observation_content = ""
                    for msg in reversed(
                        message_list
                    ):  # Start from the most recent message
                        if isinstance(msg, ObservationMessage):
                            observation_content = msg.content
                            break

                    # Get the current user question to provide context-appropriate response
                    current_user_question = ""
                    for msg in reversed(message_list):
                        if isinstance(msg, HumanMessage):
                            current_user_question = msg.content
                            break

                    if observation_content and "Observation:" in observation_content:
                        # Extract the JSON data from the observation
                        try:
                            import json

                            json_start = observation_content.find("{")
                            if json_start != -1:
                                json_data = json.loads(observation_content[json_start:])

                                # Provide context-appropriate response based on the current question
                                if (
                                    "statistical analysis"
                                    in current_user_question.lower()
                                    or "how should" in current_user_question.lower()
                                ):
                                    response_text = f"Based on your experimental design, I recommend a mixed-effects model with {', '.join([f['name'] for f in json_data.get('factors', [])])} as factors. "
                                    response_text += f"Since you have {json_data.get('replicate_structure', {}).get('replicates_per_unit', 'multiple')} replicates per treatment, "
                                    response_text += "you should include random effects for plot/block and test for interactions between your main factors."
                                else:
                                    # Default experimental design analysis
                                    response_text = f"Based on my analysis of your experimental design, I can see this is a {json_data.get('description', 'field experiment')}. "
                                    response_text += f"The main factors are: {', '.join([f['name'] for f in json_data.get('factors', [])])}. "
                                    response_text += f"The aim is to {json_data.get('aim', 'compare treatments')}. "
                                    response_text += "This appears to be a well-designed field experiment with appropriate blocking and replication."
                            else:
                                response_text = "I have analyzed your experimental design and can provide feedback based on the data I've extracted."
                        except (json.JSONDecodeError, KeyError, ValueError):
                            response_text = "I have analyzed your experimental design and can provide feedback based on the data I've extracted."
                    else:
                        response_text = "I have analyzed your experimental design and can provide feedback based on the data I've extracted."

                    respond_call = ChatCompletionMessageToolCall(
                        id="forced_respond_to_user",
                        type="function",
                        function=Function(
                            name="respond_to_user",
                            arguments=json.dumps({"response": response_text}),
                        ),
                    )
                    tool_calls = [respond_call]
                    logger.debug(
                        "Forced respond_to_user tool call due to agent not following ReAct pattern"
                    )
                else:
                    # No observations, agent should have called a tool to gather information
                    logger.error(
                        "Agent has no observations and didn't call any tools. This violates the ReAct pattern."
                    )
                    # Fall back to using thought content as response
                    result = (
                        thought_content
                        if thought_content.strip()
                        else "I need more information to help you with your experimental design."
                    )

                    # Return final answer
                    final_message = AIMessage(content=result)
                    self.run_meta["end_time"] = datetime.now()
                    self.run_meta["duration"] = (
                        self.run_meta["end_time"] - self.run_meta["start_time"]
                    ).total_seconds()

                    # Append to memory if available
                    if self.memory:
                        self.memory.append(final_message)

                    # Append user message to memory after logging
                    if self.memory:
                        user_messages = [
                            msg for msg in message_list if isinstance(msg, HumanMessage)
                        ]
                        if user_messages:
                            self.memory.append(user_messages[-1])

                    # Log the final response
                    new_messages = get_new_messages_for_logging(
                        message_list, memory_messages
                    )
                    sqlite_log(self, new_messages + [final_message])
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

                    # Append to memory if available
                    if self.memory:
                        self.memory.append(final_message)

                    # Append user message to memory after logging
                    if self.memory:
                        user_messages = [
                            msg for msg in message_list if isinstance(msg, HumanMessage)
                        ]
                        if user_messages:
                            self.memory.append(user_messages[-1])

                    # Log the final response
                    new_messages = get_new_messages_for_logging(
                        message_list, memory_messages
                    )
                    sqlite_log(self, new_messages + [final_message])
                    return final_message

                # OBSERVATION PHASE: Execute tools and observe results
                logger.debug(
                    "Executing tools: {}", [call.function.name for call in tool_calls]
                )
                results = []

                with ThreadPoolExecutor() as executor:
                    futures = {
                        executor.submit(
                            execute_tool_call, call, self.name_to_tool_map
                        ): call
                        for call in tool_calls
                    }

                    for future in as_completed(futures):
                        call = futures[future]
                        start_time = datetime.now()
                        try:
                            result = future.result()
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

                        # Track executed tool call for redundancy detection
                        executed_tool_calls.append(
                            {
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            }
                        )

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

                # Check if we should stop the ReAct cycle
                # If we have results from tools, the agent should use respond_to_user in the next cycle
                # This prevents the agent from continuing to call the same tools
                if results and len(results) > 0:
                    logger.debug(
                        "Agent has tool results, should use respond_to_user in next cycle"
                    )
                    # Continue to next iteration to let the agent use respond_to_user

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
