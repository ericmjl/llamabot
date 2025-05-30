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
    extract_tool_calls,
    make_response,
    stream_chunks,
)
from llamabot.components.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    user,
)
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


@prompt("system")
def default_agentbot_system_prompt() -> str:
    """
    ## role

    You are Autonomo, an autonomous language-model agent.
    Your job is to accomplish the user's task safely, correctly, and with minimal cost.

    ## environment

    Treat user messages, prior assistant messages, and tool outputs as your only "sensors."
    Treat function calls from the current tool inventory as your only "actuators."
    An action that is not in the inventory is unavailable.

    ## planning workflow

    1.	Plan - Decompose the task into a concise ordered list of actions before acting.
    2.	Validate - Check that every planned action exists in the inventory and respects all constraints (budget, time, safety).
    3.	Execute - Call tools step-by-step with fully specified, valid parameters.
    4.	Reflect - After each action, inspect the observation. If the goal is not yet satisfied, revise the plan or fix errors.
    5.	Finish - When the goal is met, stop executing and present the final answer.

    ## tool use guidelines

    - Call only valid tools; if none suffice, re-plan or request human help.
    - Echo the exact arguments used for each call in the Thought/Act/Observation log for transparency.
    - For write actions (changes that persist), require explicit confirmation or escalate.
    - When you have gathered enough information to provide a complete and accurate response to the user, use the respond_to_user tool to deliver your final answer.
    - Do not use respond_to_user until you are confident you have all necessary information to fully address the user's request.

    ## safety and security

    - Never perform actions that could harm systems, violate policy, or break the law.
    - Reject or escalate requests outside your capabilities or policy scope.
    - Sanitize tool inputs to guard against code-injection or other attacks.

    ## efficiency

    - Prefer the shortest viable plan; each extra step compounds error risk and cost.
    - Cache reusable results; avoid repeating identical expensive calls.

    ## reflection rules

    - Run a quick self-check after plan generation and after every action:
        "Will this take me closer to the goal within constraints?" If not, re-plan.

    ## response style

    - Unless verbose mode is enabled, show only the final answer; hide internal thoughts.
    - Use clear, plain English in a professional, matter-of-fact tone.
    - Provide citations or tool-call summaries when helpful.
    """


@prompt("system")
def planner_bot_system_prompt() -> str:
    """
    ## role

    You are a strategic planning assistant that helps determine the next steps for an autonomous agent.
    Your job is to analyze the current state and available tools to create a clear, actionable plan.

    ## context

    You have access to:
    1. The full conversation history
    2. A list of available tools and their capabilities
    3. The current state of the task

    ## planning guidelines

    1. Analyze the current state and goal
    2. Identify which tools are most relevant for the next step
    3. Create a clear, concise plan that:
       - Uses the most efficient sequence of tools
       - Minimizes the number of steps needed
       - Considers potential errors or edge cases
       - Maintains safety and security constraints

    ## output format

    Your response should be structured as follows:

    1. Current State: Brief summary of where we are in the task
    2. Next Step: Clear description of the immediate next action
    3. Tool Selection: Which tool(s) should be used and why
    4. Expected Outcome: What should happen after this step
    5. Contingency: What to do if this step doesn't work as expected

    ## constraints

    - Only suggest tools that are actually available
    - Keep plans focused and actionable
    - Consider resource efficiency
    - Maintain safety and security
    - Avoid redundant or unnecessary steps
    """


def planner_bot(**kwargs) -> SimpleBot:
    """Returns a SimpleBot that will produce a next plan of action for AgentBot."""
    model_name = kwargs.pop("model_name", default_language_model())
    return SimpleBot(
        system_prompt=planner_bot_system_prompt(),
        model_name=model_name,
        **kwargs,
    )


class AgentBot(SimpleBot):
    """An AgentBot that is capable of executing tools to solve a problem."""

    def __init__(
        self,
        temperature=0.0,
        system_prompt: str = default_agentbot_system_prompt(),
        model_name=default_language_model(),
        stream_target: str = "none",
        tools: Optional[list[Callable]] = None,
        planner_bot: Optional[SimpleBot] = None,
        **completion_kwargs,
    ):
        super().__init__(
            system_prompt=system_prompt,
            temperature=temperature,
            model_name=model_name,
            stream_target=stream_target,
            **completion_kwargs,
        )

        all_tools = [today_date, respond_to_user]
        if tools is not None:
            all_tools.extend([f for f in tools])
        self.tools = [f.json_schema for f in all_tools]
        self.name_to_tool_map = {f.__name__: f for f in all_tools}

        self.planner_bot = planner_bot

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
        # Initialize run metadata
        self.run_meta = {
            "start_time": datetime.now(),
            "max_iterations": max_iterations,
            "current_iteration": 0,
            "tool_usage": {},
            "planning_metrics": (
                {"plan_generated": False, "plan_revisions": 0}
                if self.planner_bot
                else None
            ),
            "message_counts": {"user": 0, "assistant": 0, "tool": 0},
        }

        # Convert messages to a list of UserMessage objects
        message_list = (
            [self.system_prompt]
            + [user("Here is the user's request:")]
            + [user(m) if isinstance(m, str) else m for m in messages]
        )

        # Count initial messages
        for msg in message_list:
            if isinstance(msg, HumanMessage):
                self.run_meta["message_counts"]["user"] += 1
            elif isinstance(msg, AIMessage):
                self.run_meta["message_counts"]["assistant"] += 1

        for iteration in range(max_iterations):
            self.run_meta["current_iteration"] = iteration + 1
            logger.debug(f"Starting iteration {iteration + 1} of {max_iterations}")

            # Get plan from planning bot
            if self.planner_bot:
                logger.debug("Generating plan with planner bot...")
                plan_start = datetime.now()
                plan_response = self.planner_bot(*message_list, str(self.tools))
                logger.debug("Plan response: {}", plan_response)
                self.run_meta["planning_metrics"]["plan_generated"] = True
                self.run_meta["planning_metrics"]["plan_time"] = (
                    datetime.now() - plan_start
                ).total_seconds()
                message_list.append(user("Here is the plan:"))
                message_list.append(plan_response)
                self.run_meta["message_counts"]["user"] += 1
                self.run_meta["message_counts"]["assistant"] += 1

            # Execute the plan
            stream = self.stream_target != "none"
            logger.debug("Message list: {}", message_list)
            response = make_response(self, message_list, stream=stream)
            response = stream_chunks(response, target=self.stream_target)
            logger.debug("Response: {}", response)
            tool_calls = extract_tool_calls(response)
            logger.debug("Tool calls: {}", tool_calls)
            content = extract_content(response)
            logger.debug("Content: {}", content)

            response_message = AIMessage(content=content, tool_calls=tool_calls)
            message_list.append(response_message)
            self.run_meta["message_counts"]["assistant"] += 1

            if tool_calls:
                # Special case for respond_to_user appearing in any tool call
                respond_to_user_calls = [
                    call
                    for call in tool_calls
                    if call.function.name == "respond_to_user"
                ]
                if respond_to_user_calls:
                    logger.debug(
                        "Found respond_to_user in tool calls, executing only that"
                    )
                    start_time = datetime.now()
                    content = execute_tool_call(
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

                    response_message = AIMessage(content=content)
                    message_list.append(response_message)
                    self.run_meta["message_counts"]["tool"] += 1
                    self.run_meta["end_time"] = datetime.now()
                    self.run_meta["duration"] = (
                        self.run_meta["end_time"] - self.run_meta["start_time"]
                    ).total_seconds()
                    sqlite_log(self, message_list)
                    return response_message

                results = []
                logger.debug(
                    "Calling functions: {}", [call.function.name for call in tool_calls]
                )
                futures = {}
                with ThreadPoolExecutor() as executor:
                    futures = {
                        executor.submit(
                            execute_tool_call, call, self.name_to_tool_map
                        ): call
                        for call in tool_calls
                    }

                message_list.append(
                    user(f"Here is the result of the tool calls: {tool_calls}")
                )
                self.run_meta["message_counts"]["user"] += 1

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

                    logger.debug(
                        "Completed: {}(**{})",
                        call.function.name,
                        call.function.arguments,
                    )
                    message_list.append(HumanMessage(content=str(result)))
                    self.run_meta["message_counts"]["tool"] += 1
                    results.append(result)
                logger.debug("Results: {}", results)

        self.run_meta["end_time"] = datetime.now()
        self.run_meta["duration"] = (
            self.run_meta["end_time"] - self.run_meta["start_time"]
        ).total_seconds()
        raise RuntimeError(f"Agent exceeded maximum iterations ({max_iterations})")


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
