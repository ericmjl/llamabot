"""A module implementing an agent-based bot that uses PocketFlow for orchestration.

This module provides the AgentBot class, which combines language model capabilities
with PocketFlow graph-based orchestration to execute tools based on user input.
The bot uses a decision-making node to determine which tools to call
and in what order, making it suitable for complex, multi-step tasks.
"""

from typing import Callable, List, Optional

from pocketflow import Flow, Node

from llamabot.components.pocketflow import DECIDE_NODE_ACTION, DecideNode, nodeify
from llamabot.components.tools import DEFAULT_TOOLS, respond_to_user, tool


class AgentBot:
    """An AgentBot that uses PocketFlow for tool orchestration.

    This bot wraps user-provided callables with @tool and @nodeify decorators,
    creates a decision node that uses ToolBot to select tools, and executes
    them through a PocketFlow graph.

    :param tools: List of callable functions to use as tools
    :param decide_node: Optional custom decision node (defaults to DecideNode)
    :param model_name: The name of the model to use for decision making
    """

    def __init__(
        self,
        tools: List[Callable],
        decide_node: Optional[Node] = None,
        model_name: str = "gpt-4.1",
    ):
        # Combine default and user-provided tools
        all_tools = DEFAULT_TOOLS + tools

        # Wrap all tools with @tool and @nodeify decorators
        # respond_to_user should be terminal (no loopback)
        wrapped_tools = []
        for tool_func in all_tools:
            # Check if it's respond_to_user (by identity, not name) to make it terminal
            if tool_func is respond_to_user:
                wrapped = nodeify(loopback_name=None)(tool(tool_func))
            else:
                wrapped = nodeify(loopback_name=DECIDE_NODE_ACTION)(tool(tool_func))
            wrapped_tools.append(wrapped)

        self.tools = wrapped_tools

        # Create decide node if not provided
        if decide_node is None:
            decide_node = DecideNode(tools=wrapped_tools, model_name=model_name)

        self.decide_node = decide_node

        # Build PocketFlow graph: connect tools to decide node
        for wrapped_tool in wrapped_tools:
            self.decide_node - wrapped_tool.func.__name__ >> wrapped_tool
            if wrapped_tool.loopback_name is not None:
                wrapped_tool - wrapped_tool.loopback_name >> self.decide_node

        # Create the flow
        self.flow = Flow(start=self.decide_node)

        # Initialize shared state
        self.shared = dict(memory=[])

    def __call__(self, query: str):
        """Execute the agent with a query.

        :param query: The user's query string
        :return: The result from running the flow
        """
        # Reset shared state for this call
        self.shared["memory"].append(query)

        # Run the flow
        result = self.flow.run(self.shared)

        return result

    def _display_(self):
        """Display the agent's flow graph as a Mermaid diagram.

        Requires marimo to be installed in the environment.

        :return: A marimo Mermaid visualization
        :raises ImportError: If marimo is not installed
        """
        try:
            import marimo as mo
        except ImportError:
            raise ImportError(
                "marimo is required for AgentBot visualization. "
                "Please install marimo in your environment."
            )
        from llamabot.components.pocketflow import flow_to_mermaid

        return mo.mermaid(flow_to_mermaid(self.flow))
