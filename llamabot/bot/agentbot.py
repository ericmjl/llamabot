"""A module implementing an agent-based bot that uses PocketFlow for orchestration.

This module provides the AgentBot class, which combines language model capabilities
with PocketFlow graph-based orchestration to execute tools based on user input.
The bot uses a decision-making node to determine which tools to call
and in what order, making it suitable for complex, multi-step tasks.
"""

from typing import Callable, List, Optional

from pocketflow import Flow, Node

from llamabot.components.pocketflow import DecideNode
from llamabot.components.tools import DEFAULT_TOOLS


class AgentBot:
    """An AgentBot that uses PocketFlow for tool orchestration.

    This bot expects user-provided tools to already be decorated with both
    @tool and @nodeify decorators. It creates a decision node that uses
    ToolBot to select tools and executes them through a PocketFlow graph.

    **Tool Requirements:**
    Tools must be decorated with both `@tool` and `@nodeify` before being
    passed to AgentBot. Example:

    ```python
    from llamabot import tool, nodeify

    @tool
    @nodeify(loopback_name="decide")
    def my_tool(arg: str) -> str:
        return arg

    bot = AgentBot(tools=[my_tool])
    ```

    For terminal tools (like `respond_to_user`), use `@nodeify(loopback_name=None)`.

    :param tools: List of tools already decorated with @tool and @nodeify
    :param decide_node: Optional custom decision node (defaults to DecideNode)
    :param model_name: The name of the model to use for decision making
    :param completion_kwargs: Additional keyword arguments to pass to the
        completion function of `litellm` (e.g., `api_base`, `api_key`).
    """

    def __init__(
        self,
        tools: List[Callable],
        decide_node: Optional[Node] = None,
        model_name: str = "gpt-4.1",
        **completion_kwargs,
    ):
        # Combine default and user-provided tools
        # Default tools are already nodeified, user tools should be too
        all_tools = list(DEFAULT_TOOLS) + tools

        self.tools = all_tools

        # Create decide node if not provided
        if decide_node is None:
            decide_node = DecideNode(
                tools=all_tools, model_name=model_name, **completion_kwargs
            )

        self.decide_node = decide_node

        # Build PocketFlow graph: connect tools to decide node
        for tool_node in all_tools:
            # Get the function name - could be from .func or .name attribute
            if hasattr(tool_node, "func"):
                tool_name = tool_node.func.__name__
            elif hasattr(tool_node, "name"):
                tool_name = tool_node.name
            else:
                tool_name = tool_node.__name__

            self.decide_node - tool_name >> tool_node
            if (
                hasattr(tool_node, "loopback_name")
                and tool_node.loopback_name is not None
            ):
                tool_node - tool_node.loopback_name >> self.decide_node

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
