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


def _validate_tools(tools: List[Callable]) -> None:
    """Validate that all tools are properly decorated with @tool and @nodeify.

    :param tools: List of tool functions to validate
    :raises ValueError: If any tool is not properly decorated
    """
    invalid_tools = []
    for tool_func in tools:
        issues = []

        # Check for @tool decorator (has json_schema attribute)
        if not hasattr(tool_func, "json_schema"):
            issues.append("missing @tool decorator")

        # Check for @nodeify decorator (has func attribute)
        if not hasattr(tool_func, "func"):
            issues.append("missing @nodeify decorator")

        # Check for loopback_name attribute (from nodeify)
        if not hasattr(tool_func, "loopback_name"):
            issues.append("missing loopback_name attribute (from @nodeify)")

        if issues:
            tool_name = getattr(tool_func, "__name__", str(tool_func))
            invalid_tools.append((tool_name, issues))

    if invalid_tools:
        error_parts = ["The following tools are not properly decorated:"]
        for tool_name, issues in invalid_tools:
            error_parts.append(f"\n  - {tool_name}:")
            for issue in issues:
                error_parts.append(f"    • {issue}")

        error_parts.append("\nTo fix this, decorate your tools like this:")
        error_parts.append("\n  from llamabot.components.tools import tool")
        error_parts.append(
            "  from llamabot.components.pocketflow import nodeify, DECIDE_NODE_ACTION"
        )
        error_parts.append("\n  @nodeify(loopback_name=DECIDE_NODE_ACTION)")
        error_parts.append("  @tool")
        error_parts.append("  def your_tool(arg: str) -> str:")
        error_parts.append("      return arg")
        error_parts.append("\n  bot = AgentBot(tools=[your_tool])")
        error_parts.append(
            "\n  Note: @nodeify must be applied last (outermost decorator)"
        )

        raise ValueError("\n".join(error_parts))


class AgentBot:
    """An AgentBot that uses PocketFlow for tool orchestration.

    This bot requires user-provided tools to be decorated with both `@tool` and
    `@nodeify` decorators. It creates a decision node that uses ToolBot to select
    tools and executes them through a PocketFlow graph.

    **Tool Requirements:**
    Tools must be decorated with both `@tool` and `@nodeify` before being
    passed to AgentBot. **Important:** `@nodeify` must be applied last (outermost
    decorator) so it wraps the `@tool`-decorated function. Example:

    ```python
    from llamabot.components.tools import tool
    from llamabot.components.pocketflow import nodeify, DECIDE_NODE_ACTION

    @nodeify(loopback_name=DECIDE_NODE_ACTION)
    @tool
    def my_tool(arg: str) -> str:
        return arg

    bot = AgentBot(tools=[my_tool])
    ```

    **Decorator Order:**
    - `@tool` is applied first (innermost)
    - `@nodeify` is applied last (outermost)
    - This ensures the FuncNode can proxy to the tool-decorated function

    For terminal tools (like `respond_to_user`), use `@nodeify(loopback_name=None)`.

    :param tools: List of tools that must be decorated with both @tool and @nodeify
    :param decide_node: Optional custom decision node (defaults to DecideNode).
        If provided, overrides `system_prompt` parameter.
    :param system_prompt: System prompt string for decision-making.
        If None, uses the default `decision_bot_system_prompt` from `llamabot.prompt_library.agentbot`.
        Only used if `decide_node` is None.
    :param model_name: The name of the model to use for decision making
    :param max_iterations: Maximum number of tool calls before forcing termination.
        If None, no limit is enforced. Defaults to None.
    :param completion_kwargs: Additional keyword arguments to pass to the
        completion function of `litellm` (e.g., `api_base`, `api_key`).
    :raises ValueError: If any tool is not properly decorated with @tool and @nodeify
    """

    def __init__(
        self,
        tools: List[Callable],
        decide_node: Optional[Node] = None,
        system_prompt: Optional[str] = None,
        model_name: str = "gpt-4.1",
        max_iterations: Optional[int] = None,
        **completion_kwargs,
    ):
        # Validate that all user-provided tools are properly decorated
        _validate_tools(tools)

        # Combine default and user-provided tools
        # Default tools are already nodeified
        all_tools = list(DEFAULT_TOOLS) + tools

        self.tools = all_tools

        # Create decide node if not provided
        if decide_node is None:
            # Generate default system prompt if not provided
            if system_prompt is None:
                from llamabot.components.messages import BaseMessage
                from llamabot.prompt_library.agentbot import decision_bot_system_prompt

                prompt_result = decision_bot_system_prompt()
                if isinstance(prompt_result, BaseMessage):
                    system_prompt = prompt_result.content
                else:
                    system_prompt = prompt_result

            decide_node = DecideNode(
                tools=all_tools,
                system_prompt=system_prompt,
                model_name=model_name,
                max_iterations=max_iterations,
                **completion_kwargs,
            )

        self.decide_node = decide_node
        self.max_iterations = max_iterations

        # Always ensure decide_node has the correct tools
        # This ensures consistency even when a custom decide_node is provided
        if hasattr(decide_node, "tools"):
            decide_node.tools = all_tools

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
        self.shared = dict(memory=[], globals_dict={})

        # Track all trace_ids from this bot instance for span visualization
        self._trace_ids = []

    def __call__(self, query: str, globals_dict: Optional[dict] = None):
        """Execute the agent with a query.

        :param query: The user's query string
        :param globals_dict: Optional dictionary of global variables from the calling context.
            This allows tools like `return_object_to_user` to access variables from the
            calling scope (e.g., from a notebook's globals()). If provided, updates the
            globals_dict in shared state. If None, preserves existing globals_dict.
        :return: The result from running the flow
        """
        from llamabot.recorder import Span, is_span_recording_enabled

        # Initialize shared state if it doesn't exist
        if not hasattr(self, "shared") or self.shared is None:
            self.shared = dict(memory=[], globals_dict={})

        # Preserve memory and append the new query
        if "memory" not in self.shared:
            self.shared["memory"] = []
        self.shared["memory"].append(query)

        # Update globals_dict if provided, otherwise preserve existing
        if globals_dict is not None:
            # Merge with existing globals_dict to preserve any variables created during execution
            if "globals_dict" not in self.shared:
                self.shared["globals_dict"] = {}
            # Update with new globals_dict (this will add new keys and update existing ones)
            self.shared["globals_dict"].update(globals_dict)

        # Ensure globals_dict exists
        if "globals_dict" not in self.shared:
            self.shared["globals_dict"] = {}

        # Initialize iteration tracking for loop detection
        if "iteration_count" not in self.shared:
            self.shared["iteration_count"] = 0

        # Add span support for agent execution
        if is_span_recording_enabled():
            import uuid

            # Always create a new trace_id for bot calls to avoid inheriting from context
            new_trace_id = str(uuid.uuid4())
            agent_span = Span(
                "agentbot_call",
                trace_id=new_trace_id,
                query=query,
                max_iterations=self.max_iterations,
            )
            # Track trace_id for this bot instance
            if agent_span.trace_id not in self._trace_ids:
                self._trace_ids.append(agent_span.trace_id)
            with agent_span:
                # Store trace_id in shared state so nodes can access it
                self.shared["trace_id"] = agent_span.trace_id
                # Run the flow
                self.flow.run(self.shared)
                # Record result in span
                result = self.shared.get("result")
                agent_span["result"] = str(result)[:200] if result else None
                agent_span["iterations"] = self.shared.get("iteration_count", 0)
                return result
        else:
            # Run the flow without spans
            self.flow.run(self.shared)
            # Retrieve result from shared state (set by terminal nodes)
            return self.shared.get("result")

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

    def display_spans(self) -> str:
        """Display all spans from all bot calls as HTML.

        Queries spans associated with all trace_ids from this bot instance
        and generates an HTML visualization showing all spans from all calls.

        :return: HTML string for displaying spans in marimo notebooks
        """
        from llamabot.recorder import build_hierarchy, generate_span_html, get_spans

        if not self._trace_ids:
            return '<div style="padding: 1rem; color: #2E3440;">No spans recorded for this bot instance yet.</div>'

        # Collect all spans from all trace_ids for this bot instance
        all_spans = []
        for trace_id in self._trace_ids:
            spans = get_spans(trace_id=trace_id)
            all_spans.extend(spans)

        if not all_spans:
            return '<div style="padding: 1rem; color: #2E3440;">No spans found in database for this bot instance.</div>'

        # Find root spans (spans with no parent) to use as current span
        # Use the most recent root span (last one in the list)
        root_spans = [s for s in all_spans if s.get("parent_span_id") is None]
        if root_spans:
            # Use the last root span (most recent) as the current span for highlighting
            current_span_dict = root_spans[-1]
            current_span_id = current_span_dict["span_id"]
        else:
            # Fallback to last span if no root spans found
            current_span_dict = all_spans[-1]
            current_span_id = current_span_dict["span_id"]

        # Build hierarchical structure
        trace_tree = build_hierarchy(all_spans)

        # Generate HTML visualization
        return generate_span_html(
            span_dict=current_span_dict,
            all_spans=all_spans,
            trace_tree=trace_tree,
            current_span_id=current_span_id,
        )

    def _repr_html_(self) -> str:
        """Return HTML representation for marimo display.

        When an AgentBot object is the last expression in a marimo cell,
        this method is automatically called to display the spans visualization
        from the most recent bot call.

        :return: HTML string for displaying spans
        """
        return self.display_spans()
