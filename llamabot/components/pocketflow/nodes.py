"""PocketFlow node components for LlamaBot."""

import json
from typing import Callable, List

from pocketflow import Node

from llamabot.bot.toolbot import ToolBot

# Constant for the default loopback action name
DECIDE_NODE_ACTION = "decide"


def nodeify(func=None, *, loopback_name: str = DECIDE_NODE_ACTION):
    """Decorator to turn a function into a PocketFlow Node.

    This decorator wraps a function (or a @tool-decorated function) in a FuncNode,
    which is a PocketFlow Node that can execute the function and route back to
    a decision node in the flow graph.

    **Interaction with @tool decorator:**
    The `nodeify` decorator works seamlessly with `@tool` decorated functions.
    When wrapping a `@tool` function, the FuncNode preserves access to the tool's
    `json_schema` attribute (via `__getattr__` proxying), allowing ToolBot to
    discover and use the tool. The typical pattern is:
    ```python
    @tool
    def my_tool(arg: str) -> str:
        return arg

    wrapped_tool = nodeify(loopback_name="decide")(my_tool)
    # or equivalently:
    wrapped_tool = nodeify(loopback_name="decide")(tool(my_tool))
    ```

    **Decorator usage patterns:**
    - Without parentheses: `@nodeify` or `@nodeify(loopback_name=None)`
    - With parentheses: `nodeify(my_function)` or `nodeify(loopback_name="decide")(my_function)`

    **Loopback behavior:**
    - If `loopback_name` is a string (default: "decide"), the node will route back
      to the decision node after execution, allowing multi-step tool execution.
    - If `loopback_name` is `None`, the node is terminal and execution stops after
      this node completes (e.g., `respond_to_user`).

    **PocketFlow integration:**
    The returned FuncNode is a subclass of PocketFlow's `Node` base class, implementing
    the required `prep`, `exec`, and `post` methods. The `exec` method calls the
    wrapped function with arguments from the shared state's `func_call` dictionary.

    :param func: The function to wrap (when used without parentheses)
    :param loopback_name: The action name to use when looping back to the decide node.
                          Defaults to `DECIDE_NODE_ACTION` ("decide"). Set to `None`
                          for terminal nodes that don't loop back.
    :return: A FuncNode instance wrapping the function
    """

    def decorator(func):
        """Wrap a function in a FuncNode.

        :param func: The function to wrap
        :return: A FuncNode instance
        """

        class FuncNode(Node):
            """A PocketFlow Node that wraps a callable function."""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.loopback_name = loopback_name
                # Store reference to the original function
                self.func = func

            @property
            def name(self):
                """Return the name of the wrapped function.

                :return: The function name
                """
                return self.func.__name__

            def prep(self, shared):
                """Prepare the node for execution.

                :param shared: Shared state dictionary
                :return: The shared state
                """
                return shared

            def exec(self, prep_result):
                """Execute the wrapped function.

                :param prep_result: Prepared result from prep method
                :return: The result of calling the function
                """
                # Extract function call arguments from shared
                func_call = prep_result.get("func_call", {})

                # Call the function with the arguments
                return self.func(**func_call)

            def post(self, shared, prep_result, exec_res):
                """Post-process the execution result.

                :param shared: Shared state dictionary
                :param prep_result: Prepared result from prep method
                :param exec_res: Execution result
                :return: Loopback name or execution result for terminal nodes
                """
                shared["memory"].append(exec_res)

                if self.loopback_name is None:
                    return exec_res
                return self.loopback_name

            def __getattr__(self, name):
                """Proxy attribute access to the original function.

                This allows ToolBot to access json_schema and __name__
                from the original function.

                :param name: Attribute name to access
                :return: The attribute value from the wrapped function
                :raises AttributeError: If the attribute doesn't exist
                """
                if name == "func":
                    # Avoid infinite recursion when accessing self.func
                    raise AttributeError(
                        f"'{self.__class__.__name__}' object has no attribute '{name}'"
                    )
                # Proxy to the original function
                return getattr(self.func, name)

            def __call__(self, *args, **kwargs):
                """Make FuncNode callable, proxying to the original function.

                :param args: Positional arguments
                :param kwargs: Keyword arguments
                :return: Result of calling the wrapped function
                """
                return self.func(*args, **kwargs)

        return FuncNode()

    # If called without parentheses
    if func is not None:
        return decorator(func)

    # If called with parentheses
    return decorator


class DecideNode(Node):
    """A PocketFlow Node that decides which tool to execute using ToolBot.

    This node uses ToolBot to analyze the conversation history and decide
    which tool should be executed next based on the user's query.

    :param tools: List of tool functions (already wrapped with @tool and @nodeify)
    :param model_name: The name of the model to use for decision making
    """

    def __init__(
        self, tools: List[Callable], model_name: str = "gpt-4.1", *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.tools = tools
        self.model_name = model_name

    def prep(self, shared):
        """Prepare the node for execution.

        :param shared: Shared state dictionary
        :return: The shared state
        """
        return shared

    def exec(self, prep_res):
        """Decide which tool to use based on query.

        Uses ToolBot to analyze the conversation history and select a tool to execute.
        This method is the core decision-making logic of the AgentBot flow.

        **Input format:**
        The `prep_res` dictionary must contain a `"memory"` key with a list of strings
        representing the conversation history. Typically, this includes:
        - User queries
        - Tool execution results
        - Previous decision messages

        Example structure:
        ```python
        prep_res = {
            "memory": [
                "What is the date today?",
                "Chosen Tool: today_date",
                "2024-01-15"
            ]
        }
        ```

        **ToolBot interaction:**
        This method creates a ToolBot instance and passes `prep_res["memory"]` to it.
        ToolBot analyzes the conversation history and returns a list of tool call objects.
        Each tool call object has:
        - `function.name`: The name of the tool to execute (string)
        - `function.arguments`: JSON string containing the tool's arguments

        **Tool selection:**
        Only the first tool call from ToolBot's response is used. If ToolBot returns
        multiple tool calls, subsequent calls are ignored. This is by design, as
        AgentBot executes tools sequentially in a flow graph.

        **Shared state modification:**
        After extracting the tool name and arguments, this method:
        1. Parses the JSON arguments string into a dictionary
        2. Stores the parsed arguments in `prep_res["func_call"]` for the next node
        3. Returns the function name as the routing action/path

        The next node (the selected tool) will receive `prep_res["func_call"]` in its
        `exec` method and can unpack it as `**prep_res["func_call"]` to call the tool.

        **Error handling:**
        Raises `ValueError` in two cases:
        - If ToolBot returns no tool calls (empty list)
        - If the tool call arguments cannot be parsed as valid JSON

        :param prep_res: Prepared result from prep method. Must contain:
            - `"memory"`: List of strings representing conversation history
        :return: The name of the tool to execute (string), used for routing to the
                 corresponding tool node in the flow graph
        :raises ValueError: If no tool calls are returned from ToolBot or if JSON
                            parsing fails
        """
        from llamabot.prompt_library.agentbot import decision_bot_system_prompt

        bot = ToolBot(
            model_name=self.model_name,
            tools=self.tools,
            system_prompt=decision_bot_system_prompt(),
        )

        # Get tool calls from ToolBot
        tool_calls = bot(prep_res["memory"])

        # Handle case where no tool calls are returned
        if not tool_calls:
            raise ValueError("No tool calls returned from ToolBot")

        # Get the first tool call (ToolBot typically returns one)
        tool_call = tool_calls[0]

        # Extract function name and arguments
        func_name = tool_call.function.name
        func_args_json = tool_call.function.arguments

        # Parse JSON arguments string to dict
        try:
            func_args = json.loads(func_args_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse tool call arguments: {e}")

        # Store in shared for the next node
        prep_res["func_call"] = func_args

        # Return the function name as the action/path
        return func_name

    def post(self, shared, prep_res, exec_res):
        """Post-process the execution result.

        :param shared: Shared state dictionary
        :param prep_res: Prepared result from prep method
        :param exec_res: Execution result (tool name)
        :return: The tool name to route to
        """
        shared["memory"].append(f"Chosen Tool: {exec_res}")
        return exec_res
