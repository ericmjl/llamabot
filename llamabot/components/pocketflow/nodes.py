"""PocketFlow node components for LlamaBot."""

import json
from typing import Callable, List, Optional

from loguru import logger
from pocketflow import Node

from llamabot.recorder import Span, get_current_span, is_span_recording_enabled

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
    discover and use the tool. **Important:** `@nodeify` must be applied last
    (outermost decorator) so it wraps the `@tool`-decorated function. The typical
    pattern is:
    ```python
    @nodeify(loopback_name="decide")
    @tool
    def my_tool(arg: str) -> str:
        return arg
    ```

    **Decorator Order:**
    - `@tool` is applied first (innermost)
    - `@nodeify` is applied last (outermost)
    - This ensures the FuncNode can proxy to the tool-decorated function

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
                func_call = prep_result.get("func_call", {}).copy()

                # Inject globals_dict if available and function accepts it
                globals_dict = prep_result.get("globals_dict", {})
                import inspect

                sig = inspect.signature(self.func)
                if "_globals_dict" in sig.parameters:
                    func_call["_globals_dict"] = globals_dict

                # Add span support for tool execution
                if is_span_recording_enabled():
                    current_span = get_current_span()
                    if current_span:
                        tool_span = current_span.span(
                            "tool_call", tool_name=self.func.__name__, **func_call
                        )
                        with tool_span:
                            result = self.func(**func_call)
                            tool_span.log("tool_completed", result=str(result)[:200])
                            return result
                    else:
                        # No parent span, create root span
                        trace_id = prep_result.get("trace_id")
                        tool_span = Span(
                            "tool_call",
                            trace_id=trace_id,
                            tool_name=self.func.__name__,
                            **func_call,
                        )
                        with tool_span:
                            result = self.func(**func_call)
                            tool_span.log("tool_completed", result=str(result)[:200])
                            return result
                else:
                    # No span recording, execute normally
                    return self.func(**func_call)

            def post(self, shared, prep_result, exec_res):
                """Post-process the execution result.

                :param shared: Shared state dictionary
                :param prep_result: Prepared result from prep method
                :param exec_res: Execution result
                :return: Loopback name or None for terminal nodes
                """
                # Format error dicts nicely for the LLM to see in memory
                if isinstance(exec_res, dict) and "error" in exec_res:
                    # Format error message clearly for self-healing
                    error_msg = f"Error from {self.func.__name__}:\n{exec_res.get('error', 'Unknown error')}"
                    if "code" in exec_res:
                        error_msg += f"\n\nFailed code:\n{exec_res['code']}"
                    shared["memory"].append(error_msg)
                elif isinstance(exec_res, dict) and "created_variables" in exec_res:
                    # Code execution was successful - format success message with created variables
                    created_vars = exec_res.get("created_variables", [])
                    function_name = exec_res.get("function_name", "")
                    result = exec_res.get("result")

                    # Store the result in globals_dict so it can be returned to the user
                    # Use a predictable name based on the function name
                    if function_name and result is not None:
                        result_var_name = f"{function_name}_result"
                        # Update shared state's globals_dict so it persists
                        if "globals_dict" not in shared:
                            shared["globals_dict"] = {}
                        shared["globals_dict"][result_var_name] = result
                        success_msg = f"Code executed successfully. Function '{function_name}' was created and executed.\nThe result is stored in variable '{result_var_name}' and should be returned to the user using return_object_to_user."
                    else:
                        success_msg = f"Code executed successfully. Created variables: {', '.join(created_vars)}"
                        if function_name:
                            success_msg += f"\nFunction '{function_name}' was created and is now available in globals."
                        if result is not None:
                            success_msg += f"\nResult: {result}"

                    shared["memory"].append(success_msg)
                else:
                    shared["memory"].append(exec_res)

                if self.loopback_name is None:
                    # For terminal nodes, store result in shared state and return None
                    # to signal termination to PocketFlow. This prevents PocketFlow from
                    # trying to use the result (e.g., a DataFrame) as a routing action.
                    logger.info(
                        f"Terminal node reached: {self.func.__name__}. "
                        f"Storing result and terminating flow execution."
                    )
                    shared["result"] = exec_res
                    return None

                logger.debug(
                    f"Non-terminal node: {self.func.__name__}. "
                    f"Looping back to: {self.loopback_name}"
                )
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
    :param system_prompt: System prompt string to use for decision-making
    :param max_iterations: Maximum number of tool calls before forcing termination.
        If None, no limit is enforced. Defaults to None.
    :param completion_kwargs: Additional keyword arguments to pass to the
        completion function of `litellm` (e.g., `api_base`, `api_key`).
    """

    def __init__(
        self,
        tools: List[Callable],
        system_prompt: str,
        model_name: str = "gpt-4.1",
        max_iterations: Optional[int] = None,
        *args,
        **completion_kwargs,
    ):
        super().__init__(*args)

        self.tools = tools
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.completion_kwargs = completion_kwargs

    def prep(self, shared):
        """Prepare the node for execution.

        Increments iteration count and checks if max_iterations has been exceeded.
        If exceeded, sets a flag to force termination via respond_to_user.

        :param shared: Shared state dictionary
        :return: The shared state
        """
        # Initialize iteration_count if not present
        if "iteration_count" not in shared:
            shared["iteration_count"] = 0
            logger.debug("Initialized iteration_count in shared state")

        # Increment iteration count (this tracks how many times we've been to DecideNode)
        shared["iteration_count"] += 1
        current_iteration = shared["iteration_count"]

        logger.debug(
            f"DecideNode prep: iteration_count={current_iteration}, "
            f"max_iterations={self.max_iterations}"
        )

        # Check if we've exceeded max_iterations
        if (
            self.max_iterations is not None
            and shared["iteration_count"] > self.max_iterations
        ):
            # Set flag to force termination
            shared["_force_terminate"] = True
            logger.warning(
                f"Max iterations ({self.max_iterations}) exceeded "
                f"(current: {current_iteration}). "
                f"Forcing termination via respond_to_user."
            )

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
        # Add span support for decision making
        if is_span_recording_enabled():
            current_span = get_current_span()
            if current_span:
                decision_span = current_span.span(
                    "decision",
                    iteration=prep_res.get("iteration_count", 0),
                    model=self.model_name,
                )
                with decision_span:
                    return self._exec_decision(prep_res, decision_span)
            else:
                # No parent span, create root span
                trace_id = prep_res.get("trace_id")
                decision_span = Span(
                    "decision",
                    trace_id=trace_id,
                    iteration=prep_res.get("iteration_count", 0),
                    model=self.model_name,
                )
                with decision_span:
                    return self._exec_decision(prep_res, decision_span)
        else:
            return self._exec_decision(prep_res, None)

    def _exec_decision(self, prep_res, span_obj: Optional[Span]):
        """Internal method to execute decision logic.

        :param prep_res: Prepared result from prep method
        :param span_obj: Optional span object for logging
        :return: Function name to execute
        """
        # Check if we need to force termination due to max_iterations
        if prep_res.get("_force_terminate", False):
            logger.info(
                f"Force termination flag detected. "
                f"Max iterations ({self.max_iterations}) exceeded. "
                f"Current iteration: {prep_res.get('iteration_count', 'unknown')}"
            )

            # Find respond_to_user tool to force termination
            respond_to_user_tool = None
            for tool in self.tools:
                tool_name = getattr(tool, "func", tool).__name__
                if tool_name == "respond_to_user":
                    respond_to_user_tool = tool
                    break

            if respond_to_user_tool is None:
                # If respond_to_user not found, raise an error
                logger.error(
                    f"Max iterations ({self.max_iterations}) exceeded but "
                    f"respond_to_user tool not found. Cannot force termination."
                )
                raise RuntimeError(
                    f"Max iterations ({self.max_iterations}) exceeded and "
                    f"respond_to_user tool not found. Cannot force termination."
                )

            # Force respond_to_user with a termination message
            prep_res["func_call"] = {
                "response": (
                    f"Maximum iteration limit ({self.max_iterations}) reached. "
                    f"Terminating execution to prevent infinite loop."
                )
            }
            logger.warning(
                f"Forcing respond_to_user due to max_iterations limit ({self.max_iterations}). "
                f"Termination message prepared."
            )
            if span_obj:
                span_obj["forced_termination"] = True
                span_obj.log("forced_termination", reason="max_iterations_exceeded")
            return "respond_to_user"

        from llamabot.bot.toolbot import ToolBot

        # For Ollama models, enhance the system prompt to explicitly require tool calls
        # since they don't support tool_choice='required'
        is_ollama_model = self.model_name.startswith(
            "ollama"
        ) or self.model_name.startswith("ollama_chat")

        if is_ollama_model:
            # Append explicit requirement for tool calls to the system prompt
            tool_requirement_suffix = """

**CRITICAL FOR OLLAMA MODELS**: You MUST ALWAYS select and call a tool. Never return a response without calling a tool. Every user request requires you to select one of the available tools and execute it. This is mandatory - you cannot skip tool selection."""
            enhanced_system_prompt = self.system_prompt + tool_requirement_suffix
            logger.debug(
                f"Enhanced system prompt for Ollama model {self.model_name} "
                f"to explicitly require tool calls"
            )
        else:
            enhanced_system_prompt = self.system_prompt

        bot = ToolBot(
            model_name=self.model_name,
            tools=self.tools,
            system_prompt=enhanced_system_prompt,
            **self.completion_kwargs,
        )
        # Force tool calls - the model must always select a tool
        # Note: Ollama models don't support tool_choice='required', so we use 'auto' instead
        # and rely on the enhanced system prompt to require tool usage
        if is_ollama_model:
            bot.tool_choice = "auto"
            logger.debug(
                f"Using tool_choice='auto' for Ollama model {self.model_name} "
                f"(Ollama doesn't support tool_choice='required')"
            )
        else:
            bot.tool_choice = "required"

        # Get tool calls from ToolBot
        tool_calls = bot(prep_res["memory"])

        # Handle case where no tool calls are returned
        if not tool_calls:
            error_msg = (
                f"No tool calls returned from ToolBot. Model: {self.model_name}. "
            )
            if self.model_name.startswith("ollama") or self.model_name.startswith(
                "ollama_chat"
            ):
                error_msg += (
                    "Ollama models don't support tool_choice='required', so tool calls are optional. "
                    "Ensure your system prompt explicitly requires tool calls and that the model "
                    "supports function calling."
                )
            else:
                error_msg += (
                    "This may indicate the model doesn't support tool_choice='required'. "
                    "Check the system prompt and ensure it explicitly requires tool calls."
                )
            raise ValueError(error_msg)

        # Get the first tool call (ToolBot typically returns one)
        tool_call = tool_calls[0]

        # Extract function name and arguments
        func_name = tool_call.function.name
        func_args_json = tool_call.function.arguments

        # Log to span if available
        if span_obj:
            span_obj["chosen_tool"] = func_name
            span_obj.log("tool_selected", tool_name=func_name)

        # Log the chosen tool
        iteration_count = prep_res.get("iteration_count", "unknown")
        logger.info(
            f"DecideNode chose tool: {func_name} "
            f"(iteration: {iteration_count}/{self.max_iterations if self.max_iterations else 'unlimited'})"
        )

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
