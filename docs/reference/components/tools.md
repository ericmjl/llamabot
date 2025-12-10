# Tools API Reference

The tools system provides function calling capabilities for bots, with built-in observability and AgentBot integration.

## Tool Decorator

```python
@tool(
    *,
    loopback_name: Optional[str] = "decide",
    span: bool = True,
    exclude_args: Optional[List[str]] = None,
    operation_name: Optional[str] = None,
    **span_attributes,
)
```

Decorator to create a tool from a function. Provides built-in AgentBot integration and observability.

### Parameters

- **loopback_name** (`Optional[str]`, default: `"decide"`): Controls whether execution continues after this tool. Set to `None` for terminal tools that end the workflow. Defaults to `"decide"` to loop back to the decision node.

- **span** (`bool`, default: `True`): Whether to apply `@span` decorator for observability. Enables automatic logging of tool calls.

- **exclude_args** (`Optional[List[str]]`, default: `None`): Span parameter: exclude specific arguments from logging. Useful for sensitive data like API keys or passwords.

- **operation_name** (`Optional[str]`, default: `None`): Span parameter: custom operation name for logging. If not provided, uses the function name.

- **span_attributes**: Additional span attributes passed to `@span` decorator.

### Returns

- **Callable**: The decorated function with:
  - `json_schema` attribute: JSON schema for function calling
  - `func` attribute: AgentBot integration function
  - `loopback_name` attribute: Loopback configuration

### Usage Examples

#### Basic Tool

```python
from llamabot.components.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    :param city: The name of the city
    :return: Weather information
    """
    return f"The weather in {city} is sunny, 72°F"
```

#### Terminal Tool

```python
from llamabot.components.tools import tool

@tool(loopback_name=None)
def respond_to_user(response: str) -> str:
    """Respond to the user with a message.

    :param response: The message to send
    :return: The response message
    """
    return response
```

#### Tool with Excluded Arguments

```python
from llamabot.components.tools import tool

@tool(exclude_args=["api_key", "password"])
def secure_tool(api_key: str, password: str, data: str) -> str:
    """Process data securely.

    :param api_key: API key (excluded from logs)
    :param password: Password (excluded from logs)
    :param data: Data to process
    :return: Processed data
    """
    return f"Processed: {data}"
```

#### Custom Operation Name

```python
from llamabot.components.tools import tool

@tool(operation_name="custom_weather_lookup")
def get_weather(city: str) -> str:
    """Get weather."""
    return f"Weather in {city}"
```

## Tool Requirements

### Function Signature

Tools must have:

- Type annotations for all parameters
- Return type annotation
- Docstring with parameter descriptions

### Docstring Format

Use Sphinx-style docstrings:

```python
@tool
def my_tool(param1: str, param2: int) -> str:
    """Tool description.

    :param param1: Description of param1
    :param param2: Description of param2
    :return: Description of return value
    """
    return result
```

## Default Tools

The following tools are automatically available in AgentBot and ToolBot:

### `today_date`

```python
@tool
def today_date() -> str:
    """Returns the current date in YYYY-MM-DD format."""
```

Returns the current date. Loops back to decision node.

### `respond_to_user`

```python
@tool(loopback_name=None)
def respond_to_user(response: str) -> str:
    """Respond to the user with a message."""
```

Sends a text response to the user. Terminal tool (no loopback).

### `return_object_to_user`

```python
@tool(loopback_name=None)
def return_object_to_user(variable_name: str, _globals_dict: dict = None) -> Any:
    """Return an object from the calling context's globals."""
```

Returns an object from the calling context's globals dictionary. Terminal tool (no loopback).

### `inspect_globals`

```python
@tool
def inspect_globals(_globals_dict: dict = None) -> str:
    """Inspect available global variables."""
```

Inspects available global variables. Loops back to decision node.

## Tool Functions

### `write_and_execute_code`

```python
def write_and_execute_code(globals_dict: dict) -> Callable
```

Creates a tool for executing Python code with access to global variables.

#### Parameters

- **globals_dict** (`dict`): Dictionary of global variables to make available to executed code (typically `globals()`)

#### Returns

- **Callable**: A tool function that can execute Python code

#### Example

```python
from llamabot.components.tools import write_and_execute_code

code_tool = write_and_execute_code(globals_dict=globals())

# Use with ToolBot or AgentBot
bot = lmb.ToolBot(
    system_prompt="You are a code executor.",
    tools=[code_tool]
)
```

### `write_and_execute_script`

```python
def write_and_execute_script(
    code: str,
    dependencies_str: str,
    python_version: str,
) -> str
```

Executes Python scripts in a secure sandbox.

#### Parameters

- **code** (`str`): Python code to execute
- **dependencies_str** (`str`): Comma-separated list of dependencies
- **python_version** (`str`): Python version to use

#### Returns

- **str**: Execution result

## Tool Schema

Tools automatically generate a JSON schema for function calling:

```python
@tool
def my_tool(arg: str) -> str:
    return arg

# Access the schema
schema = my_tool.json_schema
# {
#     "type": "function",
#     "function": {
#         "name": "my_tool",
#         "description": "...",
#         "parameters": {...}
#     }
# }
```

## Usage with Bots

### ToolBot

```python
import llamabot as lmb

@lmb.tool
def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers."""
    return a + b

bot = lmb.ToolBot(
    system_prompt="You are a calculator.",
    model_name="gpt-4o-mini",
    tools=[calculate_sum]
)

tool_calls = bot("Calculate 5 + 3")
```

### AgentBot

```python
import llamabot as lmb

@lmb.tool
def search_web(query: str) -> str:
    """Search the web."""
    return results

@lmb.tool(loopback_name=None)
def respond_to_user(response: str) -> str:
    """Respond to user."""
    return response

bot = lmb.AgentBot(
    tools=[search_web, respond_to_user],
    model_name="gpt-4o-mini"
)

result = bot("Search for AI news and summarize")
```

## Observability

Tools decorated with `@tool` automatically include span-based logging:

- Tool calls are logged with parameters
- Results are logged
- Execution time is tracked
- Sensitive arguments can be excluded

View logs using the log viewer:

```bash
llamabot logviewer
```

## Best Practices

1. **Clear docstrings**: Provide detailed descriptions of what tools do
2. **Type annotations**: Always include type hints for all parameters
3. **Exclude sensitive data**: Use `exclude_args` for API keys, passwords, etc.
4. **Terminal tools**: Use `loopback_name=None` for tools that end workflows
5. **Error handling**: Include proper error handling in tool implementations

## Related Classes

- **ToolBot**: Single-turn tool execution bot
- **AgentBot**: Multi-turn agent with tool orchestration
- **DecideNode**: Decision-making node for AgentBot

## See Also

- [ToolBot Reference](../bots/toolbot.md)
- [AgentBot Reference](../bots/agentbot.md)
- [ToolBot Tutorial](../../tutorials/toolbot.md)
- [AgentBot Tutorial](../../tutorials/agentbot.md)
