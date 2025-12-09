# ToolBot API Reference

ToolBot is a single-turn bot designed for tool execution and function calling. It analyzes user requests and selects the most appropriate tool to execute.

## Class Definition

```python
class ToolBot(SimpleBot):
    """A single-turn bot that can execute tools.

    This bot is designed to analyze user requests and determine the most appropriate
    tool or function to execute. It's a generalization of other bot types, focusing
    on tool selection and execution rather than multi-turn conversation.
    """
```

## Constructor

```python
def __init__(
    self,
    system_prompt: str,
    model_name: str,
    tools: Optional[List[Callable]] = None,
    chat_memory: Optional[ChatMemory] = None,
    **completion_kwargs,
)
```

### Parameters

- **system_prompt** (`str`): The system prompt to use for the bot. This defines the bot's behavior and tool selection strategy.

- **model_name** (`str`): The name of the model to use. Must support function calling. Examples: `"gpt-4o-mini"`, `"gpt-4o"`, `"anthropic/claude-3-5-sonnet"`.

- **tools** (`Optional[List[Callable]]`, default: `None`): Optional list of additional tools to include. Tools should be decorated with `@tool` or be callable functions. Default tools (`today_date`, `respond_to_user`, `return_object_to_user`, `inspect_globals`) are automatically included.

- **chat_memory** (`Optional[ChatMemory]`, default: `None`): Chat memory component for context retrieval. If not provided, a new `ChatMemory()` instance is created.

- **completion_kwargs**: Additional keyword arguments to pass to the completion function of `litellm`.

## Methods

### `__call__`

```python
def __call__(
    self,
    *messages: Union[str, BaseMessage, list[Union[str, BaseMessage]], Callable],
    execution_history: Optional[List[Dict]] = None,
) -> List[ChatCompletionMessageToolCall]
```

Process messages and return tool calls to execute.

#### Parameters

- **messages**: One or more messages to process. Can be strings, `BaseMessage` objects, lists of messages, or callable functions that return strings.

- **execution_history** (`Optional[List[Dict]]`, default: `None`): Optional list of previously executed tool calls for context. Each dict should contain:
  - `tool_name`: Name of the tool
  - `args`: Arguments passed to the tool
  - `result`: Result of the tool execution
  - `was_cached`: Whether the result was cached

#### Returns

- **List[ChatCompletionMessageToolCall]**: List of tool calls to execute. Each tool call contains:
  - `function.name`: Name of the function to call
  - `function.arguments`: JSON string of arguments

#### Example

```python
import llamabot as lmb
from llamabot.components.tools import write_and_execute_code

bot = lmb.ToolBot(
    system_prompt="You are a data analyst.",
    model_name="gpt-4o-mini",
    tools=[write_and_execute_code(globals_dict=globals())]
)

tool_calls = bot("Calculate the mean of the sales_data DataFrame")
# Returns list of tool calls to execute
```

## Default Tools

ToolBot automatically includes these default tools:

- **`today_date()`**: Returns the current date in YYYY-MM-DD format
- **`respond_to_user(response: str)`**: Responds to the user with a text message
- **`return_object_to_user(variable_name: str)`**: Returns an object from the calling context's globals dictionary
- **`inspect_globals()`**: Inspects available global variables

## Attributes

- **tools** (`List[Dict]`): List of tool JSON schemas
- **name_to_tool_map** (`Dict[str, Callable]`): Mapping from tool names to tool functions
- **chat_memory** (`ChatMemory`): The chat memory component

## Usage Examples

### Basic Tool Execution

```python
import llamabot as lmb
from llamabot.components.tools import write_and_execute_code

bot = lmb.ToolBot(
    system_prompt="You are a helpful assistant that can execute Python code.",
    model_name="gpt-4o-mini",
    tools=[write_and_execute_code(globals_dict=globals())]
)

tool_calls = bot("Calculate 2 + 2")
```

### With Custom Tools

```python
import llamabot as lmb

@lmb.tool
def calculate_fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number.

    :param n: The position in the Fibonacci sequence
    :return: The nth Fibonacci number
    """
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

bot = lmb.ToolBot(
    system_prompt="You are a mathematical assistant.",
    model_name="gpt-4o-mini",
    tools=[calculate_fibonacci]
)

tool_calls = bot("Calculate the 10th Fibonacci number")
```

### With Global Variables

```python
import pandas as pd
import numpy as np
import llamabot as lmb
from llamabot.components.tools import write_and_execute_code

# Create some data
data = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100)
})

bot = lmb.ToolBot(
    system_prompt="You are a data analyst.",
    model_name="gpt-4o-mini",
    tools=[write_and_execute_code(globals_dict=globals())]
)

tool_calls = bot("Calculate the correlation between x and y in the data DataFrame")
```

### With Chat Memory

```python
import llamabot as lmb
from llamabot.components.tools import write_and_execute_code

memory = lmb.ChatMemory()

bot = lmb.ToolBot(
    system_prompt="You are a data analysis assistant.",
    model_name="gpt-4o-mini",
    tools=[write_and_execute_code(globals_dict=globals())],
    chat_memory=memory
)

# Bot remembers previous interactions
tool_calls1 = bot("Create a DataFrame with sample data")
tool_calls2 = bot("Now analyze the data you just created")
```

## Differences from AgentBot

- **ToolBot**: Single-turn execution, returns tool calls for you to execute
- **AgentBot**: Multi-turn planning, executes tools automatically in a graph-based flow

## Related Classes

- **SimpleBot**: Base class that ToolBot extends
- **AgentBot**: Multi-turn agent with automatic tool execution
- **Tools Module**: Tool decorator and utilities

## See Also

- [ToolBot Tutorial](../tutorials/toolbot.md)
- [Which Bot Should I Use?](../getting-started/which-bot.md)
- [Tools Component](../reference/components/tools.md)
