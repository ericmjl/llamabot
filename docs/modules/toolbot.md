# ToolBot Module

The ToolBot module provides a specialized bot class designed for single-turn tool execution and function calling. Unlike other bots that focus on conversation, ToolBot is optimized for analyzing user requests and selecting the most appropriate tool to execute.

## Overview

ToolBot inherits from `SimpleBot` and extends it with tool execution capabilities. It's designed to:

- Analyze user requests to understand what they want to accomplish
- Select the most appropriate tool from its available function toolkit
- Extract or infer the necessary arguments for the selected function
- Return a single function call with the proper arguments to execute

## Key Features

- **Single-turn execution**: Focuses on one tool call per request
- **Global variable access**: Can access variables in the current Python session
- **Tool introspection**: Automatically reads tool docstrings for better understanding
- **Chat memory integration**: Supports conversation memory for context
- **Streaming support**: Can stream responses to various targets

## Class Reference

### ToolBot

```python
class ToolBot(SimpleBot):
    """A single-turn bot that can execute tools.

    This bot is designed to analyze user requests and determine the most appropriate
    tool or function to execute. It's a generalization of other bot types, focusing
    on tool selection and execution rather than multi-turn conversation.
    """
```

#### Constructor

```python
def __init__(
    self,
    system_prompt: str,
    model_name: str,
    tools: Optional[List[Callable]] = None,
    chat_memory: Optional[ChatMemory] = None,
    **completion_kwargs,
):
```

**Parameters:**

- `system_prompt` (str): The system prompt to use for the bot
- `model_name` (str): The name of the language model to use
- `tools` (Optional[List[Callable]]): Optional list of additional tools to include
- `chat_memory` (Optional[ChatMemory]): Chat memory component for context retrieval
- `**completion_kwargs`: Additional keyword arguments for completion

#### Methods

##### `__call__(message)`

Process a message and return tool calls.

**Parameters:**
- `message`: The message to process

**Returns:**
- List of tool calls to execute

## Available Tools

ToolBot comes with several built-in tools:

### Core Tools

- `today_date()`: Returns the current date in YYYY-MM-DD format
- `respond_to_user(response: str)`: Responds to the user with a text message
- `return_object_to_user(variable_name: str)`: Returns an object from the calling context's globals dictionary. Use this to return actual Python objects (DataFrames, lists, dicts, etc.) instead of text responses.

### Code Execution Tools

- `write_and_execute_code(globals_dict: dict)`: Creates a tool for executing Python code with access to global variables
- `write_and_execute_script(code: str, dependencies_str: str, python_version: str)`: Executes Python scripts in a secure sandbox

### Web Tools

- `search_internet(search_term: str, max_results: int)`: Searches the internet for a given term
- `scrape_webpage(url: str)`: Scrapes content from a webpage

## Usage Examples

### Basic Usage

```python
from llamabot import ToolBot
from llamabot.components.tools import write_and_execute_code

# Create a ToolBot with code execution capabilities
bot = ToolBot(
    system_prompt="You are a helpful assistant that can execute Python code.",
    model_name="gpt-4.1",
    tools=[write_and_execute_code(globals_dict=globals())],
)

# Use the bot
response = bot("Calculate the sum of numbers from 1 to 100")
```

### With Custom Tools

```python
import llamabot as lmb

@lmb.tool
def calculate_fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

bot = ToolBot(
    system_prompt="You are a mathematical assistant.",
    model_name="gpt-4.1",
    tools=[calculate_fibonacci],
)

response = bot("Calculate the 10th Fibonacci number")
```

### With Global Variables

```python
import pandas as pd
import numpy as np

# Create some data
data = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100)
})

bot = ToolBot(
    system_prompt="You are a data analyst.",
    model_name="gpt-4.1",
    tools=[write_and_execute_code(globals_dict=globals())],
)

response = bot("Calculate the correlation between x and y in the data DataFrame")
```

## System Prompt Function

### toolbot_sysprompt

```python
@prompt("system")
def toolbot_sysprompt(globals_dict: dict = {}) -> str:
    """Generate a system prompt for ToolBot with global variable context."""
```

This function generates a comprehensive system prompt that includes information about available global variables in the current session.

**Parameters:**
- `globals_dict` (dict): Dictionary of global variables to include in the prompt

**Returns:**
- str: A formatted system prompt with global variable information

## Integration with Other Components

### Chat Memory

ToolBot integrates with the `ChatMemory` component to maintain conversation context:

```python
from llamabot.components.chat_memory import ChatMemory

memory = ChatMemory()
bot = ToolBot(
    system_prompt="You are a helpful assistant.",
    model_name="gpt-4.1",
    chat_memory=memory,
)
```

### Tool System

ToolBot works with the tool system to provide function calling capabilities:

```python
from llamabot.components.tools import tool

@tool
def my_custom_tool(param: str) -> str:
    """My custom tool."""
    return f"Processed: {param}"

bot = ToolBot(
    system_prompt="You are a helpful assistant.",
    model_name="gpt-4.1",
    tools=[my_custom_tool],
)
```

## Best Practices

### Tool Documentation

Always provide comprehensive docstrings for your tools:

```python
@lmb.tool
def my_tool(param1: str, param2: int) -> str:
    """Clear description of what the tool does.

    Detailed explanation of when and how to use this tool.
    Include examples and edge cases.

    Parameters
    ----------
    param1 : str
        Description of the first parameter
    param2 : int
        Description of the second parameter

    Returns
    -------
    str
        Description of what the tool returns
    """
    # Tool implementation
    pass
```

### Global Variable Management

- Always pass `globals_dict=globals()` when using `write_and_execute_code`
- Keep your global namespace clean and well-organized
- Use descriptive variable names

### Error Handling

Include proper error handling in your custom tools:

```python
@lmb.tool
def robust_tool(input_data: str) -> str:
    """A tool with proper error handling."""
    try:
        # Tool logic here
        result = process_data(input_data)
        return result
    except Exception as e:
        return f"Error processing data: {str(e)}"
```

## Limitations

- **Single-turn only**: ToolBot is designed for single tool calls, not multi-step planning
- **Tool dependency**: Requires tools to be properly decorated with `@lmb.tool`
- **Global scope**: Code execution tools need access to the global namespace
- **Model dependency**: Requires a language model that supports function calling

## Related Components

- [SimpleBot](simplebot.md): Base bot class that ToolBot inherits from
- [ChatMemory](chat_memory.md): Conversation memory component
- [Tools](tools.md): Tool system for function calling
- [AgentBot](agentbot.md): Multi-turn planning bot for comparison
