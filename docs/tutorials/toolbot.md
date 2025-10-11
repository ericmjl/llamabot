# ToolBot Tutorial

Welcome to the ToolBot tutorial! In this tutorial, we will learn how to use the `ToolBot` class to create a single-turn bot that can execute tools and functions. ToolBot is designed to analyze user requests and determine the most appropriate tool to execute, making it perfect for automation tasks and function calling scenarios.

## What is ToolBot?

ToolBot is a specialized bot that focuses on **tool selection and execution** rather than multi-turn conversation. It's designed to:

- Analyze user requests to understand what they want to accomplish
- Select the most appropriate tool from its available function toolkit
- Extract or infer the necessary arguments for the selected function
- Return a single function call with the proper arguments to execute

This makes ToolBot ideal for:
- **Automation workflows** where you need to execute specific functions
- **Data analysis tasks** that require custom code execution
- **API integrations** that need to call external services
- **Single-turn function calling** scenarios

## Prerequisites

Before you begin, ensure you have the following:

- Basic knowledge of Python programming
- Familiarity with function calling and tool execution concepts
- Access to a Python environment with the necessary libraries installed

## Installation

First, ensure you have the `llamabot` library installed:

```bash
pip install llamabot
```

## Basic Usage

### Step 1: Import ToolBot

```python
from llamabot import ToolBot
from llamabot.components.tools import write_and_execute_code
```

### Step 2: Create a Simple ToolBot

Let's start with a basic ToolBot that can execute code:

```python
# Create a ToolBot with code execution capabilities
bot = ToolBot(
    system_prompt="You are a helpful assistant that can execute Python code.",
    model_name="gpt-4.1",
    tools=[write_and_execute_code(globals_dict=globals())],
    # Optional: Add chat memory for conversation context
    # chat_memory=lmb.ChatMemory(),
)
```

### Step 3: Use the ToolBot

```python
# Ask the bot to perform a calculation
response = bot("Calculate the sum of numbers from 1 to 100")
print(response)
```

## Advanced Usage with Custom Tools

### Creating Custom Tools

ToolBot works with any function decorated with `@lmb.tool`. Here's how to create custom tools:

```python
import llamabot as lmb

@lmb.tool
def calculate_fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number.

    This function calculates the Fibonacci sequence up to the nth term.
    Use this when you need to compute Fibonacci numbers for mathematical
    calculations or sequence analysis.

    Parameters
    ----------
    n : int
        The position in the Fibonacci sequence (must be non-negative)

    Returns
    -------
    int
        The nth Fibonacci number

    Examples
    --------
    >>> calculate_fibonacci(10)
    55
    """
    if n < 0:
        raise ValueError("Input must be non-negative")
    if n <= 1:
        return n

    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

@lmb.tool
def analyze_dataframe(df_name: str, operation: str) -> str:
    """Analyze a pandas DataFrame with common operations.

    This function performs basic analysis on DataFrames including
    shape, columns, data types, and summary statistics.

    Parameters
    ----------
    df_name : str
        The name of the DataFrame variable in the global scope
    operation : str
        The analysis operation to perform ('shape', 'columns', 'dtypes', 'describe')

    Returns
    -------
    str
        The analysis results as a formatted string
    """
    import pandas as pd

    # Get the DataFrame from globals
    if df_name not in globals():
        return f"DataFrame '{df_name}' not found in global scope"

    df = globals()[df_name]

    if operation == "shape":
        return f"DataFrame shape: {df.shape}"
    elif operation == "columns":
        return f"Columns: {list(df.columns)}"
    elif operation == "dtypes":
        return f"Data types:\n{df.dtypes}"
    elif operation == "describe":
        return f"Summary statistics:\n{df.describe()}"
    else:
        return f"Unknown operation: {operation}"
```

### Using Custom Tools with ToolBot

```python
# Create a ToolBot with custom tools
bot = ToolBot(
    system_prompt="You are a data analysis assistant that can perform calculations and analyze DataFrames.",
    model_name="gpt-4.1",
    tools=[
        write_and_execute_code(globals_dict=globals()),
        calculate_fibonacci,
        analyze_dataframe
    ],
)

# Create some sample data
import pandas as pd
sample_df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': ['a', 'b', 'c', 'd', 'e'],
    'C': [1.1, 2.2, 3.3, 4.4, 5.5]
})

# Use the bot to analyze the data
response = bot("Analyze the shape of sample_df")
print(response)

# Use the bot to calculate Fibonacci numbers
response = bot("Calculate the 15th Fibonacci number")
print(response)
```

## Working with Global Variables

One of ToolBot's key features is its ability to access and work with global variables in your Python session. This is particularly useful for data analysis workflows:

```python
# Create some global variables
import pandas as pd
import numpy as np

# Sample data
sales_data = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=100),
    'sales': np.random.randint(100, 1000, 100),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
})

# Create a ToolBot that can access these variables
bot = ToolBot(
    system_prompt="You are a data analyst. You have access to sales_data DataFrame.",
    model_name="gpt-4.1",
    tools=[write_and_execute_code(globals_dict=globals())],
)

# Ask the bot to analyze the sales data
response = bot("Calculate the total sales and average sales by region")
print(response)
```

## Using Chat Memory with ToolBot

ToolBot supports chat memory for maintaining conversation context across multiple interactions:

```python
import llamabot as lmb

# Create a ToolBot with chat memory
bot = ToolBot(
    system_prompt="You are a data analysis assistant.",
    model_name="gpt-4.1",
    tools=[write_and_execute_code(globals_dict=globals())],
    chat_memory=lmb.ChatMemory(),  # Enable conversation memory
)

# The bot will remember previous interactions
response1 = bot("Create a DataFrame with sample data")
response2 = bot("Now analyze the data you just created")  # Bot remembers the DataFrame
```

**Memory Options:**
- `lmb.ChatMemory()` - Linear memory (fast, no LLM calls)
- `lmb.ChatMemory.threaded(model="gpt-4o-mini")` - Intelligent threading (uses LLM for smart connections)

## ToolBot vs Other Bots

### ToolBot vs SimpleBot

- **SimpleBot**: Focuses on conversation and text generation
- **ToolBot**: Focuses on tool execution and function calling

### ToolBot vs AgentBot

- **AgentBot**: Multi-turn planning and execution with multiple tools
- **ToolBot**: Single-turn tool selection and execution

### ToolBot vs QueryBot

- **QueryBot**: Document retrieval and question answering
- **ToolBot**: Function execution and automation

## Best Practices

### 1. Tool Documentation

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

    Examples
    --------
    >>> my_tool("example", 42)
    "expected output"
    """
    # Tool implementation
    pass
```

### 2. Global Variable Management

- Always pass `globals_dict=globals()` when using `write_and_execute_code`
- Keep your global namespace clean and well-organized
- Use descriptive variable names

### 3. Error Handling

ToolBot will handle tool execution errors gracefully, but it's good practice to include error handling in your custom tools:

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

### 4. System Prompt Design

Design your system prompt to be specific about the tools available:

```python
system_prompt = """
You are a data analysis assistant with access to the following tools:
- write_and_execute_code: Execute Python code with access to global variables
- calculate_fibonacci: Calculate Fibonacci numbers
- analyze_dataframe: Analyze pandas DataFrames

Use the most appropriate tool for each request.
"""
```

## Real-World Example: Data Analysis Workflow

Here's a complete example of using ToolBot for a data analysis workflow:

```python
import llamabot as lmb
import pandas as pd
import numpy as np
from llamabot.components.tools import write_and_execute_code

# Create sample data
customer_data = pd.DataFrame({
    'customer_id': range(1, 101),
    'age': np.random.randint(18, 80, 100),
    'income': np.random.randint(20000, 150000, 100),
    'purchase_amount': np.random.randint(10, 1000, 100),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
})

# Create a ToolBot for data analysis
bot = ToolBot(
    system_prompt="""
    You are a data analyst assistant. You have access to customer_data DataFrame
    and can execute Python code to perform analysis. Focus on providing insights
    about customer demographics, purchasing patterns, and regional differences.
    """,
    model_name="gpt-4.1",
    tools=[write_and_execute_code(globals_dict=globals())],
)

# Perform various analyses
analyses = [
    "Calculate the average age and income by region",
    "Find the top 10 customers by purchase amount",
    "Create a correlation matrix between age, income, and purchase amount",
    "Generate summary statistics for all numeric columns"
]

for analysis in analyses:
    print(f"\n--- {analysis} ---")
    response = bot(analysis)
    print(response)
```

## Troubleshooting

### Common Issues

1. **Tool not found**: Ensure your tool is properly decorated with `@lmb.tool`
2. **Global variables not accessible**: Always pass `globals_dict=globals()` to `write_and_execute_code`
3. **Import errors**: Make sure all required libraries are installed
4. **Type errors**: Ensure your tool parameters match the expected types

### Debug Mode

Enable debug mode to see detailed information about tool execution:

```python
import llamabot as lmb
lmb.set_debug_mode(True)

# Your ToolBot code here
```

## Conclusion

ToolBot provides a powerful way to create single-turn bots that can execute tools and functions. It's particularly useful for:

- **Data analysis workflows** where you need to execute custom code
- **Automation tasks** that require specific function calls
- **API integrations** that need to call external services
- **Single-turn function calling** scenarios

By following the best practices outlined in this tutorial, you can create robust and efficient ToolBot instances that handle complex automation tasks with ease.

For more advanced usage patterns and examples, check out the other bot tutorials in the LlamaBot documentation.
