# AgentBot Tutorial

Welcome to the AgentBot tutorial! In this tutorial, we will guide you
through the process of building an agent that uses PocketFlow for
graph-based tool orchestration. AgentBot automatically wraps your
functions as tools and uses a decision node to orchestrate tool
execution through a flow graph.

## Prerequisites

Before you begin, ensure you have the following:

- Basic knowledge of Python programming
- Familiarity with the concept of bots and automation
- Access to a Python environment with the necessary libraries installed

## What is AgentBot?

AgentBot is a graph-based agent that uses PocketFlow to orchestrate tool
execution. Unlike traditional agents that use loops, AgentBot builds a
flow graph where:

1. A **decision node** (DecideNode) analyzes the conversation and
   selects which tool to execute
2. **Tool nodes** execute the selected tools
3. Tools can **loop back** to the decision node (except terminal tools
   like `respond_to_user`)
4. The flow continues until a terminal node is reached

This graph-based approach provides:

- **Visual flow representation**: You can visualize the agent's flow
  graph
- **Flexible orchestration**: Tools are connected in a graph, not a
  linear sequence
- **Automatic tool wrapping**: You provide plain callables; AgentBot
  handles the rest
- **Default tools**: `today_date`, `respond_to_user`, and `return_object_to_user` are always
  available

## Part 1: Basic Usage

### Step 1: Setting Up the Environment

First, ensure you have the `llamabot` library installed:

```bash
pip install llamabot
```

### Step 2: Creating a Simple AgentBot

The simplest way to create an AgentBot is to provide a list of callable functions:

```python
import llamabot as lmb

def get_weather(city: str) -> str:
    """Get the current weather for a city.

    :param city: The name of the city
    :return: Weather information
    """
    # In practice, you'd call a real weather API
    return f"The weather in {city} is sunny, 72°F"

# Create an AgentBot with your function
agent = lmb.AgentBot(
    tools=[get_weather],
    model_name="gpt-4o-mini"
)

# Use the agent
result = agent("What's the weather in New York?")
print(result)
```

**What happens behind the scenes:**

1. AgentBot automatically wraps `get_weather` with `@tool` and
   `@nodeify` decorators
2. Default tools (`today_date`, `respond_to_user`, and `return_object_to_user`) are added
   automatically
3. A `DecideNode` is created to decide which tool to use
4. The flow graph is built connecting the decision node to all tools
5. When you call the agent, it runs the flow with your query

### Step 3: Understanding Default Tools

AgentBot always includes three default tools:

- **`today_date`**: Returns the current date (loops back to decide
  node)
- **`respond_to_user`**: Sends a text response to the user (terminal node,
  no loopback)
- **`return_object_to_user`**: Returns an object from the calling context's globals
  (terminal node, no loopback). Use this when you want to return actual Python
  objects like DataFrames, lists, or dictionaries.

These tools are automatically available, so you don't need to provide them:

```python
agent = lmb.AgentBot(tools=[])

# The agent can still use today_date, respond_to_user, and return_object_to_user
result = agent("What's today's date?")
```

**When to use `respond_to_user` vs `return_object_to_user`:**
- Use `respond_to_user` for text responses, explanations, or conversational replies
- Use `return_object_to_user` when you want to return actual Python objects (DataFrames,
  lists, dicts, etc.) from your notebook's or script's globals

**Using `return_object_to_user` with globals:**

To enable `return_object_to_user` to access variables from your calling context, pass
`globals_dict` when calling the agent:

```python
import pandas as pd

# Create some data in your notebook/script
df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

agent = lmb.AgentBot(tools=[])

# Pass globals() so the agent can access 'df'
result = agent("Return the dataframe", globals_dict=globals())

# result will be the DataFrame object
print(result)
```

The agent can now use `return_object_to_user` to return objects from your globals dictionary.

**Fuzzy Variable Name Matching:**

The agent can intelligently match partial variable names. For example, if you have a variable
named `ic50_data_with_confounders` in your globals, you can ask for it using a shorter name:

```python
# Variable in globals
ic50_data_with_confounders = pd.read_csv("data.csv")

agent = lmb.AgentBot(tools=[])

# You can use a partial name - the agent will match it intelligently
result = agent("show me ic50", globals_dict=globals())

# The agent will match "ic50" to "ic50_data_with_confounders" and return it
```

The agent sees all available variables in globals and can match partial names based on context
and similarity, making it easier to access your data without typing full variable names.

## Part 2: Building a Financial Analysis Agent

Let's create a more sophisticated agent that can analyze financial data
using multiple tools.

### Step 1: Defining Custom Tools

You can define tools as plain Python functions - no decorators needed:

```python
def get_stock_price(symbol: str) -> float:
    """Get the current stock price for a given symbol.

    :param symbol: The stock symbol (e.g., 'AAPL', 'MSFT', 'GOOGL')
    :return: The current stock price
    """
    # This is a simplified example - in practice, you'd use a real API
    mock_prices = {
        'AAPL': 150.25,
        'MSFT': 300.50,
        'GOOGL': 2800.75,
        'TSLA': 200.30
    }

    if symbol.upper() not in mock_prices:
        raise ValueError(f"Symbol {symbol} not found")

    return mock_prices[symbol.upper()]

def calculate_percentage_change(old_price: float, new_price: float) -> float:
    """Calculate the percentage change between two prices.

    :param old_price: The original price
    :param new_price: The new price
    :return: The percentage change (positive for increase, negative for decrease)
    """
    return ((new_price - old_price) / old_price) * 100

def analyze_portfolio(prices: list[float]) -> dict:
    """Analyze a portfolio of stock prices.

    :param prices: List of stock prices
    :return: Analysis results including average, min, max, and trend
    """
    if not prices:
        return {"error": "No prices provided"}

    avg_price = sum(prices) / len(prices)
    min_price = min(prices)
    max_price = max(prices)

    # Simple trend analysis
    if len(prices) >= 2:
        trend = "upward" if prices[-1] > prices[0] else "downward"
    else:
        trend = "insufficient data"

    return {
        "average": avg_price,
        "minimum": min_price,
        "maximum": max_price,
        "trend": trend,
        "count": len(prices)
    }
```

### Step 2: Creating the Financial Agent

```python
# Create a financial analysis agent
financial_agent = lmb.AgentBot(
    tools=[get_stock_price, calculate_percentage_change, analyze_portfolio],
    model_name="gpt-4o-mini"
)
```

### Step 3: Using the Agent for Analysis

```python
# Ask the agent to analyze multiple stocks
result = financial_agent("""
Please analyze the following stocks:
1. Get the current price of AAPL
2. Get the current price of MSFT
3. Calculate the percentage change from yesterday's prices (AAPL: $145, MSFT: $295)
4. Analyze the portfolio performance
""")

print(result)
```

The agent will:

1. Use the decision node to select `get_stock_price` for AAPL
2. Loop back to decide, then select `get_stock_price` for MSFT
3. Loop back to decide, then select `calculate_percentage_change` for
   both stocks
4. Loop back to decide, then select `analyze_portfolio`
5. Finally, use `respond_to_user` (terminal) to provide the final
   answer

## Part 3: Visualizing the Flow Graph

One of the powerful features of AgentBot is the ability to visualize
the flow graph. If you're using Marimo notebooks, you can display the
graph:

```python
agent = lmb.AgentBot(tools=[get_stock_price, calculate_percentage_change])

# In a Marimo notebook, this will display the flow graph
agent
```

The graph shows:

- The decision node (DecideNode)
- All tool nodes (with their function names)
- Edges showing how tools connect back to the decision node
- Terminal nodes (like `respond_to_user`) that don't loop back

### Visualization Features

The flow visualization includes several automatic features:

**Automatic Graph Direction**: The visualization automatically determines
whether to use a top-down (`graph TD`) or left-right (`graph LR`) layout
based on the graph structure. If the graph is wider than it is deep, it
uses a left-right layout; otherwise, it uses a top-down layout.

**Terminal Node Coloring**: Terminal nodes (nodes with no successors) are
automatically colored green to distinguish them from regular nodes, which
are colored blue. This makes it easy to identify where the flow ends.

You can also manually generate the Mermaid diagram string:

```python
from llamabot.components.pocketflow import flow_to_mermaid

agent = lmb.AgentBot(tools=[get_stock_price, calculate_percentage_change])

# Get the Mermaid diagram string
mermaid_diagram = flow_to_mermaid(agent.flow)
print(mermaid_diagram)
```

## Part 4: Custom Decision Nodes

By default, AgentBot uses `DecideNode` which uses ToolBot to decide
which tool to execute. You can provide a custom decision node:

```python
from llamabot.components.pocketflow import DecideNode

# Create a custom decision node with a specific model
custom_decide = DecideNode(
    tools=[],  # Will be set by AgentBot
    model_name="gpt-4.1"
)

agent = lmb.AgentBot(
    tools=[get_stock_price],
    decide_node=custom_decide
)
```

## Part 5: Advanced Features

### Terminal Tools

By default, all tools loop back to the decision node. However,
`respond_to_user` is a terminal tool that ends the flow. You can
create your own terminal tools using the `nodeify` decorator:

```python
from llamabot.components.pocketflow import nodeify
from llamabot.components.tools import tool

# Terminal node - no loopback
@nodeify(loopback_name=None)
@tool
def final_answer(message: str) -> str:
    """Provide the final answer to the user.

    :param message: The final answer message
    :return: The message
    """
    return message
```

### Using Already-Decorated Tools

If you have functions that are already decorated with `@tool`, that's
fine too:

```python
@lmb.tool
def my_tool(arg: str) -> str:
    """My tool function."""
    return f"Result: {arg}"

# AgentBot will still wrap it with @nodeify
agent = lmb.AgentBot(tools=[my_tool])
```

### Accessing the Flow

You can access the underlying PocketFlow flow for advanced use cases:

```python
agent = lmb.AgentBot(tools=[get_stock_price])

# Access the flow
flow = agent.flow

# Access individual nodes
decide_node = agent.decide_node
tools = agent.tools
```

## Part 6: Best Practices

### 1. Write Clear Function Documentation

Good docstrings help the decision node choose the right tool:

```python
def analyze_sentiment(text: str) -> dict:
    """Analyze the sentiment of text using a simple algorithm.

    This tool performs basic sentiment analysis by counting positive and
    negative words. It's useful for getting a quick understanding of
    text sentiment.

    :param text: The text to analyze
    :return: Dictionary with 'sentiment' (positive/negative/neutral),
        'score' (0-1), and 'confidence'
    """
    # Implementation here
    pass
```

### 2. Design Tools for Specific Use Cases

Keep tools focused and single-purpose:

```python
def get_user_profile(user_id: str) -> dict:
    """Get user profile information.

    :param user_id: The user's unique identifier
    :return: User profile dictionary
    """
    # Implementation
    pass

def update_user_profile(user_id: str, updates: dict) -> dict:
    """Update user profile information.

    :param user_id: The user's unique identifier
    :param updates: Dictionary of fields to update
    :return: Updated user profile
    """
    # Implementation
    pass
```

### 3. Handle Errors Gracefully

Tools should handle errors and return meaningful results:

```python
def safe_api_call(url: str, timeout: int = 10) -> dict:
    """Safely make an API call with error handling.

    :param url: The URL to call
    :param timeout: Request timeout in seconds, by default 10
    :return: API response or error information
    """
    import requests

    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return {
            "status": "success",
            "data": response.json(),
            "status_code": response.status_code
        }
    except requests.RequestException as e:
        return {
            "status": "error",
            "error": str(e),
            "url": url
        }
```

## Part 7: Understanding the Flow

### How Tools Are Wrapped

When you provide a function to AgentBot:

1. The function is wrapped with `@tool` to create a tool schema
2. The tool is wrapped with `@nodeify` to create a PocketFlow node
3. The node is connected to the decision node
4. If not terminal, the node loops back to the decision node

### Flow Execution

When you call the agent:

1. Your query is added to the shared state's memory
2. The flow starts at the decision node
3. The decision node uses ToolBot to select a tool
4. The selected tool executes with arguments from the decision node
5. The result is added to memory
6. If the tool loops back, the flow returns to the decision node
7. This continues until a terminal tool (like `respond_to_user`) is
   reached

### Shared State

The flow uses a shared state dictionary that contains:

- `memory`: List of conversation messages and tool results
- `func_call`: Dictionary of function arguments (set by decision node)
- `result`: Tool execution results

## Conclusion

Congratulations! You now understand how to use AgentBot with PocketFlow
for graph-based tool orchestration. AgentBot provides:

- **Automatic tool wrapping**: Just provide plain callables
- **Graph-based orchestration**: Visual flow representation
- **Default tools**: `today_date`, `respond_to_user`, and `return_object_to_user` always
  available
- **Flexible decision making**: Custom decision nodes supported
- **Terminal nodes**: Control flow termination with terminal tools

The graph-based approach makes it easy to visualize and understand how
your agent works, while the automatic wrapping makes it simple to add
new tools. Happy coding!
