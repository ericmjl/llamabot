# AgentBot Tutorial

Welcome to the AgentBot tutorial! In this tutorial, we will guide you through the process of building a simple bot that can calculate tips and split bills, and then extend it to analyze stock price trends using various tools. By the end of this tutorial, you will have a solid understanding of how to use the AgentBot to automate tasks using a sequence of tools.

## Prerequisites

Before you begin, ensure you have the following:

- Basic knowledge of Python programming.
- Familiarity with the concept of bots and automation.
- Access to a Python environment with the necessary libraries installed.

## Part 1: Building a Restaurant Bill Bot

In this section, we will create a bot that can calculate the total bill amount including a tip and split the bill among multiple people.

### Enhanced Tool Documentation

LlamaBot now supports comprehensive docstring parsing for tools, automatically extracting rich contextual information that helps LLMs understand when and how to use your tools. This means you can write self-documenting tools that don't require additional system prompt instructions.

**Supported Docstring Styles:**
- **NumPy style**: Traditional scientific Python format
- **Google style**: Clean, readable format popular in Google's Python codebase
- **Sphinx style**: reStructuredText format used in Sphinx documentation

**Key Features:**
- **Automatic parameter extraction**: Type hints and descriptions are parsed from docstrings
- **Rich descriptions**: Both short and long descriptions are combined for comprehensive context
- **Optional parameters**: Tools can have default values and optional parameters
- **Self-documenting**: Tools contain all the context an LLM needs to use them effectively

### Step 1: Setting Up the Environment

First, ensure you have the `llamabot` library installed. You can install it using pip:

```bash
pip install llamabot
```

### Step 2: Creating the Bot

We will use the `AgentBot` class to create our bot. The bot will use two tools: `calculate_total_with_tip` and `split_bill`.

```python
import llamabot as lmb

# Define the tools with comprehensive documentation
@lmb.tool
def calculate_total_with_tip(bill_amount: float, tip_rate: float) -> float:
    """Calculate the total bill amount including tip.

    This function takes a base bill amount and applies a tip percentage to calculate
    the final total. The tip rate should be provided as a decimal (e.g., 0.18 for 18%).

    Parameters
    ----------
    bill_amount : float
        The base amount of the bill before tip
    tip_rate : float
        The tip rate as a decimal (e.g., 0.18 for 18% tip)

    Returns
    -------
    float
        The total amount including tip

    Raises
    ------
    ValueError
        If tip_rate is negative or greater than 1.0

    Examples
    --------
    >>> calculate_total_with_tip(100.0, 0.18)
    118.0
    """
    if tip_rate < 0 or tip_rate > 1.0:
        raise ValueError("Tip rate must be between 0 and 1.0")
    return bill_amount * (1 + tip_rate)

@lmb.tool
def split_bill(total_amount: float, num_people: int) -> float:
    """Split a bill equally among a specified number of people.

    This function divides a total bill amount equally among the specified number
    of people. Use this when you need to calculate how much each person should pay
    for a shared expense.

    Parameters
    ----------
    total_amount : float
        The total bill amount to split
    num_people : int
        The number of people to split the bill among

    Returns
    -------
    float
        The amount each person should pay

    Raises
    ------
    ValueError
        If num_people is less than or equal to 0

    Examples
    --------
    >>> split_bill(120.0, 4)
    30.0
    """
    if num_people <= 0:
        raise ValueError("Number of people must be positive")
    return total_amount / num_people

# Create the bot
bot = lmb.AgentBot(
    system_prompt=lmb.system("You are my assistant with respect to restaurant bills."),
    tools=[calculate_total_with_tip, split_bill],  # Note: use 'tools', not 'functions'
    model_name="gpt-4.1",
)
```

**Key Parameters Explained:**

- `tools`: List of callable functions that the bot can use (required)
- `system_prompt`: Instructions for the bot's behavior
- `model_name`: The language model to use
- `temperature`: Controls randomness (default: 0.0 for deterministic responses)
- `stream_target`: Where to stream responses ("none", "stdout", "panel", "api")

### Benefits of Self-Documenting Tools

With comprehensive docstring parsing, your tools become self-documenting:

1. **No Additional Context Needed**: The LLM automatically understands when to use each tool based on the rich descriptions
2. **Automatic Parameter Validation**: Type hints and descriptions help the LLM provide correct arguments
3. **Clear Usage Examples**: Examples in docstrings show the LLM exactly how to use the tool
4. **Error Handling**: The LLM understands what errors to expect and how to handle them

### Best Practices for Tool Documentation

When writing tools for LLM agents, follow these best practices:

1. **Use Descriptive Names**: Function names should clearly indicate their purpose
2. **Comprehensive Docstrings**: Include both short and long descriptions
3. **Parameter Documentation**: Document each parameter with its type, purpose, and constraints
4. **Return Value Documentation**: Explain what the function returns and in what format
5. **Error Handling**: Document what errors can occur and under what conditions
6. **Usage Examples**: Provide clear examples of how to use the function
7. **Type Hints**: Always include type hints for better LLM understanding

### Step 3: Using the Bot

Now, let's use the bot to calculate the total bill with a tip and split it among people.

```python
# Calculate total with tip
calculate_total_only_prompt = "My dinner was $2300 without tips. Calculate my total with an 18% tip."
response = bot(calculate_total_only_prompt)
print(response.content)

# Split the bill
split_bill_only_prompt = "My dinner was $2300 in total. Split the bill between 4 people."
response = bot(split_bill_only_prompt)
print(response.content)
```

### Step 4: Combining Both Actions

You can also combine both actions in a single prompt:

```python
split_and_calculate_prompt = "My dinner was $2300 without tips. Calculate my total with an 18% tip and split the bill between 4 people."
response = bot(split_and_calculate_prompt)
print(response.content)
```

### Step 5: Understanding Bot Execution

The AgentBot works by:

1. **Planning**: Breaking down your request into steps
2. **Executing**: Calling tools in sequence
3. **Iterating**: Repeating until the task is complete or max iterations reached

You can control the maximum number of iterations:

```python
# Limit to 5 iterations for faster responses
response = bot("Complex task here", max_iterations=5)
```

## Part 2: Building a Stock Analysis Bot

In this section, we will extend the bot to analyze stock price trends using additional tools.

### Docstring Style Examples

LlamaBot supports multiple docstring styles. Here are examples of how to document the same function in different styles:

#### NumPy Style (Scientific Python)
```python
@lmb.tool
def calculate_compound_interest(principal: float, rate: float, time: float) -> float:
    """Calculate compound interest using the standard formula.

    Parameters
    ----------
    principal : float
        The initial amount of money invested
    rate : float
        The annual interest rate as a decimal (e.g., 0.05 for 5%)
    time : float
        The time period in years

    Returns
    -------
    float
        The final amount after compound interest
    """
    return principal * (1 + rate) ** time
```

#### Google Style (Clean and Readable)
```python
@lmb.tool
def calculate_compound_interest(principal: float, rate: float, time: float) -> float:
    """Calculate compound interest using the standard formula.

    Args:
        principal (float): The initial amount of money invested
        rate (float): The annual interest rate as a decimal (e.g., 0.05 for 5%)
        time (float): The time period in years

    Returns:
        float: The final amount after compound interest
    """
    return principal * (1 + rate) ** time
```

#### Sphinx Style (reStructuredText)
```python
@lmb.tool
def calculate_compound_interest(principal: float, rate: float, time: float) -> float:
    """Calculate compound interest using the standard formula.

    :param principal: The initial amount of money invested
    :type principal: float
    :param rate: The annual interest rate as a decimal (e.g., 0.05 for 5%)
    :type rate: float
    :param time: The time period in years
    :type time: float
    :return: The final amount after compound interest
    :rtype: float
    """
    return principal * (1 + rate) ** time
```

All three styles are automatically parsed and provide the same rich context to the LLM. Choose the style that best fits your project's conventions.

### Step 1: Defining New Tools

We will define tools for scraping stock prices, calculating moving averages, and determining trends.

```python
import numpy as np
import httpx
from typing import List

@lmb.tool
def scrape_stock_prices(symbol: str) -> List[float]:
    """Fetch historical stock prices for a given symbol.

    Retrieves the last 100 days of daily closing prices for a stock symbol
    from Yahoo Finance. This data can be used for technical analysis and
    trend calculations.

    Parameters
    ----------
    symbol : str
        The stock symbol to fetch data for (e.g., 'AAPL', 'MSFT', 'GOOGL')

    Returns
    -------
    List[float]
        List of closing prices for the last 100 trading days

    Raises
    ------
    httpx.RequestError
        If the request to Yahoo Finance fails
    KeyError
        If the response format is unexpected

    Examples
    --------
    >>> prices = scrape_stock_prices('AAPL')
    >>> len(prices)
    100
    >>> all(isinstance(p, float) for p in prices)
    True
    """
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"range": "100d", "interval": "1d"}
    with httpx.Client() as client:
        response = client.get(url, params=params)
        data = response.json()
        prices = data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
        return [float(price) for price in prices if price is not None]

@lmb.tool
def calculate_moving_average(data: List[float], window: int = 20) -> List[float]:
    """Calculate the moving average of a time series.

    Computes a simple moving average over a specified window size. The moving
    average is useful for smoothing out price fluctuations and identifying trends.
    Values before the window size are filled with NaN.

    Parameters
    ----------
    data : List[float]
        The time series data to calculate moving average for
    window : int, optional
        The window size for the moving average, by default 20

    Returns
    -------
    List[float]
        List of moving average values (same length as input data)

    Raises
    ------
    ValueError
        If window size is larger than the data length

    Examples
    --------
    >>> data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> ma = calculate_moving_average(data, window=3)
    >>> ma[2:5]  # First valid moving average values
    [2.0, 3.0, 4.0]
    """
    if window > len(data):
        raise ValueError("Window size cannot be larger than data length")
    ma = np.full(len(data), np.nan)
    for i in range(window - 1, len(data)):
        ma[i] = np.mean(data[i - window + 1 : i + 1])
    return ma.tolist()

@lmb.tool
def calculate_slope(data: List[float], days: int = None) -> float:
    """Calculate the slope of a linear trend in time series data.

    Fits a linear regression to the time series data and returns the slope,
    which indicates the direction and strength of the trend. A positive slope
    indicates an upward trend, while a negative slope indicates a downward trend.

    Parameters
    ----------
    data : List[float]
        The time series data to analyze
    days : int, optional
        Number of most recent days to analyze. If None, uses all data.

    Returns
    -------
    float
        The slope of the linear trend (positive = upward, negative = downward)

    Raises
    ------
    ValueError
        If the requested number of days exceeds the data length

    Examples
    --------
    >>> data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> slope = calculate_slope(data)
    >>> slope > 0  # Should be positive for upward trend
    True
    """
    if days is not None and days > len(data):
        raise ValueError("Requested days exceeds data length")
    if days:
        data = data[-days:]
    x = np.arange(len(data))
    slope, _ = np.polyfit(x, data, 1)
    return float(slope)
```

### Step 2: Creating the Stock Analysis Bot

We will create a new `AgentBot` for stock analysis.

```python
from llamabot.bot.agentbot import AgentBot

stats_bot = AgentBot(
    system_prompt=lmb.system("You are a stock market analysis assistant."),
    tools=[scrape_stock_prices, calculate_moving_average, calculate_slope],
    model_name="gpt-4.1",
)
```

### Step 3: Analyzing Stock Data

Use the bot to analyze stock data for MRNA and AAPL.

```python
response = stats_bot(
    """Please analyze the last 100 days of MRNA and AAPL stock prices. For each stock:
    1. Calculate the 20-day moving average
    2. Calculate the slope over the full 100 day period
    3. Tell me if the stock is trending upward or downward based on both metrics
    4. Compare the trends between MRNA and AAPL
    """
)
print(response.content)
```

## Part 3: Using the Planner Bot

The AgentBot includes a powerful planning feature that helps break down complex tasks into manageable steps. The planner bot analyzes the current state, available tools, and task requirements to create an efficient execution plan.

### Step 1: Creating a Bot with a Planner

To use the planner bot, you can create an AgentBot with a planner:

```python
from llamabot.bot.agentbot import AgentBot, planner_bot

# Create a planner bot
planner = planner_bot(model_name="gpt-4.1")

# Create the main bot with the planner
bot_with_planner = AgentBot(
    system_prompt=lmb.system("You are a task automation assistant."),
    tools=[calculate_total_with_tip, split_bill],
    planner_bot=planner,
    model_name="gpt-4.1",
)
```

### Step 2: Using the Planner for Complex Tasks

The planner bot is particularly useful for complex tasks that require multiple steps or careful sequencing:

```python
complex_prompt = """
I need to:
1. Calculate the total bill for a $150 dinner with 20% tip
2. Split it between 3 people
3. Calculate how much each person should pay if one person is paying for drinks ($30)
"""

response = bot_with_planner(complex_prompt)
print(response.content)
```

The planner bot will:
1. Analyze the task requirements
2. Break down the steps into a logical sequence
3. Consider potential edge cases
4. Create an efficient plan using the available tools
5. Execute the plan step by step

### Step 3: Monitoring Planning Metrics

The AgentBot tracks planning metrics that you can use to analyze the bot's performance:

```python
# Access planning metrics
planning_metrics = bot_with_planner.run_meta["planning_metrics"]
print(f"Plan generated: {planning_metrics['plan_generated']}")
print(f"Plan revisions: {planning_metrics['plan_revisions']}")
print(f"Planning time: {planning_metrics['plan_time']} seconds")
```

## Part 4: Understanding Bot Analytics and Built-in Tools

### Built-in Tools

AgentBot automatically includes two built-in tools:

```python
# today_date - Returns current date
# respond_to_user - Used for final responses to users
```

You don't need to define these - they're automatically available to your bot.

### Run Metadata and Analytics

Every AgentBot execution provides detailed analytics through `run_meta`:

```python
# Access comprehensive execution data
run_data = bot.run_meta

print(f"Execution time: {run_data['duration']} seconds")
print(f"Iterations used: {run_data['current_iteration']}")
print(f"Max iterations: {run_data['max_iteration']}")

# Message counts
message_counts = run_data['message_counts']
print(f"User messages: {message_counts['user']}")
print(f"Assistant messages: {message_counts['assistant']}")
print(f"Tool messages: {message_counts['tool']}")

# Tool usage statistics
tool_usage = run_data['tool_usage']
for tool_name, stats in tool_usage.items():
    print(f"{tool_name}: {stats['calls']} calls, {stats['success']} successes, {stats['failures']} failures")
    print(f"  Total duration: {stats['total_duration']:.2f} seconds")
```

### Error Handling

AgentBot handles errors gracefully:

```python
# If max_iterations is exceeded
try:
    response = bot("Very complex task", max_iterations=3)
except RuntimeError as e:
    print(f"Bot exceeded iteration limit: {e}")

# Tool errors are captured and returned
# The bot will continue with other tools even if one fails
```

## Part 5: Advanced Configuration

### Customizing Bot Behavior

```python
# Create a bot with custom settings
custom_bot = AgentBot(
    system_prompt=lmb.system("You are a helpful assistant."),
    tools=[your_tools],
    temperature=0.1,  # Slightly more creative responses
    stream_target="stdout",  # Stream responses to console
    model_name="gpt-4.1",
)

# Use with custom iteration limit
response = custom_bot("Your request", max_iterations=15)
```

### Streaming Responses

You can stream responses in real-time:

```python
# Stream to console
streaming_bot = AgentBot(
    tools=[your_tools],
    stream_target="stdout",  # Options: "stdout", "panel", "api", "none"
)

# Responses will appear as they're generated
response = streaming_bot("Your request")
```

## Part 6: The Impact on System Prompts

With self-documenting tools, you can significantly simplify your system prompts. Instead of providing detailed instructions about when and how to use each tool, the LLM automatically understands this from the tool documentation.

### Before: Complex System Prompts
```python
# Old approach - required detailed system prompt
bot = lmb.AgentBot(
    system_prompt=lmb.system("""
    You are a financial assistant. You have access to these tools:

    1. calculate_total_with_tip: Use this when users want to add a tip to a bill.
       Takes bill_amount (float) and tip_rate (float, 0-1). Returns total with tip.

    2. split_bill: Use this when users want to split a bill among people.
       Takes total_amount (float) and num_people (int). Returns amount per person.

    3. scrape_stock_prices: Use this to get stock data for analysis.
       Takes symbol (str). Returns list of prices.

    Always use the appropriate tool for each task.
    """),
    tools=[calculate_total_with_tip, split_bill, scrape_stock_prices],
)
```

### After: Simple System Prompts
```python
# New approach - tools are self-documenting
bot = lmb.AgentBot(
    system_prompt=lmb.system("You are a helpful financial assistant."),
    tools=[calculate_total_with_tip, split_bill, scrape_stock_prices],
)
```

The LLM automatically understands:
- When to use each tool based on the comprehensive descriptions
- What parameters each tool expects and their types
- What each tool returns and in what format
- How to handle errors and edge cases
- Usage examples for each tool

This makes your bots more maintainable and reduces the cognitive load of writing complex system prompts.

## Conclusion

Congratulations! You have successfully built and used an AgentBot to automate tasks related to restaurant bills and stock price analysis. You've also learned how to:

- Create bots with custom tools using comprehensive documentation
- Write self-documenting tools that don't require additional system prompt instructions
- Use multiple docstring styles (NumPy, Google, Sphinx) for tool documentation
- Follow best practices for tool documentation that maximize LLM understanding
- Use the planner bot for complex tasks
- Monitor bot performance with analytics
- Handle errors and configure advanced settings
- Stream responses in real-time
- Simplify system prompts through self-documenting tools

The enhanced docstring parsing capabilities make LlamaBot tools more powerful and easier to use. By writing comprehensive, well-documented tools, you create a more intelligent and autonomous agent system that requires less manual instruction and configuration.

You can now explore further by adding more tools and customizing the bot to suit your needs. Happy coding!
