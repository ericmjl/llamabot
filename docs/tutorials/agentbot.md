---
diataxis_type: tutorial
intents:
- Build a diataxis-style tutorial for the AgentBot.
- Show how to use it to build a simple bot that calculates tips and splits bills.
- Then show how to use it to build a bot that uses tools to calculate recent trends
  in stock prices.
linked_files:
- scratch_notebooks/tools.ipynb
- llamabot/bot/agentbot.py
---

# AgentBot Tutorial

Welcome to the AgentBot tutorial! In this tutorial, we will guide you through the process of building a simple bot that can calculate tips and split bills, and then extend it to analyze stock price trends using various tools. By the end of this tutorial, you will have a solid understanding of how to use the AgentBot to automate tasks using a sequence of tools.

## Prerequisites

Before you begin, ensure you have the following:

- Basic knowledge of Python programming.
- Familiarity with the concept of bots and automation.
- Access to a Python environment with the necessary libraries installed.

## Part 1: Building a Restaurant Bill Bot

In this section, we will create a bot that can calculate the total bill amount including a tip and split the bill among multiple people.

### Step 1: Setting Up the Environment

First, ensure you have the `llamabot` library installed. You can install it using pip:

```bash
pip install llamabot
```

### Step 2: Creating the Bot

We will use the `AgentBot` class to create our bot. The bot will use two tools: `calculate_total_with_tip` and `split_bill`.

```python
import llamabot as lmb

# Define the tools
@lmb.tool
def calculate_total_with_tip(bill_amount: float, tip_rate: float) -> float:
    if tip_rate < 0 or tip_rate > 1.0:
        raise ValueError("Tip rate must be between 0 and 1.0")
    return bill_amount * (1 + tip_rate)

@lmb.tool
def split_bill(total_amount: float, num_people: int) -> float:
    return total_amount / num_people

# Create the bot
bot = lmb.AgentBot(
    system_prompt=lmb.system("You are my assistant with respect to restaurant bills."),
    functions=[calculate_total_with_tip, split_bill],
    model_name="gpt-4.1",
)
```

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

## Part 2: Building a Stock Analysis Bot

In this section, we will extend the bot to analyze stock price trends using additional tools.

### Step 1: Defining New Tools

We will define tools for scraping stock prices, calculating moving averages, and determining trends.

```python
import numpy as np
import httpx
from typing import List

@lmb.tool
def scrape_stock_prices(symbol: str) -> List[float]:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"range": "100d", "interval": "1d"}
    with httpx.Client() as client:
        response = client.get(url, params=params)
        data = response.json()
        prices = data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
        return [float(price) for price in prices if price is not None]

@lmb.tool
def calculate_moving_average(data: List[float], window: int = 20) -> List[float]:
    if window > len(data):
        raise ValueError("Window size cannot be larger than data length")
    ma = np.full(len(data), np.nan)
    for i in range(window - 1, len(data)):
        ma[i] = np.mean(data[i - window + 1 : i + 1])
    return ma.tolist()

@lmb.tool
def calculate_slope(data: List[float], days: int = None) -> float:
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
    functions=[scrape_stock_prices, calculate_moving_average, calculate_slope],
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
    functions=[calculate_total_with_tip, split_bill],
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

## Conclusion

Congratulations! You have successfully built and used an AgentBot to automate tasks related to restaurant bills and stock price analysis. You've also learned how to use the planner bot to handle complex, multi-step tasks efficiently. You can now explore further by adding more tools and customizing the bot to suit your needs. Happy coding!
