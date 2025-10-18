# AgentBot Tutorial

Welcome to the AgentBot tutorial! In this tutorial, we will guide you through the process of building a ReAct (Reasoning and Acting) agent that can solve complex problems through explicit reasoning cycles. By the end of this tutorial, you will have a solid understanding of how to use the AgentBot with the ReAct pattern to create transparent, reasoning agents.

## Prerequisites

Before you begin, ensure you have the following:

- Basic knowledge of Python programming.
- Familiarity with the concept of bots and automation.
- Access to a Python environment with the necessary libraries installed.

## What is the ReAct Pattern?

The ReAct (Reasoning and Acting) pattern is a powerful approach for building AI agents that explicitly show their reasoning process. Unlike traditional agents that work "behind the scenes," ReAct agents make their thinking transparent through a structured cycle:

1. **Thought**: The agent analyzes the current situation and plans its next action
2. **Action**: The agent executes a tool or function based on its reasoning
3. **Observation**: The agent processes the results and updates its understanding

This cycle repeats until the agent has enough information to provide a complete answer.

## Part 1: Understanding ReAct with a Simple Example

Let's start with a simple example to understand how the ReAct pattern works.

### Step 1: Setting Up the Environment

First, ensure you have the `llamabot` library installed:

```bash
pip install llamabot
```

### Step 2: Creating a Simple ReAct Agent

We'll create an AgentBot that can search for information and provide answers:

```python
import llamabot as lmb

# Create a ReAct agent with search capabilities
agent = lmb.AgentBot(
    system_prompt=lmb.system("You are a helpful assistant that can search for information."),
    tools=[lmb.search_internet_and_summarize, lmb.today_date],
    model_name="gpt-4o-mini"
)
```

### Step 3: Observing the ReAct Pattern

When you ask the agent a question, you'll see the explicit reasoning process:

```python
response = agent("What's the current weather in New York?")
print(response.content)
```

**Example Output:**
```
Thought: I need to search for current weather information in New York.

[Agent searches the internet for weather data]

Observation: Search results show that the current weather in New York is 72°F and sunny with light winds.

Thought: I now have the weather information needed to answer the user's question.

Answer: The weather in New York is currently 72°F and sunny with light winds.
```

## Part 2: Building a Financial Analysis Agent

Let's create a more sophisticated agent that can analyze financial data using multiple tools.

### Step 1: Defining Custom Tools

```python
import llamabot as lmb
from typing import List
import requests

@lmb.tool
def get_stock_price(symbol: str) -> float:
    """Get the current stock price for a given symbol.

    Parameters
    ----------
    symbol : str
        The stock symbol (e.g., 'AAPL', 'MSFT', 'GOOGL')

    Returns
    -------
    float
        The current stock price

    Raises
    ------
    ValueError
        If the symbol is invalid or data cannot be retrieved
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

@lmb.tool
def calculate_percentage_change(old_price: float, new_price: float) -> float:
    """Calculate the percentage change between two prices.

    Parameters
    ----------
    old_price : float
        The original price
    new_price : float
        The new price

    Returns
    -------
    float
        The percentage change (positive for increase, negative for decrease)
    """
    return ((new_price - old_price) / old_price) * 100

@lmb.tool
def analyze_portfolio(prices: List[float]) -> dict:
    """Analyze a portfolio of stock prices.

    Parameters
    ----------
    prices : List[float]
        List of stock prices

    Returns
    -------
    dict
        Analysis results including average, min, max, and trend
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
    system_prompt=lmb.system("You are a financial analysis assistant. Use your tools to analyze stock data and provide insights."),
    tools=[get_stock_price, calculate_percentage_change, analyze_portfolio],
    model_name="gpt-4o-mini"
)
```

### Step 3: Using the Agent for Analysis

```python
# Ask the agent to analyze multiple stocks
response = financial_agent("""
Please analyze the following stocks:
1. Get the current price of AAPL
2. Get the current price of MSFT
3. Calculate the percentage change from yesterday's prices (AAPL: $145, MSFT: $295)
4. Analyze the portfolio performance
""")

print(response.content)
```

**Example ReAct Output:**
```
Thought: I need to get current prices for AAPL and MSFT, then calculate percentage changes and analyze the portfolio.

Action: get_stock_price(symbol="AAPL")

Observation: AAPL current price is $150.25

Thought: Now I need to get MSFT price and then calculate the percentage changes.

Action: get_stock_price(symbol="MSFT")

Observation: MSFT current price is $300.50

Thought: Now I can calculate the percentage changes from yesterday's prices.

Action: calculate_percentage_change(old_price=145, new_price=150.25)

Observation: AAPL increased by 3.62%

Action: calculate_percentage_change(old_price=295, new_price=300.50)

Observation: MSFT increased by 1.86%

Thought: Now I can analyze the portfolio with both current prices.

Action: analyze_portfolio(prices=[150.25, 300.50])

Observation: Portfolio analysis shows average price of $225.38, minimum of $150.25, maximum of $300.50, with upward trend

Answer: Based on my analysis:
- AAPL: $150.25 (up 3.62% from yesterday's $145)
- MSFT: $300.50 (up 1.86% from yesterday's $295)
- Portfolio shows positive performance with an average price of $225.38 and upward trend
```

## Part 3: Advanced ReAct Features

### Understanding the ReAct Cycle

The ReAct pattern provides several benefits:

1. **Transparency**: You can see exactly how the agent thinks
2. **Debugging**: Easy to identify where problems occur
3. **Learning**: You can understand the agent's reasoning process
4. **Control**: You can intervene if the agent goes off track

### Customizing the ReAct Agent

```python
# Create a specialized agent with custom settings
research_agent = lmb.AgentBot(
    system_prompt=lmb.system("""
    You are a research assistant. Always think step by step and use your tools to gather information.
    Be thorough in your analysis and provide detailed explanations.
    """),
    tools=[lmb.search_internet_and_summarize, lmb.today_date],
    model_name="gpt-4o-mini",
    temperature=0.1,  # Lower temperature for more focused reasoning
    stream_target="stdout"  # Stream the reasoning process
)
```

### Monitoring Agent Performance

```python
# The agent tracks its performance automatically
response = research_agent("Research the latest developments in AI")

# Access performance metrics
metrics = research_agent.run_meta
print(f"Execution time: {metrics['duration']:.2f} seconds")
print(f"ReAct cycles used: {metrics['current_iteration']}")
print(f"Tools used: {list(metrics['tool_usage'].keys())}")

# Tool usage statistics
for tool_name, stats in metrics['tool_usage'].items():
    print(f"{tool_name}: {stats['calls']} calls, {stats['success']} successes")
```

## Part 4: Best Practices for ReAct Agents

### 1. Write Clear Tool Documentation

```python
@lmb.tool
def analyze_sentiment(text: str) -> dict:
    """Analyze the sentiment of text using a simple algorithm.

    This tool performs basic sentiment analysis by counting positive and negative words.
    It's useful for getting a quick understanding of text sentiment.

    Parameters
    ----------
    text : str
        The text to analyze

    Returns
    -------
    dict
        Dictionary with 'sentiment' (positive/negative/neutral), 'score' (0-1), and 'confidence'

    Examples
    --------
    >>> analyze_sentiment("I love this product!")
    {'sentiment': 'positive', 'score': 0.8, 'confidence': 'high'}
    """
    # Simple sentiment analysis implementation
    positive_words = ['love', 'great', 'excellent', 'amazing', 'wonderful']
    negative_words = ['hate', 'terrible', 'awful', 'bad', 'horrible']

    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)

    if pos_count > neg_count:
        return {'sentiment': 'positive', 'score': 0.8, 'confidence': 'high'}
    elif neg_count > pos_count:
        return {'sentiment': 'negative', 'score': 0.2, 'confidence': 'high'}
    else:
        return {'sentiment': 'neutral', 'score': 0.5, 'confidence': 'medium'}
```

### 2. Design Tools for Specific Use Cases

```python
@lmb.tool
def summarize_document(text: str, max_length: int = 200) -> str:
    """Summarize a document to a specified length.

    This tool creates a concise summary of longer text, useful for quick understanding
    of documents, articles, or reports.

    Parameters
    ----------
    text : str
        The text to summarize
    max_length : int, optional
        Maximum length of the summary, by default 200

    Returns
    -------
    str
        The summarized text

    Raises
    ------
    ValueError
        If text is empty or max_length is invalid
    """
    if not text.strip():
        raise ValueError("Text cannot be empty")

    if max_length <= 0:
        raise ValueError("Max length must be positive")

    # Simple summarization (in practice, you'd use more sophisticated methods)
    sentences = text.split('. ')
    summary = '. '.join(sentences[:3])  # Take first 3 sentences

    if len(summary) > max_length:
        summary = summary[:max_length-3] + "..."

    return summary
```

### 3. Handle Errors Gracefully

```python
@lmb.tool
def safe_api_call(url: str, timeout: int = 10) -> dict:
    """Safely make an API call with error handling.

    Parameters
    ----------
    url : str
        The URL to call
    timeout : int, optional
        Request timeout in seconds, by default 10

    Returns
    -------
    dict
        API response or error information

    Raises
    ------
    requests.RequestException
        If the API call fails
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

## Part 5: Advanced ReAct Patterns

### Multi-Step Reasoning

```python
# Create an agent that can handle complex multi-step tasks
complex_agent = lmb.AgentBot(
    system_prompt=lmb.system("""
    You are a research analyst. Break down complex tasks into smaller steps.
    Always think through the problem before acting, and use multiple tools as needed.
    """),
    tools=[
        lmb.search_internet_and_summarize,
        analyze_sentiment,
        summarize_document,
        safe_api_call
    ],
    model_name="gpt-4o-mini"
)

# The agent will naturally break down complex tasks
response = complex_agent("""
Research the latest news about renewable energy, analyze the sentiment of the articles,
and provide a comprehensive summary of the current state of the industry.
""")
```

### Iterative Refinement

The ReAct pattern naturally supports iterative refinement:

```python
response = complex_agent("""
Find information about electric vehicles, then analyze the sentiment of the findings,
and finally provide recommendations for someone considering buying an EV.
""")
```

The agent will:
1. Search for information about electric vehicles
2. Analyze the sentiment of the found information
3. Use the sentiment analysis to inform its recommendations
4. Provide a comprehensive answer based on all the gathered information

## Part 6: Debugging and Optimization

### Understanding Agent Behavior

```python
# Monitor the agent's reasoning process
response = agent("Complex question here")

# Check the conversation history
print("Full conversation:")
for i, msg in enumerate(agent.conversation_history):
    print(f"{i+1}. {msg.role}: {msg.content[:100]}...")
```

### Optimizing Performance

```python
# Create an optimized agent
optimized_agent = lmb.AgentBot(
    system_prompt=lmb.system("You are an efficient assistant. Be concise but thorough."),
    tools=[your_tools],
    model_name="gpt-4o-mini",
    temperature=0.0,  # More deterministic
    max_iterations=5  # Limit iterations for faster responses
)
```

### Error Handling and Recovery

```python
try:
    response = agent("Your question here")
except RuntimeError as e:
    if "exceeded maximum ReAct cycles" in str(e):
        print("Agent couldn't complete the task within the iteration limit")
        print("Try simplifying your request or increasing max_iterations")
    else:
        print(f"Unexpected error: {e}")
```

## Conclusion

Congratulations! You now understand how to use AgentBot with the ReAct pattern to create transparent, reasoning agents. The ReAct pattern provides:

- **Explicit reasoning**: You can see how the agent thinks
- **Transparent decision-making**: Every step is visible
- **Easy debugging**: Problems are easy to identify
- **Flexible tool usage**: Agents can use multiple tools in sequence
- **Iterative refinement**: Agents can build on previous results

The ReAct pattern makes AI agents more trustworthy and understandable, which is crucial for applications where transparency and explainability are important.

You can now build sophisticated agents that can handle complex, multi-step tasks while maintaining full transparency in their reasoning process. Happy coding!
