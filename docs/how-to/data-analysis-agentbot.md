---
title: Data Analysis Agentbot
marimo-version: 0.18.4
width: full
header: |-
  # /// script
  # requires-python = ">=3.10"
  # dependencies = [
  #     "llamabot[all]",
  #     "marimo>=0.17.0",
  #     "pandas",
  #     "numpy",
  # ]
  # ///
---

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/ericmjl/llamabot/blob/main/docs/how-to/data-analysis-agentbot.py)

```python {.marimo}
import marimo as mo
```

## How to Build a Data Analysis Chatbot with AgentBot

Learn how to build a chatbot that executes code for data analysis using AgentBot.
Unlike ToolBot which handles single-turn function calls, AgentBot can orchestrate
multi-step workflows and make decisions about which tools to use.
<!---->
## Prerequisites

Before you begin, ensure you have:

- **Ollama installed and running locally**: Visit [ollama.ai](https://ollama.ai) to install
- **Required Ollama model**: Run `ollama pull deepseek-r1:32b` (or another model that supports tool calling)
- **Python 3.10+** with llamabot, pandas, and numpy installed
- **Sample data** to analyze (or we'll create some in this guide)

All llamabot models in this guide use the `ollama_chat/` prefix for local execution.
<!---->
## Goal

By the end of this guide, you'll have built a data analysis chatbot that:

- Executes Python code to analyze data
- Makes multi-step decisions about which analyses to perform
- Returns DataFrames and visualizations
- Provides observability through spans and workflow visualization

```python {.marimo}
import pandas as pd
import numpy as np

import llamabot as lmb
from llamabot.bot.agentbot import AgentBot
from llamabot.components.tools import tool
```

## Step 1: Create Sample Data

Let's create some sample data to analyze. In a real scenario, you'd load your own data.

```python {.marimo}
# Create sample sales data
np.random.seed(42)
dates = pd.date_range("2024-01-01", periods=100, freq="D")
sales_data = pd.DataFrame(
    {
        "date": dates,
        "product": np.random.choice(["Widget A", "Widget B", "Widget C"], 100),
        "sales": np.random.randint(10, 100, 100),
        "revenue": np.random.uniform(100, 1000, 100),
        "region": np.random.choice(["North", "South", "East", "West"], 100),
    }
)

sales_data.head()
```

## Step 2: Create Data Analysis Tools

We'll create tools that the agent can use to analyze data. Each tool is decorated with `@tool`
to make it agent-callable.

```python {.marimo}
@tool
def calculate_statistics(
    dataframe_name: str, column: str, _globals_dict: dict = None
) -> str:
    """Calculate basic statistics (mean, median, std) for a column in a DataFrame.

    :param dataframe_name: Name of the DataFrame variable in globals
    :param column: Name of the column to analyze
    :param _globals_dict: Internal parameter - automatically injected by AgentBot
    :return: String summary of statistics
    """
    if _globals_dict is None or dataframe_name not in _globals_dict:
        return f"DataFrame '{dataframe_name}' not found in workspace."

    df = _globals_dict[dataframe_name]
    if column not in df.columns:
        return f"Column '{column}' not found in DataFrame."

    stats = {
        "mean": df[column].mean(),
        "median": df[column].median(),
        "std": df[column].std(),
        "min": df[column].min(),
        "max": df[column].max(),
    }

    return f"Statistics for {column}:\n" + "\n".join(
        f"  {k}: {v:.2f}" for k, v in stats.items()
    )
```

```python {.marimo}
@tool
def group_by_analysis(
    dataframe_name: str,
    group_by: str,
    aggregate_column: str,
    _globals_dict: dict = None,
) -> str:
    """Group DataFrame by a column and aggregate another column.

    :param dataframe_name: Name of the DataFrame variable in globals
    :param group_by: Column name to group by
    :param aggregate_column: Column name to aggregate
    :param _globals_dict: Internal parameter - automatically injected by AgentBot
    :return: String representation of grouped results
    """
    if _globals_dict is None or dataframe_name not in _globals_dict:
        return f"DataFrame '{dataframe_name}' not found in workspace."

    df = _globals_dict[dataframe_name]
    if group_by not in df.columns or aggregate_column not in df.columns:
        return "One or more columns not found in DataFrame."

    grouped = (
        df.groupby(group_by)[aggregate_column]
        .agg(["sum", "mean", "count"])
        .round(2)
    )
    return f"Grouped analysis by {group_by}:\n{grouped.to_string()}"
```

```python {.marimo}
@tool
def execute_custom_code(code: str, _globals_dict: dict = None) -> str:
    """Execute custom Python code for data analysis.

    This tool allows the agent to execute arbitrary Python code for complex analyses.
    Use this when standard tools aren't sufficient.

    :param code: Python code to execute (must be safe and data-focused)
    :param _globals_dict: Internal parameter - automatically injected by AgentBot
    :return: String representation of the result
    """
    if _globals_dict is None:
        return "No workspace available."

    try:
        # Execute code with access to globals (including DataFrames)
        exec(code, _globals_dict)
        return "Code executed successfully. Check workspace for results."
    except Exception as e:
        return f"Error executing code: {str(e)}"
```

## Step 3: Create the AgentBot

AgentBot orchestrates multiple tools and makes decisions about which ones to use.
It uses a graph-based workflow where tools can loop back to the decision node.

```python {.marimo}
# Create AgentBot with our data analysis tools
analysis_agent = AgentBot(
    tools=[calculate_statistics, group_by_analysis, execute_custom_code],
    system_prompt="""You are a data analysis assistant. You help users analyze data by:
    1. Understanding what analysis they want
    2. Selecting the appropriate tool(s) to use
    3. Executing multi-step analyses when needed
    4. Returning clear, informative results

    Available tools:
    - calculate_statistics: Get basic stats for a column
    - group_by_analysis: Group and aggregate data
    - execute_custom_code: Run custom Python code for complex analyses

    Always use return_object_to_user() to return DataFrames or results to the user.
    """,
    model_name="ollama_chat/deepseek-r1:32b",
)
```

## Step 4: Visualize the Agent Workflow

AgentBot automatically generates a mermaid diagram showing the workflow graph.
Blue nodes are tools that loop back to the decision node, green nodes are terminal tools.

```python {.marimo}
# Display the agent to see the workflow graph
analysis_agent
```

The mermaid diagram shows:

- **Decision node**: Where the agent decides which tool to use
- **Tool nodes (blue)**: Tools that can loop back for multi-step workflows
- **Terminal nodes (green)**: Tools like `respond_to_user` that end the workflow

This visualization helps you understand how the agent orchestrates tools.
<!---->
## Step 5: Use the Agent for Data Analysis

Now let's use the agent to analyze our data. The agent will decide which tools to use
and can perform multi-step analyses.

```python {.marimo}
# Use the agent to analyze data
# The agent will decide which tools to use based on the query
result = analysis_agent(
    "Calculate the mean and standard deviation of sales, then group by region and show total revenue per region.",
    globals(),
)

print(result)
```

## Step 6: View Observability with Spans

AgentBot creates spans that track the entire workflow, including decision-making and tool execution.

```python {.marimo}
# Display spans to see the agent's decision-making process
analysis_agent.spans
```

The spans show:

- **agentbot_call**: The main agent call with query and max_iterations
- **iterations**: How many tool calls were made
- **result**: The final result
- **Nested spans**: Each tool execution creates its own span

This observability helps you understand:

- Which tools the agent chose to use
- How many steps were needed
- What decisions were made at each step
<!---->
## Step 7: Create an Interactive Chat Interface

Let's create a Marimo chat interface so users can interact with the agent naturally.

```python {.marimo}
def chat_turn(messages, config):
    """Handle a chat turn with the data analysis agent."""
    user_message = messages[-1].content

    # Make sure sales_data is available in globals
    globals_dict = {"sales_data": sales_data}

    # Call the agent
    result = analysis_agent(user_message, globals_dict)

    return result

# Create chat interface with example prompts
example_prompts = [
    "What's the average sales by product?",
    "Show me total revenue per region",
    "Calculate statistics for the sales column",
]

chat = mo.ui.chat(chat_turn, max_height=600, prompts=example_prompts)
```

```python {.marimo}
mo.vstack(
    [
        mo.md("### Data Analysis Agent"),
        mo.md(
            "Ask questions about the sales data. The agent will decide which tools to use."
        ),
        chat,
    ]
)
```

## Summary

You've built a data analysis chatbot with AgentBot that:

- Executes code for data analysis
- Makes multi-step decisions about which tools to use
- Orchestrates complex workflows automatically
- Provides workflow visualization through mermaid diagrams
- Tracks observability through spans
- Offers an interactive chat interface

**Key Takeaways:**

- AgentBot orchestrates multiple tools in a graph-based workflow
- Tools decorated with `@tool` become agent-callable
- Display the agent to see the workflow graph visualization
- Spans track decision-making and tool execution
- AgentBot can handle multi-step workflows automatically
- Use `globals_dict` to share data between tool calls
- Terminal tools (like `respond_to_user`) end the workflow
