# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "llamabot",
# ]
#
# [tool.uv.sources]
# llamabot = { path = "..", editable = true }
# ///
# ---
# title: ReAct Pattern AgentBot Demo
# description: |
#   This notebook demonstrates the new ReAct (Reasoning and Acting) pattern in AgentBot.
#   The agent explicitly shows its reasoning process through Thought-Action-Observation cycles,
#   making its decision-making transparent and easy to follow.
# ---

import marimo

__generated_with = "0.8.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import llamabot as lmb
    from llamabot.bot.agentbot import AgentBot
    from llamabot.components.tools import tool
    from typing import List, Dict
    import json

    return AgentBot, Dict, List, json, lmb, tool


@app.cell
def __():
    # Create a simple demo tool for testing
    @tool
    def get_weather(city: str) -> str:
        """Get current weather information for a city.

        This is a mock weather service for demonstration purposes.
        In a real application, this would connect to a weather API.

        Parameters
        ----------
        city : str
            The name of the city to get weather for

        Returns
        -------
        str
            Weather information including temperature and conditions
        """
        # Mock weather data
        weather_data = {
            "New York": "72°F, sunny with light winds",
            "London": "55°F, cloudy with occasional rain",
            "Tokyo": "68°F, partly cloudy",
            "Paris": "62°F, overcast",
            "Sydney": "75°F, clear skies",
        }

        return weather_data.get(city, f"Weather data not available for {city}")

    return (get_weather,)


@app.cell
def __():
    # Create another demo tool
    @tool
    def calculate_tip(bill_amount: float, tip_percentage: float) -> Dict[str, float]:
        """Calculate tip amount and total bill.

        Parameters
        ----------
        bill_amount : float
            The original bill amount
        tip_percentage : float
            The tip percentage as a decimal (e.g., 0.18 for 18%)

        Returns
        -------
        Dict[str, float]
            Dictionary with tip_amount and total_bill
        """
        tip_amount = bill_amount * tip_percentage
        total_bill = bill_amount + tip_amount

        return {
            "tip_amount": round(tip_amount, 2),
            "total_bill": round(total_bill, 2),
            "tip_percentage": tip_percentage * 100,
        }

    return (calculate_tip,)


@app.cell
def __():
    # Create the ReAct AgentBot
    agent = AgentBot(
        system_prompt=lmb.system(
            """
        You are a helpful assistant that can help with weather information and bill calculations.
        Always think step by step and use your tools when needed.
        """
        ),
        tools=[get_weather, calculate_tip],
        model_name="gpt-4o-mini",
    )

    return (agent,)


@app.cell
def __():
    # Demo 1: Simple weather query
    print("🌤️  Demo 1: Weather Query")
    print("=" * 50)

    weather_response = agent("What's the weather like in New York?")
    print(f"Response: {weather_response.content}")
    print()

    return (weather_response,)


@app.cell
def __():
    # Demo 2: Bill calculation
    print("💰 Demo 2: Bill Calculation")
    print("=" * 50)

    bill_response = agent(
        "I had dinner for $45.50 and want to leave an 18% tip. How much should I pay total?"
    )
    print(f"Response: {bill_response.content}")
    print()

    return (bill_response,)


@app.cell
def __():
    # Demo 3: Complex multi-step task
    print("🔄 Demo 3: Multi-step Task")
    print("=" * 50)

    multi_response = agent(
        """
    I'm planning a trip to London and Paris.
    Can you tell me the weather in both cities,
    and also calculate a 20% tip on a $120 dinner bill?
    """
    )
    print(f"Response: {multi_response.content}")
    print()

    return (multi_response,)


@app.cell
def __():
    # Show the agent's execution metadata
    print("📊 Agent Execution Analytics")
    print("=" * 50)

    if hasattr(agent, "run_meta") and agent.run_meta:
        meta = agent.run_meta
        print(f"Execution time: {meta.get('duration', 'N/A'):.2f} seconds")
        print(f"ReAct cycles used: {meta.get('current_iteration', 'N/A')}")
        print(f"Max iterations: {meta.get('max_iterations', 'N/A')}")

        # Message counts
        message_counts = meta.get("message_counts", {})
        print(f"User messages: {message_counts.get('user', 0)}")
        print(f"Assistant messages: {message_counts.get('assistant', 0)}")
        print(f"Tool messages: {message_counts.get('tool', 0)}")

        # Tool usage statistics
        tool_usage = meta.get("tool_usage", {})
        if tool_usage:
            print("\nTool Usage:")
            for tool_name, stats in tool_usage.items():
                print(f"  {tool_name}: {stats.get('calls', 0)} calls")
                print(f"    Success: {stats.get('success', 0)}")
                print(f"    Failures: {stats.get('failures', 0)}")
                print(f"    Total duration: {stats.get('total_duration', 0):.2f}s")
    else:
        print("No execution metadata available")

    return meta, message_counts, stats, tool_name, tool_usage


@app.cell
def __():
    # Demonstrate the ReAct pattern explicitly
    print("🧠 Understanding the ReAct Pattern")
    print("=" * 50)
    print(
        """
    The ReAct (Reasoning and Acting) pattern works as follows:

    1. **Thought**: The agent analyzes the situation and plans its next action
    2. **Action**: The agent executes a tool or function based on its reasoning
    3. **Observation**: The agent processes the results and updates its understanding

    This cycle repeats until the agent has enough information to provide a complete answer.

    Key Benefits:
    - ✅ Transparent reasoning process
    - ✅ Easy to debug and understand
    - ✅ Iterative learning from observations
    - ✅ Clear separation of thinking and acting
    """
    )

    return


@app.cell
def __():
    # Show how to create custom tools for the ReAct agent
    print("🔧 Creating Custom Tools for ReAct Agent")
    print("=" * 50)

    print(
        """
    To create tools for the ReAct agent, use the @tool decorator with comprehensive docstrings:

    ```python
    @tool
    def my_custom_tool(param1: str, param2: int) -> str:
        \"\"\"Description of what the tool does.

        Parameters
        ----------
        param1 : str
            Description of param1
        param2 : int
            Description of param2

        Returns
        -------
        str
            Description of return value
        \"\"\"
        # Tool implementation
        return result
    ```

    The agent will automatically understand when and how to use your tools
    based on the docstring documentation.
    """
    )

    return


@app.cell
def __():
    # Demonstrate error handling
    print("⚠️  Error Handling Demo")
    print("=" * 50)

    # Create a tool that might fail
    @tool
    def divide_numbers(a: float, b: float) -> float:
        """Divide two numbers.

        Parameters
        ----------
        a : float
            The dividend
        b : float
            The divisor

        Returns
        -------
        float
            The result of a / b

        Raises
        ------
        ValueError
            If b is zero
        """
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

    # Create agent with the new tool
    error_agent = AgentBot(
        system_prompt=lmb.system("You are a math assistant. Be careful with division."),
        tools=[divide_numbers],
        model_name="gpt-4o-mini",
    )

    # Test normal case
    print("Testing normal division:")
    normal_response = error_agent("What is 10 divided by 2?")
    print(f"Response: {normal_response.content}")
    print()

    # Test error case
    print("Testing division by zero:")
    error_response = error_agent("What is 10 divided by 0?")
    print(f"Response: {error_response.content}")

    return divide_numbers, error_agent, normal_response, error_response


@app.cell
def __():
    # Show conversation history and message flow
    print("💬 Message Flow in ReAct Pattern")
    print("=" * 50)

    print(
        """
    The ReAct pattern creates a clear conversation flow:

    1. User: "What's the weather in Tokyo?"

    2. Agent Thought: "I need to get weather information for Tokyo"

    3. Agent Action: get_weather(city="Tokyo")

    4. Observation: "Weather data shows 68°F, partly cloudy"

    5. Agent Thought: "I have the weather information needed"

    6. Agent Action: respond_to_user(response="The weather in Tokyo is 68°F, partly cloudy")

    7. Final Answer: "The weather in Tokyo is 68°F, partly cloudy"

    Each step is clearly labeled and visible, making the agent's reasoning transparent.
    """
    )

    return


@app.cell
def __():
    # Performance comparison
    print("⚡ Performance and Efficiency")
    print("=" * 50)

    print(
        """
    The ReAct pattern provides several advantages:

    🎯 **Transparency**: Every decision is visible
    🔍 **Debugging**: Easy to identify where issues occur
    📚 **Learning**: Can understand the agent's reasoning process
    🛠️ **Control**: Can intervene if the agent goes off track
    📊 **Analytics**: Rich metadata about tool usage and performance

    The agent tracks:
    - Execution time and iteration counts
    - Tool usage statistics (success/failure rates)
    - Message flow and conversation structure
    - Performance metrics for optimization
    """
    )

    return


@app.cell
def __():
    # Best practices for ReAct agents
    print("📋 Best Practices for ReAct Agents")
    print("=" * 50)

    print(
        """
    When building ReAct agents:

    1. **Write Clear Tool Documentation**
       - Use comprehensive docstrings
       - Include parameter types and descriptions
       - Document return values and possible errors

    2. **Design Tools for Specific Use Cases**
       - Each tool should have a clear purpose
       - Avoid overly complex tools
       - Make tools composable

    3. **Handle Errors Gracefully**
       - Use proper exception handling
       - Provide meaningful error messages
       - Allow the agent to recover from failures

    4. **Monitor Performance**
       - Track tool usage statistics
       - Monitor execution times
       - Optimize based on metrics

    5. **Test Thoroughly**
       - Test normal use cases
       - Test error conditions
       - Verify ReAct cycle behavior
    """
    )

    return


@app.cell
def __():
    # Final summary
    print("🎉 ReAct AgentBot Summary")
    print("=" * 50)

    print(
        """
    The new ReAct pattern AgentBot provides:

    ✅ **Explicit Reasoning**: Visible "Thought:" messages
    ✅ **ToolBot Integration**: Intelligent tool selection
    ✅ **Structured Observations**: Clear "Observation:" labels
    ✅ **Transparent Process**: Easy to follow decision-making
    ✅ **Rich Analytics**: Comprehensive performance tracking
    ✅ **Error Handling**: Graceful failure recovery

    This makes AI agents more trustworthy, debuggable, and understandable
    - perfect for applications where transparency and explainability matter!
    """
    )

    return


if __name__ == "__main__":
    app.run()
