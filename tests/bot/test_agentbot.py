"""Test suite for the AgentBot class.

This module contains tests that verify the functionality of the AgentBot,
particularly its caching mechanism and tool execution capabilities.
"""

from llamabot.bot.agentbot import AgentBot
from llamabot.components.tools import tool


def test_agent_caching():
    """Test the caching mechanism of AgentBot.

    This test verifies that:
    1. The AgentBot properly caches results from tool executions
    2. The cached results are correctly stored in the agent's memory
    3. The specific results from tool executions match expected values

    The test uses two simple tools:
    - create_data: Creates a list of sequential numbers
    - sum_data: Sums a list of numbers
    """

    @tool
    def create_data(size: int) -> list:
        """Create a list of numbers.

        :param size: The size of the list to create
        :return: A list of sequential integers from 0 to size-1
        """
        return list(range(size))

    @tool
    def sum_data(numbers: list) -> int:
        """Sum a list of numbers.

        :param numbers: List of numbers to sum
        :return: The sum of all numbers in the list
        """
        return sum(numbers)

    agent = AgentBot(
        system_prompt="Help me with calculations",
        functions=[create_data, sum_data],
        mock_response="mock",  # You'll need to set appropriate mock responses
    )

    # First call should create data and cache it
    _ = agent("Create a list of 5 numbers and then sum them")

    # Verify that we have cached results
    assert len(agent.memory) > 0

    # Verify that one of our cached results is the list [0,1,2,3,4]
    assert any(
        isinstance(cached.result, list) and cached.result == [0, 1, 2, 3, 4]
        for cached in agent.memory.values()
    )
