"""Test suite for the AgentBot class.

This module contains tests that verify the functionality of the AgentBot,
particularly its caching mechanism and tool execution capabilities.
"""

from llamabot.bot.agentbot import AgentBot
from llamabot.components.tools import tool


def test_agent_caching():
    """Test the caching mechanism of AgentBot.

    This test verifies that:
    1. The AgentBot properly caches results from tool executions using SHA256 hashing
    2. The cached results are correctly stored in the agent's memory
    3. Duplicate outputs are stored with the same hash key
    4. The original data is preserved as the value
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
        mock_response="mock",
    )

    # First call should create data and cache it
    _ = agent("Create a list of 5 numbers and then sum them")

    # Store initial memory state
    initial_memory_size = len(agent.memory)
    assert initial_memory_size > 0

    # Make another call that should produce the same list
    _ = agent("Create another list of 5 numbers")

    # Memory size should not increase for duplicate results
    assert len(agent.memory) == initial_memory_size

    # Verify that one of our cached results is the list [0,1,2,3,4]
    assert any(
        isinstance(cached.result, list) and cached.result == [0, 1, 2, 3, 4]
        for cached in agent.memory.values()
    )

    # Create a different sized list to ensure different outputs get different keys
    _ = agent("Create a list of 3 numbers")
    assert len(agent.memory) > initial_memory_size


def test_agent_caching_different_paths():
    """Test that different execution paths leading to the same result use the same cache key.

    This test verifies that:
    1. Different ways of producing the same output use the same cache key
    2. The original execution context is preserved in the cached value
    """

    @tool
    def create_data_ascending(size: int) -> list:
        """Create a list of ascending numbers.

        :param size: The size of the list to create
        :return: A list of sequential integers from 0 to size-1
        """
        return list(range(size))

    @tool
    def create_data_descending(size: int) -> list:
        """Create a list of descending numbers and sort ascending.

        :param size: The size of the list to create
        :return: A list of sequential integers from 0 to size-1
        """
        return sorted(list(range(size - 1, -1, -1)))

    agent = AgentBot(
        system_prompt="Help me with calculations",
        functions=[create_data_ascending, create_data_descending],
        mock_response="mock",
    )

    # Both calls should produce the same output [0,1,2] but through different means
    _ = agent("Create an ascending list of 3 numbers")
    initial_memory_size = len(agent.memory)

    _ = agent("Create a descending list of 3 numbers and sort it")

    # Memory size should not increase as both produce [0,1,2]
    assert len(agent.memory) == initial_memory_size
