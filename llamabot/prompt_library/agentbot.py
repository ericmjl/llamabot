"""Prompts for AgentBot.

This module provides system prompts used by AgentBot and related components.
"""

from llamabot.prompt_manager import prompt


@prompt("system")
def decision_bot_system_prompt() -> str:
    """System prompt for the decision-making bot.

    Given the chat history, pick for me one or more tools to execute
    in order to satisfy the user's query.

    Give me just the tool name to pick.
    Use the tools judiciously to help answer the user's query.
    Query is always related to one of the tools.
    Use respond_to_user if you have enough information to answer the original query.
    """
    return ""
