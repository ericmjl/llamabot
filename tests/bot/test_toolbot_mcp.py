"""Tests for ToolBot with MCP integration."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from fastmcp import Client

from llamabot.bot.toolbot import ToolBot, toolbot_sysprompt


@pytest.fixture
def mock_mcp_clients():
    """Create mock MCP clients for testing."""
    # Mock local client
    local_client = AsyncMock(spec=Client)
    local_tool = MagicMock()
    local_tool.name = "add"
    local_tool.description = "Add two numbers"
    local_tool.inputSchema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
    }
    local_client.list_tools.return_value = [local_tool]
    local_client.call_tool.return_value = 5

    # Mock remote client
    remote_client = AsyncMock(spec=Client)
    remote_tool = MagicMock()
    remote_tool.name = "multiply"
    remote_tool.description = "Multiply two numbers"
    remote_tool.inputSchema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
    }
    remote_client.list_tools.return_value = [remote_tool]
    remote_client.call_tool.return_value = 20

    return [local_client, remote_client]


def test_toolbot_initialization_with_mcp_clients(mock_mcp_clients):
    """Test ToolBot initialization with MCP clients."""
    toolbot = ToolBot(
        system_prompt=toolbot_sysprompt(globals_dict={}),
        model_name="test-model",
        mcp_clients=mock_mcp_clients,
    )

    assert toolbot.mcp_clients == mock_mcp_clients
    assert toolbot._tool_schemas is None  # Lazy loaded


@pytest.mark.asyncio
async def test_load_tool_schemas(mock_mcp_clients):
    """Test loading tool schemas from MCP clients."""
    toolbot = ToolBot(
        system_prompt=toolbot_sysprompt(globals_dict={}),
        model_name="test-model",
        mcp_clients=mock_mcp_clients,
    )

    schemas = await toolbot._load_tool_schemas()

    assert len(schemas) == 2  # One from each client
    assert schemas[0]["function"]["name"] == "add"
    assert schemas[1]["function"]["name"] == "multiply"

    # Test caching
    schemas2 = await toolbot._load_tool_schemas()
    assert schemas == schemas2


@pytest.mark.asyncio
async def test_execute_tool_call_success(mock_mcp_clients):
    """Test successful tool execution."""
    toolbot = ToolBot(
        system_prompt=toolbot_sysprompt(globals_dict={}),
        model_name="test-model",
        mcp_clients=mock_mcp_clients,
    )

    # Mock tool call
    tool_call = MagicMock()
    tool_call.function.name = "add"
    tool_call.function.arguments = '{"a": 2, "b": 3}'

    result = await toolbot.execute_tool_call(tool_call)

    assert result == "5"
    mock_mcp_clients[0].call_tool.assert_called_once_with("add", {"a": 2, "b": 3})


@pytest.mark.asyncio
async def test_execute_tool_call_fallback(mock_mcp_clients):
    """Test tool execution with fallback to second client."""
    toolbot = ToolBot(
        system_prompt=toolbot_sysprompt(globals_dict={}),
        model_name="test-model",
        mcp_clients=mock_mcp_clients,
    )

    # Make first client fail
    mock_mcp_clients[0].call_tool.side_effect = Exception("Tool not found")

    # Mock tool call for remote tool
    tool_call = MagicMock()
    tool_call.function.name = "multiply"
    tool_call.function.arguments = '{"a": 4, "b": 5}'

    result = await toolbot.execute_tool_call(tool_call)

    assert result == "20"
    mock_mcp_clients[1].call_tool.assert_called_once_with("multiply", {"a": 4, "b": 5})


@pytest.mark.asyncio
async def test_execute_tool_call_not_found(mock_mcp_clients):
    """Test tool execution when tool is not found on any client."""
    toolbot = ToolBot(
        system_prompt=toolbot_sysprompt(globals_dict={}),
        model_name="test-model",
        mcp_clients=mock_mcp_clients,
    )

    # Make all clients fail
    for client in mock_mcp_clients:
        client.call_tool.side_effect = Exception("Tool not found")

    # Mock tool call for non-existent tool
    tool_call = MagicMock()
    tool_call.function.name = "nonexistent"
    tool_call.function.arguments = "{}"

    with pytest.raises(
        ValueError, match="Tool nonexistent not found on any MCP server"
    ):
        await toolbot.execute_tool_call(tool_call)


def test_toolbot_call_sync_wrapper(mock_mcp_clients):
    """Test that ToolBot.__call__ works as a sync wrapper."""
    toolbot = ToolBot(
        system_prompt=toolbot_sysprompt(globals_dict={}),
        model_name="test-model",
        mcp_clients=mock_mcp_clients,
    )

    # Mock the async call to return empty tool calls
    toolbot._async_call = AsyncMock(return_value=[])

    result = toolbot("What is 2 + 2?")

    assert result == []
    toolbot._async_call.assert_called_once()
