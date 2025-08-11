"""Tools for AgentBots to call on.

The design of a tool is as follows:

1. It is a function that is callable.
2. It is decorated with @tool.
3. Being decorated by @tool,
   it will immediately have a pydantic model created for it
   that is attached as an attribute.
"""

import asyncio
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple, List
from uuid import uuid4
from abc import ABC, abstractmethod

from bs4 import BeautifulSoup
from duckduckgo_search.exceptions import DuckDuckGoSearchException
from litellm.utils import function_to_dict
from loguru import logger

from llamabot.bot.simplebot import SimpleBot
from llamabot.components.messages import user
from llamabot.components.sandbox import ScriptExecutor, ScriptMetadata
from llamabot.prompt_manager import prompt


# ============================================================================
# ASYNC-FIRST UNIFIED TOOL EXECUTION ARCHITECTURE
# ============================================================================


class ToolExecutor(ABC):
    """Abstract base class for tool execution - all tools go through this."""

    @abstractmethod
    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool with given arguments."""
        pass

    @abstractmethod
    def get_schema(self, tool_name: str) -> Dict[str, Any]:
        """Get the JSON schema for a tool."""
        pass

    @abstractmethod
    def list_tool_names(self) -> List[str]:
        """List all available tool names."""
        pass


class LocalToolExecutor(ToolExecutor):
    """Executes local @tool decorated functions - async-first."""

    def __init__(self, tools: List[Callable]):
        self.tools = {func.__name__: func for func in tools}

    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a local tool - handles both sync and async functions."""
        tool_func = self.tools.get(tool_name)
        if not tool_func:
            raise ValueError(f"Local tool {tool_name} not found")

        logger.debug(f"Executing local tool: {tool_name}")

        # Check if function is async
        if asyncio.iscoroutinefunction(tool_func):
            # Native async function
            return await tool_func(**arguments)
        else:
            # Sync function - run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: tool_func(**arguments))

    def get_schema(self, tool_name: str) -> Dict[str, Any]:
        """Get schema for a local tool."""
        tool_func = self.tools.get(tool_name)
        if not tool_func:
            raise ValueError(f"Local tool {tool_name} not found")
        return tool_func.json_schema

    def list_tool_names(self) -> List[str]:
        """List all local tool names."""
        return list(self.tools.keys())


class MCPToolExecutor(ToolExecutor):
    """Executes tools via MCP client - naturally async."""

    def __init__(self, mcp_client, server_name: str = "mcp"):
        self.client = mcp_client
        self.server_name = server_name
        self._tools_cache = None
        self._schemas_cache = {}

    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute an MCP tool."""
        logger.debug(f"Executing MCP tool: {tool_name} via {self.server_name}")

        try:
            # Add timeout for MCP calls
            result = await asyncio.wait_for(
                self.client.call_tool(tool_name, arguments),
                timeout=30,  # Default MCP timeout
            )
            return result
        except asyncio.TimeoutError:
            raise Exception(f"MCP tool {tool_name} timed out")
        except Exception as e:
            raise Exception(f"MCP tool {tool_name} failed: {e}")

    def get_schema(self, tool_name: str) -> Dict[str, Any]:
        """Get schema for an MCP tool."""
        if tool_name in self._schemas_cache:
            return self._schemas_cache[tool_name]

        # Would need to get from MCP client - simplified for now
        return {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": f"MCP tool: {tool_name}",
                "parameters": {"type": "object", "properties": {}},
            },
        }

    def list_tool_names(self) -> List[str]:
        """List all MCP tool names."""
        # Would need to discover from MCP client - simplified for now
        return []

    async def discover_tools(self):
        """Discover available tools from MCP client."""
        try:
            tools = await self.client.list_tools()
            self._tools_cache = tools

            # Cache schemas
            for tool in tools:
                tool_name = tool.get("name")
                if tool_name:
                    # Convert MCP schema to llamabot format
                    self._schemas_cache[tool_name] = {
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "description": tool.get("description", ""),
                            "parameters": tool.get("inputSchema", {}),
                        },
                    }

            return list(self._schemas_cache.keys())
        except Exception as e:
            logger.warning(f"Failed to discover MCP tools from {self.server_name}: {e}")
            return []


class UnifiedToolExecutor(ToolExecutor):
    """Unified executor that manages both local and MCP tools."""

    def __init__(self):
        self.executors: List[ToolExecutor] = []
        self._tool_to_executor = {}  # tool_name -> executor

    def add_local_tools(self, tools: List[Callable]):
        """Add local tools via LocalToolExecutor."""
        if tools:
            executor = LocalToolExecutor(tools)
            self.executors.append(executor)

            # Map tool names to executor
            for tool_name in executor.list_tool_names():
                self._tool_to_executor[tool_name] = executor

    def add_mcp_client(self, mcp_client, server_name: str = "mcp"):
        """Add MCP tools via MCPToolExecutor."""
        executor = MCPToolExecutor(mcp_client, server_name)
        self.executors.append(executor)
        return executor  # Return for async discovery

    async def discover_all_tools(self):
        """Discover tools from all MCP executors."""
        for executor in self.executors:
            if isinstance(executor, MCPToolExecutor):
                tool_names = await executor.discover_tools()
                # Map discovered tools to executor
                for tool_name in tool_names:
                    self._tool_to_executor[tool_name] = executor

    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute any tool - unified interface."""
        executor = self._tool_to_executor.get(tool_name)
        if not executor:
            raise ValueError(f"Tool {tool_name} not found in any executor")

        return await executor.execute(tool_name, arguments)

    def get_schema(self, tool_name: str) -> Dict[str, Any]:
        """Get schema for any tool."""
        executor = self._tool_to_executor.get(tool_name)
        if not executor:
            raise ValueError(f"Tool {tool_name} not found")

        return executor.get_schema(tool_name)

    def list_tool_names(self) -> List[str]:
        """List all available tool names."""
        return list(self._tool_to_executor.keys())

    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """Get all tool schemas for LLM."""
        schemas = []
        for tool_name in self.list_tool_names():
            try:
                schemas.append(self.get_schema(tool_name))
            except Exception as e:
                logger.warning(f"Failed to get schema for {tool_name}: {e}")
        return schemas


# Global unified executor instance
_unified_executor = None


def get_unified_executor() -> UnifiedToolExecutor:
    """Get the global unified tool executor."""
    global _unified_executor
    if _unified_executor is None:
        _unified_executor = UnifiedToolExecutor()
    return _unified_executor


def tool(func: Callable) -> Callable:
    """Decorator to create a tool from a function.

    Now works with the unified async execution system.
    The function can be sync or async - execution is handled automatically.

    :param func: The function to decorate.
    :returns: The decorated function with an attached Function schema.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        """Wrapper function for tool decorators.

        This wrapper preserves the original function's metadata and signature
        while allowing it to be used as a tool in the agent system.
        It passes through all arguments and return values unchanged.

        :param *args: Positional arguments to be passed to the wrapped function
        :param **kwargs: Keyword arguments to be passed to the wrapped function
        :return: The result of calling the wrapped function with the provided arguments
        """
        return func(*args, **kwargs)

    # Create and attach the schema
    function_dict = function_to_dict(func)
    wrapper.json_schema = {
        "type": "function",
        "function": function_dict,  # Nest function dict under "function" key
    }

    # Mark as async-compatible tool
    wrapper._is_llamabot_tool = True
    wrapper._tool_type = "async" if asyncio.iscoroutinefunction(func) else "sync"

    return wrapper


# ============================================================================
# ASYNC-COMPATIBLE TOOL EXECUTION FUNCTIONS
# ============================================================================


async def execute_tool_call_async(tool_name: str, arguments: Dict[str, Any]) -> Any:
    """Execute any tool call asynchronously using the unified executor.

    This is the new primary interface for tool execution.
    """
    executor = get_unified_executor()
    return await executor.execute(tool_name, arguments)


@tool
def add(a: int, b: int) -> int:
    """Add two integers, a and b, and return the result.

    :param a: The first integer
    :param b: The second integer
    :return: The sum of the two integers
    """
    return a + b


@tool
def respond_to_user(response: str) -> str:
    """Respond to the user with a message. Use this tool to respond to the user when you have a final answer."""
    return response


async def search_internet(search_term: str, max_results: int = 10) -> Dict[str, str]:
    """Search the internet for a given term and get webpage contents.

    :param search_term: The search term to look up
    :param max_results: Maximum number of search results to return
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        raise ImportError(
            "The Python package `duckduckgo_search` cannot be found. Please install it using `pip install llamabot[agent]`."
        )

    try:
        from markdownify import markdownify as md
    except ImportError:
        raise ImportError(
            "The Python package `markdownify` cannot be found. Please install it using `pip install llamabot[agent]`."
        )

    try:
        import aiohttp
    except ImportError:
        raise ImportError(
            "The Python package `aiohttp` cannot be found. Please install it using `pip install aiohttp`."
        )

    def perform_search(ddgs, search_term, max_results):
        """
        Perform a search using the DuckDuckGo search engine with retry logic.

        This function attempts to perform a search operation using the provided
        DuckDuckGoSearch instance (`ddgs`). If the search fails due to a
        `DuckDuckGoSearchException`, it will retry up to 3 times with an
        exponential backoff strategy.

        Args:
            ddgs (DuckDuckGoSearch): An instance of the DuckDuckGo search client.
            search_term (str): The search query to be executed.
            max_results (int): The maximum number of search results to retrieve.

        Returns:
            str: The search results in text format.

        Raises:
            DuckDuckGoSearchException: If all retry attempts fail.
        """
        logger.debug("Attempting DuckDuckGo search with term: {}", search_term)
        try:
            results = ddgs.text(
                search_term, max_results=int(max_results), backend="lite"
            )
            logger.debug("DuckDuckGo search successful, got {} results", len(results))
            return results
        except DuckDuckGoSearchException as e:
            logger.error("DuckDuckGo search failed: {}", str(e))
            logger.error("Error type: {}", type(e).__name__)
            import traceback

            logger.error("Traceback: {}", traceback.format_exc())
            raise

    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 15_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.5 Mobile/15E148 Safari/604.1",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:102.0) Gecko/20100101 Firefox/102.0",
    ]

    accept_headers = [
        "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "text/html,application/xhtml+xml,application/xml;q=0.8,image/webp,*/*;q=0.7",
        "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    ]

    accept_languages = [
        "en-US,en;q=0.5",
        "en-GB,en;q=0.7",
        "en-US,en;q=0.8,fr;q=0.5",
    ]

    headers = {
        "User-Agent": random.choice(user_agents),
        "Accept": random.choice(accept_headers),
        "Accept-Language": random.choice(accept_languages),
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
    }

    # Use the retry-enabled function (DuckDuckGo search is still sync)
    ddgs = DDGS(headers=headers)
    # Run DDG search in executor since it's sync
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None, perform_search, ddgs, search_term, max_results
    )

    webpage_contents = {}

    async def fetch_url(session: aiohttp.ClientSession, url: str) -> Tuple[str, str]:
        """Fetch a URL asynchronously and return the content.

        :param session: The aiohttp session to use
        :param url: The URL to fetch
        :return: Tuple of (url, content) or (url, None) on error
        """
        try:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                text = await response.text()
                soup = BeautifulSoup(text, "html.parser")
                return url, md(soup.get_text())
        except Exception as e:
            logger.debug("Error fetching {}: {}", url, e)
            return url, None

    # Use aiohttp for concurrent async HTTP requests
    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = [fetch_url(session, result["href"]) for result in results]
        fetch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in fetch_results:
            if isinstance(result, Exception):
                logger.debug("Fetch task failed: {}", result)
                continue

            url, content = result
            if content is not None:
                webpage_contents[url] = content

    return webpage_contents


@prompt("system")
def summarization_bot_system_prompt() -> str:
    """
    ## role

    You are a precise summarization assistant that creates focused, relevant summaries of web content.
    Your job is to extract and synthesize the most important information based on the user's query.

    ## summarization guidelines

    1. Focus on relevance:
       - Prioritize information directly related to the search term
       - Filter out tangential or unrelated content
       - Maintain context while being concise

    2. Structure your summary:
       - Start with the most relevant information
       - Include key facts, figures, or findings
       - End with any important implications or conclusions

    3. Quality standards:
       - Be factual and objective
       - Preserve critical context
       - Avoid redundancy
       - Use clear, professional language
       - Keep to one focused paragraph

    4. When information is missing or unclear:
       - Acknowledge gaps explicitly
       - Don't make assumptions
       - Focus on what is known with confidence

    ## output format

    Your summary should be:
    - One paragraph (3-5 sentences)
    - Directly relevant to the search term
    - Self-contained and clear
    - Professional in tone
    - Free of unnecessary qualifiers or hedging
    """


def summarize_web_results(
    search_term: str, webpage_contents: Dict[str, str], **bot_kwargs
) -> Dict[str, str]:
    """Summarize the content of a webpage in 1 paragraph based on a query.

    :param search_term: The search term to look up
    :param webpage_contents: The content of the webpage
    :return: The summarized content
    """
    default_model_name = os.getenv(
        "LMB_INTERNET_SUMMARIZER_MODEL_NAME", "ollama_chat/llama3.1:latest"
    )
    model_name = bot_kwargs.pop("model_name", default_model_name)
    stream_target = bot_kwargs.pop("stream_target", "none")
    system_prompt = bot_kwargs.pop("system_prompt", summarization_bot_system_prompt())
    bot = SimpleBot(
        system_prompt=system_prompt,
        stream_target=stream_target,
        model_name=model_name,
        **bot_kwargs,
    )

    def summarize_content(url: str, content: str) -> Tuple[str, str]:
        """Summarize the content of a webpage in 1 paragraph based on a query.

        :param url: The URL of the webpage
        :param content: The content of the webpage
        :return: The summarized content
        """
        logger.debug("Summarizing {}:", url)
        return url, bot(user(f"Query: {search_term}"), user(content)).content

    summaries = {}
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(summarize_content, url, content)
            for url, content in webpage_contents.items()
        ]
        for future in as_completed(futures):
            url, summary = future.result()
            summaries[url] = summary

    return summaries


@tool
async def search_internet_and_summarize(
    search_term: str, max_results: int
) -> Dict[str, str]:
    """Search internet for a given term and summarize the results.
    To get a good summary, try increasing the number of results.
    When using search_internet, if you don't get an answer with 1 result,
    try increasing the number of results to get more web pages to summarize.

    :param search_term: The search term to look up
    :param max_results: Maximum number of search results to return
    :param bot_kwargs: Keyword arguments to pass to the bot.
    :return: Dictionary mapping URLs to markdown-formatted webpage contents
    """
    logger.debug("Starting search for term: {}", search_term)
    try:
        # Ensure max_results is an integer
        max_results = int(max_results)
        logger.debug("Using max_results: {}", max_results)

        # Run search in executor to avoid blocking
        loop = asyncio.get_event_loop()
        webpage_contents = await loop.run_in_executor(
            None, search_internet, search_term, max_results
        )
        logger.debug("Search completed. Found {} webpages", len(webpage_contents))
        if not webpage_contents:
            logger.warning("No webpage contents found for search term: {}", search_term)
            return {}

        logger.debug("Starting summarization of {} webpages", len(webpage_contents))
        # Run summarization in executor to avoid blocking
        summaries = await loop.run_in_executor(
            None, summarize_web_results, search_term, webpage_contents
        )
        logger.debug("Summarization completed. Generated {} summaries", len(summaries))
        return summaries
    except Exception as e:
        logger.error("Error in search_internet_and_summarize: {}", str(e))
        logger.error("Error type: {}", type(e).__name__)
        import traceback

        logger.error("Traceback: {}", traceback.format_exc())
        raise


@tool
def today_date() -> str:
    """Get the current date."""
    return datetime.now().strftime("%Y-%m-%d")


@tool
async def write_and_execute_script(
    code: str,
    dependencies_str: Optional[str] = None,
    python_version: str = ">=3.11",
) -> Dict[str, Any]:
    """Write and execute a Python script in a secure sandbox.
    Dependencies should be specified as a comma-separated string, e.g. "requests,beautifulsoup4".
    Script output will be captured from stdout. Use print() to output results.
    Include lots of print() statements in your code to see what is happening.
    Estimate the timeout parameter to avoid errors.

    :param code: The Python code to execute
    :param dependencies_str: Comma-separated string of pip dependencies
    :param python_version: Python version requirement
    :param timeout: Execution timeout in seconds
    :return: Dictionary containing script execution results
    """
    # Parse dependencies string into list
    dependencies = list(
        dep.strip()
        for dep in (dependencies_str or "").split(",")
        if dep.strip()  # Filter out empty strings
    )
    # Modify the Python version if it doesn't have a version specifier
    if not any(python_version.startswith(op) for op in (">=", "<=", "==", "~=")):
        python_version = f">={python_version}"

    # Create metadata
    metadata = ScriptMetadata(
        requires_python=python_version,
        dependencies=dependencies,
        auth=str(uuid4()),  # Generate unique ID for this execution
        timestamp=datetime.now(),
    )

    # Run script execution in executor to avoid blocking
    loop = asyncio.get_event_loop()

    def _execute_script():
        """Execute a Python script in a secure sandbox."""
        # Initialize executor
        executor = ScriptExecutor()
        # Write and run script
        script_path = executor.write_script(code, metadata)
        return executor.run_script(script_path, 600)

    result = await loop.run_in_executor(None, _execute_script)

    # Return structured output
    return {
        "stdout": result["stdout"].strip(),
        "stderr": result["stderr"].strip(),
        "status": result["status"],
    }
