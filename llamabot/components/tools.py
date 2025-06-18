"""Tools for AgentBots to call on.

The design of a tool is as follows:

1. It is a function that is callable.
2. It is decorated with @tool.
3. Being decorated by @tool,
   it will immediately have a pydantic model created for it
   that is attached as an attribute.
"""

import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple
from uuid import uuid4

import requests
from bs4 import BeautifulSoup
from duckduckgo_search.exceptions import DuckDuckGoSearchException
from litellm.utils import function_to_dict
from loguru import logger

from llamabot.bot.simplebot import SimpleBot
from llamabot.components.messages import user
from llamabot.components.sandbox import ScriptExecutor, ScriptMetadata
from llamabot.prompt_manager import prompt


def tool(func: Callable) -> Callable:
    """Decorator to create a tool from a function.

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
    return wrapper


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


def search_internet(search_term: str, max_results: int = 10) -> Dict[str, str]:
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

    # Use the retry-enabled function
    ddgs = DDGS(headers=headers)
    results = perform_search(ddgs, search_term, max_results)
    webpage_contents = {}

    def fetch_url(url: str, headers: Dict[str, str]) -> Tuple[str, str]:
        """Fetch a URL and return the content.

        :param url: The URL to fetch
        :param headers: The headers to use for the request
        :return: The content of the URL
        """
        try:
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            return url, md(soup.get_text())
        except Exception as e:
            logger.debug("Error fetching {}: {}", url, e)
            return url, None

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(fetch_url, result["href"], headers) for result in results
        ]
    for future in as_completed(futures):
        url, summarized_content = future.result()
        if summarized_content is not None:
            webpage_contents[url] = summarized_content

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
def search_internet_and_summarize(search_term: str, max_results: int) -> Dict[str, str]:
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

        webpage_contents = search_internet(search_term, max_results)
        logger.debug("Search completed. Found {} webpages", len(webpage_contents))
        if not webpage_contents:
            logger.warning("No webpage contents found for search term: {}", search_term)
            return {}

        logger.debug("Starting summarization of {} webpages", len(webpage_contents))
        summaries = summarize_web_results(search_term, webpage_contents)
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
def write_and_execute_script(
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

    # Initialize executor
    executor = ScriptExecutor()

    # Write and run script
    script_path = executor.write_script(code, metadata)
    result = executor.run_script(script_path, 600)

    # Return structured output
    return {
        "stdout": result["stdout"].strip(),
        "stderr": result["stderr"].strip(),
        "status": result["status"],
    }
