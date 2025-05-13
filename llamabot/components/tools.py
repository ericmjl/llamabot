"""Tools for AgentBots to call on.

The design of a tool is as follows:

1. It is a function that is callable.
2. It is decorated with @tool.
3. Being decorated by @tool,
   it will immediately have a pydantic model created for it
   that is attached as an attribute.
"""

from typing import Any, Callable, Dict, Optional, Tuple
from litellm.utils import function_to_dict
from llamabot.bot.simplebot import SimpleBot
from llamabot.components.messages import user
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from uuid import uuid4
import requests
from llamabot.components.sandbox import ScriptExecutor, ScriptMetadata


def tool(func: Callable) -> Callable:
    """Decorator to create a tool from a function.

    :param func: The function to decorate.
    :returns: The decorated function with an attached Function schema.
    """
    # Create and attach the schema
    func.json_schema = function_to_dict(func)
    return func


@tool
def add(a: int, b: int) -> int:
    """Add two integers, a and b, and return the result.

    :param a: The first integer
    :param b: The second integer
    :return: The sum of the two integers
    """
    return a + b


def search_internet(
    search_term: str, max_results: int = 10, backend: str = "lite"
) -> Dict[str, str]:
    """Search the internet for a given term and get webpage contents.

    :param search_term: The search term to look up
    :param max_results: Maximum number of search results to return
    :param backend: The DuckDuckGo backend to use. Default is "lite".
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

    ddgs = DDGS()
    results = ddgs.text(search_term, max_results=max_results, backend=backend)
    webpage_contents = {}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
    }

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
            print(f"Error fetching {url}: {e}")
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


def summarize_web_results(
    search_term: str, webpage_contents: Dict[str, str]
) -> Dict[str, str]:
    """Summarize the content of a webpage in 1 paragraph based on a query.

    :param search_term: The search term to look up
    :param webpage_contents: The content of the webpage
    :return: The summarized content
    """
    bot = SimpleBot(
        system_prompt="You are a helpful assistant that summarizes the content of a webpage in 1 paragraph based on a query.",
        stream_target="none",
    )

    def summarize_content(url: str, content: str) -> Tuple[str, str]:
        """Summarize the content of a webpage in 1 paragraph based on a query.

        :param url: The URL of the webpage
        :param content: The content of the webpage
        :return: The summarized content
        """
        print(f"Summarizing {url}:")
        return url, bot(user(f"Query: {search_term}"), user(content))

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
def search_internet_and_summarize(
    search_term: str, max_results: int, backend: str = "lite"
) -> Dict[str, str]:
    """Search internet for a given term and summarize the results.
    To get a good summary, try increasing the number of results.
    When using search_internet, if you don't get an answer with 1 result,
    try increasing the number of results to get more web pages to summarize.

    :param search_term: The search term to look up
    :param max_results: Maximum number of search results to return
    :param backend: The DuckDuckGo backend to use. Default is "lite".
    :return: Dictionary mapping URLs to markdown-formatted webpage contents
    """
    webpage_contents = search_internet(search_term, max_results, backend)
    summaries = summarize_web_results(search_term, webpage_contents)
    return summaries


@tool
def today_date() -> str:
    """Get the current date."""
    return datetime.now().strftime("%Y-%m-%d")


@tool
def write_and_execute_script(
    code: str,
    dependencies_str: Optional[str] = None,
    python_version: str = ">=3.11",
    timeout: int = 30,
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
    result = executor.run_script(script_path, timeout)

    # Return structured output
    return {
        "stdout": result["stdout"].strip(),
        "stderr": result["stderr"].strip(),
        "status": result["status"],
    }
