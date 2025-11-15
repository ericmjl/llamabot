"""Tools for AgentBots to call on.

The design of a tool is as follows:

1. It is a function that is callable.
2. It is decorated with @tool.
3. Being decorated by @tool,
   it will immediately have a pydantic model created for it
   that is attached as an attribute.
"""

import ast
import os
import random
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, Tuple
from uuid import uuid4

import requests
from bs4 import BeautifulSoup
from docstring_parser import parse
from duckduckgo_search.exceptions import DuckDuckGoSearchException
from loguru import logger

from llamabot.bot.simplebot import SimpleBot
from llamabot.components.messages import user
from llamabot.components.pocketflow import DECIDE_NODE_ACTION, nodeify
from llamabot.components.sandbox import ScriptExecutor, ScriptMetadata
from llamabot.prompt_manager import prompt


def json_schema_type(type_name: str) -> str:
    """Convert Python type names to JSON schema types.

    :param type_name: The Python type name
    :return: The corresponding JSON schema type
    """
    type_mapping = {
        "str": "string",
        "string": "string",
        "int": "integer",
        "integer": "integer",
        "float": "number",
        "number": "number",
        "bool": "boolean",
        "boolean": "boolean",
        "list": "array",
        "array": "array",
        "dict": "object",
        "object": "object",
        "tuple": "array",
        "set": "array",
        "datetime": "string",
        "date": "string",
        "time": "string",
        "path": "string",
        "uuid": "string",
        "decimal": "number",
        "none": "null",
        "null": "null",
    }
    return type_mapping.get(type_name.lower(), "string")


def _extract_type_from_annotation(annotation) -> str:
    """Extract JSON schema type from type annotation using typing module.

    :param annotation: The type annotation to extract from
    :return: The corresponding JSON schema type
    """
    import typing
    from typing import Union

    try:
        # Handle basic types (int, str, etc.) - they have __name__ attribute
        # But skip type aliases like Optional, Union, etc.
        if hasattr(annotation, "__name__") and not hasattr(annotation, "__origin__"):
            return json_schema_type(annotation.__name__)

        # Handle complex types
        origin = typing.get_origin(annotation)

        if origin is Union:
            args = typing.get_args(annotation)
            # Handle Optional[T] and Union[T, None]
            non_none_args = [arg for arg in args if arg is not type(None)]
            if non_none_args:
                return _extract_type_from_annotation(non_none_args[0])
        elif origin is list:
            return "array"
        elif origin is dict:
            return "object"
    except Exception:
        pass
    return "string"  # fallback


def function_to_dict(input_function: Callable) -> Dict[str, Any]:
    """Convert a function to a dictionary for OpenAI function calling.

    Supports numpy, google, and sphinx-style docstrings.

    :param input_function: The function to convert
    :return: Dictionary suitable for OpenAI function calling
    """
    import inspect

    name = input_function.__name__
    docstring = inspect.getdoc(input_function) or ""

    # Parse docstring
    parsed = parse(docstring)

    # Combine short and long descriptions
    description_parts = []
    if parsed.short_description:
        description_parts.append(parsed.short_description)
    if parsed.long_description:
        description_parts.append(parsed.long_description)

    description = "\n\n".join(description_parts) if description_parts else ""

    # Get function parameters and their types from annotations
    parameters = {}
    required_params = []
    param_info = inspect.signature(input_function).parameters

    for param_name, param in param_info.items():
        # Skip internal parameters (those starting with _)
        if param_name.startswith("_"):
            continue

        # Require type hints for all parameters
        if param.annotation == inspect.Parameter.empty:
            raise ValueError(
                f"Parameter '{param_name}' in function '{name}' must have a type annotation"
            )

        # Handle optional parameters
        is_required = param.default == param.empty
        if not is_required:
            # For optional parameters, we'll still include them but mark as not required
            pass

        # Get type from annotation
        param_type = _extract_type_from_annotation(param.annotation)
        if not param_type:
            raise ValueError(
                f"Could not extract type for parameter '{param_name}' in function '{name}'"
            )

        # Get description from parsed docstring
        param_description = ""
        for param in parsed.params:
            if param.arg_name == param_name:
                param_description = param.description or ""
                break

        param_dict = {
            "type": param_type,
            "description": param_description,
        }

        # Add default value if parameter has one
        if not is_required:
            # Convert default value to appropriate JSON-serializable type
            default_value = param.default
            if default_value is not None:
                param_dict["default"] = default_value

        # Remove None values but keep empty strings
        parameters[param_name] = {k: v for k, v in param_dict.items() if v is not None}

        # Only add to required list if parameter has no default
        if is_required:
            required_params.append(param_name)

    # Create the dictionary
    result = {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": parameters,
        },
    }

    # Always add "required" key (even if empty)
    result["parameters"]["required"] = required_params

    return result


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


@nodeify(loopback_name=None)
@tool
def respond_to_user(response: str) -> str:
    """Respond to the user with a message.

    Use this tool when you don't think there's code to write (e.g., greetings, general questions,
    explanations, or when the user just needs a conversational response).

    :param response: The message to send to the user
    :return: The response message
    """
    return response


@nodeify(loopback_name=None)
@tool
def return_object_to_user(variable_name: str, _globals_dict: dict = None) -> Any:
    """Return an object from the calling context's globals to the user.

    Use this tool when you want to return a specific object (e.g., DataFrame, list, dict, etc.)
    from the calling context's globals() dictionary. This is useful when the agent has created
    or modified variables that the user wants to access directly.

    Use `respond_to_user` for text responses, and `return_object_to_user` for returning
    actual Python objects.

    :param variable_name: The name of the variable in globals() to return
    :param _globals_dict: Internal parameter - automatically injected by AgentBot
    :return: The object from globals() with the given name
    :raises ValueError: If the variable is not found in globals_dict
    """
    if _globals_dict is None:
        _globals_dict = {}

    if variable_name not in _globals_dict:
        available_vars = list(_globals_dict.keys())
        raise ValueError(
            f"Variable '{variable_name}' not found in globals(). "
            f"Available variables: {available_vars if available_vars else 'none'}"
        )

    return _globals_dict[variable_name]


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


@nodeify(loopback_name=DECIDE_NODE_ACTION)
@tool
def today_date() -> str:
    """Get the current date.

    :return: The current date in YYYY-MM-DD format
    """
    return datetime.now().strftime("%Y-%m-%d")


# Default tools that are always available in AgentBot and ToolBot
DEFAULT_TOOLS = [today_date, respond_to_user, return_object_to_user]


@nodeify(loopback_name=DECIDE_NODE_ACTION)
@tool
def write_and_execute_script(
    code: str,
    dependencies_str: str = "",
    python_version: str = ">=3.11",
) -> Dict[str, Any]:
    """Write and execute a Python script in a secure sandbox.
    Dependencies should be specified as a comma-separated string, e.g. "requests,beautifulsoup4".
    Script output will be captured from stdout. Use print() to output results.
    Include lots of print() statements in your code to see what is happening.

    :param code: The Python code to execute
    :param dependencies_str: Comma-separated string of pip dependencies
    :param python_version: Python version requirement. Should look like ">=3.11"
    :return: Dictionary containing script execution results
    """
    # Parse dependencies string into list
    dependencies = list(
        dep.strip()
        for dep in dependencies_str.split(",")
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


def _detect_imported_modules(globals_dict: dict) -> list[str]:
    """Detect imported modules from globals_dict.

    :param globals_dict: Dictionary of global variables (typically globals())
    :return: Sorted list of module names that are already imported
    """
    imported_modules = []
    for name, value in globals_dict.items():
        if isinstance(value, types.ModuleType):
            # Get the module name, handling both direct imports and aliases
            module_name = getattr(value, "__name__", None)
            if module_name:
                imported_modules.append(module_name)
    return sorted(set(imported_modules))


def write_and_execute_code(globals_dict: dict):
    """Write and execute code in a secure sandbox.

    Note: This function raises exceptions for error cases (syntax errors, missing functions, etc.).
    Users are expected to handle these exceptions when using this function.

    :param globals_dictionary: The dictionary of global variables to use in the sandbox.
    :return: A function that can be used to execute code in the sandbox.
    """
    # Detect imported modules from globals_dict
    available_modules = _detect_imported_modules(globals_dict)

    # Generate text for available libraries section
    if available_modules:
        module_list = ", ".join(available_modules)
        available_libraries_text = f"The following libraries are already imported and available: {module_list}\n\nYou can use these libraries directly without importing them again. For example, if 'pandas' is listed, you can use 'pd.DataFrame()' if it was imported as 'import pandas as pd', or use the library name directly if imported without an alias."
    else:
        available_libraries_text = "No libraries are currently imported. You should import any libraries you need within your function."

    # Build the docstring with available libraries information
    docstring_template = """Write and execute `placeholder_function` with the passed in `keyword_args`.

Use this tool for any task that requires custom Python code generation and execution.
This tool has access to ALL globals in the current runtime environment (variables, dataframes, functions, etc.).
Perfect for: data analysis, calculations, transformations, visualizations, custom algorithms.

## Code Generation Guidelines:

1. **Write self-contained Python functions** with ALL imports inside the function body
2. **Place all imports at the beginning of the function**: import statements must be the first lines inside the function
3. **Use already-imported libraries when available** (see Available Imported Libraries below), otherwise import everything the function needs
4. **Leverage existing global variables**: Can reference variables that exist in the runtime
5. **Include proper error handling** and docstrings
6. **Provide keyword arguments** when the function requires parameters
7. **Make functions reusable** - they will be stored globally for future use
8. **ALWAYS RETURN A VALUE**: Every function must explicitly return something - never just print, display, or show results without returning them. Even for plotting functions, return the figure/axes object.

## Available Imported Libraries:

{available_libraries_text}

## Function Arguments Handling:

**CRITICAL**: You MUST always pass in keyword_args, which is a dictionary that can be empty, and match the function signature with the keyword_args:

- **If your function takes NO parameters** (e.g., `def analyze_data():`), then pass keyword_args as an **empty dictionary**: `{{}}`
- **If your function takes parameters** (e.g., `def filter_data(min_age, department):`), then pass keyword_args as a dictionary: `{{"min_age": 30, "department": "Engineering"}}`
- **Never pass keyword_args that don't match the function signature** - this will cause execution errors

## Code Structure Example:

```python
# Function with NO parameters - use empty dict {{}}
def analyze_departments():
    '''Analyze department performance.'''
    import pandas as pd
    import numpy as np
    result = fake_df.groupby('department')['salary'].mean()
    return result
# Function WITH parameters - pass matching keyword_args
def filter_employees(min_age, department):
    '''Filter employees by criteria.'''
    import pandas as pd
    filtered = fake_df[(fake_df['age'] >= min_age) & (fake_df['department'] == department)]
    return filtered
```

## Return Value Requirements:

- **Data analysis functions**: Return the computed results (numbers, DataFrames, lists, dictionaries)
- **Plotting functions**: Return the figure or axes object (e.g., `return fig` or `return plt.gca()`)
- **Filter/transformation functions**: Return the processed data
- **Calculation functions**: Return the calculated values
- **Utility functions**: Return relevant output (status, processed data, etc.)
- **Never return None implicitly** - always have an explicit return statement

## Output Format:

This tool returns a dictionary with two keys:
- `"code"`: The generated Python function code as a string
- `"result"`: The execution result of the function

Example access pattern:
```python
output = write_and_execute_code_wrapper(function_code, keyword_args)
print("Generated code:", output["code"])
print("Execution result:", output["result"])
```

## Code Access Capabilities:

The generated code will have access to:

- All global variables and dataframes in the current session
- Any previously defined functions
- The ability to import any standard Python libraries within the function
- The ability to create new reusable functions that will be stored globally

:param placeholder_function: The function to execute (complete Python function as string).
:param keyword_args: The keyword arguments to pass to the function (dictionary matching function parameters).
:return: The result of the function execution.
        """

    docstring = docstring_template.format(
        available_libraries_text=available_libraries_text
    )

    @nodeify
    @tool
    def write_and_execute_code_wrapper(
        placeholder_function: str, keyword_args: dict = dict()
    ):
        """Write and execute a Python function with access to globals.

        This function's docstring is dynamically generated based on available modules.
        See the docstring set after function definition for full documentation.

        :param placeholder_function: The function to execute (complete Python function as string).
        :param keyword_args: The keyword arguments to pass to the function.
        :return: Dictionary with 'code' and 'result' keys.
        """
        import traceback

        # Parse the code to extract the function name
        try:
            tree = ast.parse(placeholder_function)
        except SyntaxError as e:
            error_traceback = traceback.format_exc()
            return {
                "code": placeholder_function,
                "result": None,
                "error": f"Syntax error in provided code: {e}\nCode:\n{placeholder_function}\n\nTraceback:\n{error_traceback}",
            }

        function_name = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_name = node.name
                break

        if function_name is None:
            return {
                "code": placeholder_function,
                "result": None,
                "error": "No function definition found in the provided code.",
            }

        # Store original state to detect new variables
        original_globals_keys = set(globals_dict.keys())
        ns = globals_dict.copy()
        try:
            compiled = compile(placeholder_function, "<llm>", "exec")
            exec(compiled, globals_dict, ns)
        except Exception as e:
            error_traceback = traceback.format_exc()
            return {
                "code": placeholder_function,
                "result": None,
                "error": f"Error compiling code: {type(e).__name__}: {e}\nCode:\n{placeholder_function}\n\nTraceback:\n{error_traceback}",
            }

        # Execute the function with error handling
        try:
            result = ns[function_name](**keyword_args)

            # Detect new variables created during execution
            # Check both globals_dict (for functions defined at module level) and ns (for local variables)
            new_variables = {}

            # Check globals_dict for new variables (functions defined at module level)
            for key, value in globals_dict.items():
                if key not in original_globals_keys:
                    new_variables[key] = value

            # Check ns for new variables (local variables created during execution)
            for key, value in ns.items():
                if key not in globals_dict:
                    new_variables[key] = value
                    # Also add to globals_dict so it's available for future use
                    globals_dict[key] = value

            # Return result with information about created variables
            return {
                "code": placeholder_function,
                "result": result,
                "created_variables": list(new_variables.keys()),
                "function_name": function_name,  # The function that was created
            }
        except Exception as e:
            error_traceback = traceback.format_exc()
            return {
                "code": placeholder_function,
                "result": None,
                "error": f"Error executing code: {type(e).__name__}: {e}\nCode:\n{placeholder_function}\nKeyword args: {keyword_args}\n\nTraceback:\n{error_traceback}",
            }

    # Set the docstring after function definition
    write_and_execute_code_wrapper.__doc__ = docstring

    # Regenerate the JSON schema now that we have the docstring
    function_dict = function_to_dict(write_and_execute_code_wrapper)
    write_and_execute_code_wrapper.json_schema = {
        "type": "function",
        "function": function_dict,
    }

    return write_and_execute_code_wrapper
