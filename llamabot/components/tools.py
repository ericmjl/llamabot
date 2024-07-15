"""A component that provides tools to the chatbot."""

import inspect
from openai.types.chat import ChatCompletionMessageToolCall
import typing
import json


def python_type_to_json_type(python_type):
    """Map Python types to JSON Schema types."""
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        # Add more mappings as needed
    }
    return type_map.get(python_type, "any")


def type_to_str(type_hint):
    """Convert type hints to JSON-friendly string representations."""
    if type_hint == inspect.Parameter.empty:
        return "any"
    if getattr(type_hint, "__origin__", None) is typing.Literal:
        # Handling typing.Literal to convert it to JSON enum format
        return {"enum": list(type_hint.__args__)}
    if hasattr(type_hint, "__origin__"):  # For handling generic types like List[str]
        origin = type_hint.__origin__
        if origin is list:
            # Assuming only simple types like List[str], not nested like List[List[str]]
            args = type_hint.__args__[0]
            return f"array of {python_type_to_json_type(args)}"
        # Handle other generic types (like Dict, Tuple) here as needed
    return python_type_to_json_type(type_hint)


def describe_function(func):
    """Describe a function as a JSON Schema.

    :param func: The function to describe.
    :return: A JSON Schema describing the function.
    """
    # Extract the signature of the function
    signature = inspect.signature(func)
    docstring = inspect.getdoc(func)

    # Assuming the first line of the docstring is the function description
    function_description = docstring.split("\n")[0]

    # Extracting parameter information
    parameters = {}
    for name, param in signature.parameters.items():
        # Assume the description is in the format: `name: description`
        param_description = [
            line.split(": ")[1]
            for line in docstring.split("\n")
            if line.startswith(name + ":")
        ]
        param_description = param_description[0] if param_description else ""

        # Building the parameter info
        param_type = type_to_str(param.annotation)
        param_info = {"description": param_description}
        if isinstance(param_type, dict):
            # If the type is a dictionary (e.g., for enum), merge it with param_info
            param_info.update(param_type)
        else:
            param_info["type"] = param_type

        parameters[name] = param_info

    # Required parameters are those without default values
    required_params = [
        name
        for name, param in signature.parameters.items()
        if param.default == param.empty
    ]

    # Constructing the final description
    result = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": function_description,
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required_params,
            },
        },
    }

    return result


class Tools:
    """A component that provides tools to the chatbot."""

    def __init__(self, *functions):
        self._available_tools = {func.__name__: func for func in functions}
        self._schemas = {func.__name__: describe_function(func) for func in functions}

    def __call__(
        self, tool_calls: list[ChatCompletionMessageToolCall]
    ) -> dict[str, any]:
        """Call on the tools that were provided.

        :param tool_name: The name of the tool to call.
        :param kwargs: The arguments to pass to the tool.
        """
        result = {}
        if tool_calls:
            for tool_call in tool_calls:
                func_name = tool_call.function.name
                func = self._available_tools[func_name]
                func_kwargs = json.loads(tool_call.function.arguments)
                result[func_name] = func(**func_kwargs)
        return result

    def schemas(self):
        """Return the schemas for the tools."""
        return list(self._schemas.values())
