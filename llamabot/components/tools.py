"""Tools for AgentBots to call on.

The design of a tool is as follows:

1. It is a function that is callable.
2. It is decorated with @tool.
3. Being decorated by @tool,
   it will immediately have a pydantic model created for it
   that is attached as an attribute.
"""

from typing import Any, Callable, Dict, List, Optional, Type
from pydantic import BaseModel, Field


class Function(BaseModel):
    """Schema for a function.

    :param name: Name of the function.
    :param description: Description/docstring of the function.
    :param parameters: Parameters of the function, including both args and kwargs.
    :param required: List of required parameter names.
    :param return_type: Return type annotation of the function.
    :param source_code: Source code of the function if available.
    """

    name: str = Field(..., description="Name of the function")
    description: str = Field(..., description="Description/docstring of the function")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters of the function, including both args and kwargs",
    )
    required: List[str] = Field(
        default_factory=list, description="List of required parameter names"
    )
    return_type: Optional[Any] = Field(
        None, description="Return type annotation of the function"
    )
    source_code: Optional[str] = Field(
        None, description="Source code of the function if available"
    )
    _func: Optional[Callable] = None  # Private field to store the callable

    @classmethod
    def from_callable(cls, func: Callable) -> "Function":
        """Create a Function schema from a callable.

        :param func: The callable to create a schema from.
        :returns: A Function schema.
        """
        import inspect
        from typing import get_type_hints

        # Get function signature
        sig = inspect.signature(func)

        # Get docstring
        description = inspect.getdoc(func) or ""

        # Get parameters
        parameters = {}
        required = []
        for name, param in sig.parameters.items():
            if param.default == inspect.Parameter.empty:
                required.append(name)
            parameters[name] = (
                param.annotation if param.annotation != inspect.Parameter.empty else Any
            )

        # Get return type
        type_hints = get_type_hints(func)
        return_type = type_hints.get("return", None)

        # Get source code
        try:
            source_code = inspect.getsource(func)
        except (TypeError, OSError):
            source_code = None

        instance = cls(
            name=func.__name__,
            description=description,
            parameters=parameters,
            required=required,
            return_type=return_type,
            source_code=source_code,
        )
        instance._func = func  # Store the callable for later use
        return instance

    def to_pydantic_model(self) -> Type[BaseModel]:
        """Create a Pydantic model from the function schema.

        It needs to include the function name as well as all of the parameters
        to be passed into the function.

        :returns: A Pydantic model class representing the function parameters.
        """
        from pydantic import create_model, Field

        # Create field definitions for the model
        fields = {
            "function_name": (
                str,
                Field(default=self.name, description="Name of the function"),
            ),
        }

        # Add source code field if available
        if self.source_code is not None:
            fields["source_code"] = (
                str,
                Field(
                    default=self.source_code, description="Source code of the function"
                ),
            )

        # Add parameter fields
        for name, type_annotation in self.parameters.items():
            # If parameter is required, don't provide a default
            if name in self.required:
                fields[name] = (type_annotation, ...)
            else:
                fields[name] = (type_annotation, None)

        # Create and return the model class
        model_name = f"{self.name.title()}Parameters"
        return create_model(model_name, **fields)


def tool(func: Callable) -> Callable:
    """Decorator to create a tool from a function.

    :param func: The function to decorate.
    :returns: The decorated function with an attached Function schema.
    """
    # Create and attach the schema
    func.model = Function.from_callable(func).to_pydantic_model()
    func.json_schema = str(func.model.model_json_schema())
    return func


class FunctionToCall(BaseModel):
    """Pydantic model representing a function call with its parameters.

    :param function_name: Name of the function to call
    :param parameters: Dictionary mapping parameter names to their values
    """

    function_name: str
    parameters: dict[str, Any]


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b
