"""Experiment tracking functionality for LLM applications.

This module provides tools for tracking and analyzing LLM experiments, including:

- Metric recording and validation via the @metric decorator
- Bot configuration tracking within experiment contexts
- Prompt template registration and monitoring
- Parallel execution utilities for running multiple trials

The main components are:

- Experiment: Context manager for tracking experiment runs
- metric: Decorator for validating and recording numeric metrics
- parallelize: Utility for running multiple trials in parallel

Example usage:

    with Experiment("my_experiment") as exp:
        bot = SimpleBot(system_prompt="You are a helpful assistant")

        @metric
        def response_length(text: str) -> int:
            return len(text)

        response = bot("Hello!")
        length = response_length(response)
"""

import inspect
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import wraps
from inspect import getclosurevars
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast
from uuid import UUID, uuid4

from loguru import logger
from pydantic import BaseModel, Field

from llamabot import SimpleBot

T = TypeVar("T")
MetricFunc = Callable[[Any], Union[int, float]]


def metric(func: MetricFunc) -> MetricFunc:
    """Decorator to validate that a metric function returns a scalar value
    and records it to the current experiment runner if one exists.

    :param func: Function to decorate.
    :returns: Decorated function that validates its return value is scalar.
    :raises ValueError: If the function returns a non-scalar value.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Union[int, float]:
        """Wrap a metric function to validate and record its result.

        :param args: Positional arguments to pass to the wrapped function.
        :param kwargs: Keyword arguments to pass to the wrapped function.
        :returns: The numeric result from the wrapped function.
        :raises ValueError: If the wrapped function returns a non-numeric value.
        """
        # Execute the function only once
        result = func(*args, **kwargs)
        if not isinstance(result, (int, float)):
            raise ValueError(
                f"Metric function {func.__name__} must return int or float, "
                f"got {type(result)}"
            )

        # If there's an active experiment runner, log the result
        experiment = Experiment.get_current()
        if experiment:
            # Log the result multiple times based on num_executions
            experiment.record_metric(func.__name__, result)

        return result

    setattr(wrapper, "_decorator_name", "metric")
    return cast(MetricFunc, wrapper)


def parallelize(func: Callable[[], T], num_executions: int = 1) -> List[T]:
    """Execute a function in parallel multiple times and return results.

    :param func: Function to execute
    :param num_executions: Number of times to execute the function
    :returns: List of results from each execution
    """
    results = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(func) for _ in range(num_executions)]
        for future in futures:
            results.append(future.result())
    return results


class ExperimentRun(BaseModel):
    """Data structure for experiment data.

    :param experiment_id: Unique identifier for this experiment.
    :param name: Human-readable name for the experiment.
    :param prompt_templates: Mapping of prompt names to their template information.
    :param bots: Mapping of bot names to their configurations.
    :param additional_information: Any additional metadata.
    :param created_at: When this experiment was created.
    """

    experiment_id: UUID = Field(default_factory=uuid4)
    name: str
    prompt_templates: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    bots: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    additional_information: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metrics: Dict[str, Union[int, float]] = Field(default_factory=dict)


class Experiment:
    """Context manager for logging experiments.

    :param name: Name of the experiment.
    """

    _context = threading.local()

    def __init__(self, name: str):
        self.name = name
        self.runs: List[ExperimentRun] = []
        self.current_run: Optional[ExperimentRun] = None

    def __enter__(self):
        """Enter the experiment context."""
        logger.info(f"Starting experiment: {self.name}")
        self.current_run = ExperimentRun(name=self.name)
        Experiment._context.current = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the experiment context and inspect frames for bots and prompts."""
        try:
            if self.current_run:
                # Inspect all frames in the call stack
                frame = inspect.currentframe()
                while frame:
                    # Get all variables in the frame
                    frame_locals = frame.f_locals
                    frame_globals = frame.f_globals

                    # Look for bots in local variables
                    for var_name, var_val in frame_locals.items():
                        if isinstance(var_val, SimpleBot):
                            logger.info(
                                f"Bot instance found: {var_name} ({var_val.__class__.__name__})"
                            )
                            self.register_bot(var_val, var_name)

                    # Look for prompts in both locals and globals
                    for namespace in (frame_locals, frame_globals):
                        for var_name, obj in namespace.items():
                            if (
                                hasattr(obj, "_decorator_name")
                                and obj._decorator_name == "prompt"
                            ):
                                logger.info(f"Prompt found: {var_name}")
                                self.register_prompt(obj)

                    frame = frame.f_back

                # Log run completion
                logger.info(f"Tracked metrics: {self.current_run.metrics.keys()}")
                logger.info(
                    f"Tracked prompts: {self.current_run.prompt_templates.keys()}"
                )
                logger.info(f"Tracked bots: {self.current_run.bots.keys()}")

                # Store the completed run
                self.runs.append(self.current_run)
        finally:
            self.current_run = None
            Experiment._context.current = None

    @classmethod
    def get_current(cls) -> Optional["Experiment"]:
        """Get the current experiment from context."""
        return getattr(cls._context, "current", None)

    def record_metric(self, metric_name: str, value: Union[int, float]):
        """Record a metric value."""
        if not self.current_run:
            logger.warning("No active experiment run to record metric")
            return

        if metric_name in self.current_run.metrics:
            logger.warning(
                f"Metric {metric_name} was already recorded. Overwriting previous value."
            )
        self.current_run.metrics[metric_name] = value

    def register_prompt(self, prompt_func: Callable):
        """Register a prompt template that's being used."""
        if self.current_run:
            name = prompt_func.__name__
            docstring = inspect.getdoc(prompt_func)
            self.current_run.prompt_templates[name] = {"docstring": docstring}

    def register_bot(self, bot: SimpleBot, name: str):
        """Register a bot that's being used.

        :param bot: The bot instance to register
        :param name: Name of the bot as defined in code
        :raises ValueError: If a bot with the same name was already registered
        """
        if self.current_run:
            if name in self.current_run.bots:
                raise ValueError(
                    f"Bot name '{name}' is already registered. "
                    "Please use unique variable names for each bot in your code "
                    "to enable proper experiment tracking."
                )
            self.current_run.bots[name] = {
                "class_name": bot.__class__.__name__,
                "model_name": bot.model_name,
                "temperature": bot.temperature,
            }


def experiment(func: Callable):
    """Decorator to track variables used in a generator function and monitor bot creation.

    :param func: The function to decorate
    :returns: Wrapped function that tracks bot creation and prompts
    :raises ValueError: If multiple bots are created with the same variable name
    """
    # Create a list to store bots created during function execution
    created_bots = []

    # Monkey patch SimpleBot to track instantiation
    original_init = SimpleBot.__init__

    def wrapped_init(self, *args, **kwargs):
        """Wrapped initialization function for SimpleBot to track bot creation.

        This function wraps the original SimpleBot.__init__ to track bot instances
        created within the decorated function. It walks up the frame stack
        to find the variable name assigned to the bot instance.

        :param self: The SimpleBot instance being initialized
        :param args: Positional arguments to pass to original __init__
        :param kwargs: Keyword arguments to pass to original __init__
        """
        original_init(self, *args, **kwargs)
        # Get the caller's frame
        frame = inspect.currentframe()
        while frame:
            # We need to walk up the frame stack until we find the frame
            # that's not in the decorator
            if frame.f_code.co_name == func.__name__:
                # Get the local variables in the caller's frame
                caller_locals = frame.f_locals
                # Find the variable name by looking for this instance in locals
                for var_name, var_val in caller_locals.items():
                    if var_val is self:
                        # Skip if the variable name is 'self' and we're inside a class method
                        if var_name == "self" and "self" in caller_locals:
                            continue
                        created_bots.append((var_name, self))
                        break
                break
            frame = frame.f_back

    @wraps(func)
    def wrapper(*args, **kwargs):
        """Wraps the decorated function to track bot creation and prompt usage.

        This wrapper function temporarily modifies the SimpleBot initialization
        to track bot creation, executes the wrapped function, and registers
        any created bots and prompts with the current experiment.

        :param args: Positional arguments to pass to the wrapped function
        :param kwargs: Keyword arguments to pass to the wrapped function
        :returns: The result of the wrapped function
        :raises ValueError: If multiple bots are created with the same variable name
        """
        # Get the current experiment
        experiment = Experiment.get_current()

        # Temporarily replace the __init__ method
        SimpleBot.__init__ = wrapped_init

        try:
            # Clear any previously created bots
            created_bots.clear()

            # Execute the function
            result = func(*args, **kwargs)

            # Register any bots that were created
            if experiment and experiment.current_run:
                # Check for duplicate bot names
                bot_names = [name for name, _ in created_bots]
                if len(bot_names) != len(set(bot_names)):
                    raise ValueError(
                        "Multiple bots were created with the same variable name. "
                        "Please use unique names for each bot to enable proper "
                        "experiment tracking."
                    )

                for var_name, bot in created_bots:
                    logger.info(
                        f"Bot instance found: {var_name} ({bot.__class__.__name__})"
                    )
                    experiment.register_bot(bot, var_name)

                # Look for prompts in closure
                closure_vars = getclosurevars(func)
                for var_name, obj in {
                    **closure_vars.globals,
                    **closure_vars.nonlocals,
                }.items():
                    if (
                        hasattr(obj, "_decorator_name")
                        and obj._decorator_name == "prompt"
                    ):
                        logger.info(f"Prompt found: {var_name}")
                        experiment.register_prompt(obj)

            return result

        finally:
            # Restore original __init__
            SimpleBot.__init__ = original_init
            created_bots.clear()

    return wrapper


# Example
# @experiment
# def name_generator():
#     bot = StructuredBot(
#         jdbot_sysprompt("data science manager"),
#         model_name="gpt-4o",
#         pydantic_model=JobDescription,
#         temperature=0.5,
#     )
#     response = bot(jdbot_user_message("someone who builds full stack AI apps"))
#     return response
