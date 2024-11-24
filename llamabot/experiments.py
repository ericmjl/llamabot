"""Experiment tracking functionality."""

from __future__ import annotations

import contextvars
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, Union, cast

from sqlalchemy import JSON, Column, Integer, String, create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

from llamabot.prompt_manager import find_or_set_db_path

# Type alias for metric functions
MetricFunc = TypeVar("MetricFunc", bound=Callable[..., Union[int, float]])

# Context variable to track current experiment run
current_run = contextvars.ContextVar[Optional["Experiment"]](
    "current_run", default=None
)

Base = declarative_base()


class ExperimentRun(Base):
    """A single run of an experiment."""

    __tablename__ = "runs"

    id = Column(Integer, primary_key=True)
    experiment_name = Column(String)
    timestamp = Column(String)

    # Store everything as JSON - no foreign keys
    run_data = Column(JSON, default={})


def upgrade_experiments_db(engine):
    """Upgrade the experiments database schema.

    :param engine: SQLAlchemy engine
    """
    inspector = inspect(engine)

    # Create runs table if it doesn't exist
    if not inspector.has_table("runs"):
        Base.metadata.create_all(engine)
        return

    # Add run_data column if it doesn't exist
    columns = [col["name"] for col in inspector.get_columns("runs")]
    if "run_data" not in columns:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE runs ADD COLUMN run_data JSON"))


class Experiment:
    """Context manager for tracking experiment runs.

    :param name: Name of the experiment
    :param metadata: Additional metadata to store with the run
    :param db_path: Path to the SQLite database. If None, uses default location
    """

    def __init__(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        db_path: Optional[Path] = None,
    ):
        self.name = name
        self.metadata = metadata or {}
        self.db_path = find_or_set_db_path(db_path)

        # Set up database connection
        self.engine = create_engine(f"sqlite:///{self.db_path}")

        # Upgrade database schema
        upgrade_experiments_db(self.engine)

        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()
        self.run: Optional[ExperimentRun] = None

        # Initialize empty run data
        self._run_data = {
            "metadata": self.metadata,
            "message_log_ids": [],
            "prompts": [],  # Will store dicts with hash and id
            "metrics": {},
        }

    def __enter__(self) -> Experiment:
        """Enter the experiment context.

        :return: The experiment instance
        """
        # Create new run
        self.run = ExperimentRun(
            experiment_name=self.name,
            timestamp=datetime.now().isoformat(),
            run_data=self._run_data,
        )
        self.session.add(self.run)
        self.session.commit()

        # Set the current run in context
        current_run.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the experiment context."""
        try:
            if self.run is not None:
                # Update run data before closing
                self.run.run_data = self._run_data
                self.session.commit()
        finally:
            # Always close and invalidate the session
            self.session.close()
            self.session.bind.dispose()  # Dispose the engine connection
            self.session = None  # Invalidate the session
            current_run.set(None)
            self.run = None

    def log_metric(self, name: str, value: float):
        """Log a metric for the current run.

        :param name: Name of the metric
        :param value: Value of the metric
        """
        if self.run is None:
            raise RuntimeError("No active run")

        self._run_data["metrics"][name] = {
            "value": value,
            "timestamp": datetime.now().isoformat(),
        }

        # Update database
        self.run.run_data = self._run_data
        self.session.commit()

    def add_message_log(self, message_log_id: int):
        """Add a message log ID to the current run.

        :param message_log_id: ID of the message log entry
        """
        if self.run is None:
            raise RuntimeError("No active run")

        if message_log_id not in self._run_data["message_log_ids"]:
            self._run_data["message_log_ids"].append(message_log_id)

            # Update database
            self.run.run_data = self._run_data
            self.session.commit()

    def add_prompt(self, prompt_hash: str):
        """Add a prompt hash to the current run.

        :param prompt_hash: Hash of the prompt template
        """
        if self.run is None:
            raise RuntimeError("No active run")

        prompt_info = {"hash": prompt_hash}
        if prompt_info not in self._run_data["prompts"]:
            self._run_data["prompts"].append(prompt_info)

            # Update database
            self.run.run_data = self._run_data
            self.session.commit()


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
        experiment = current_run.get(None)
        if experiment is not None:
            experiment.log_metric(func.__name__, result)

        return result

    return cast(MetricFunc, wrapper)
