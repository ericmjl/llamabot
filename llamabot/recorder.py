"""Prompt recorder class definition."""

import contextvars
import functools
import hashlib
import inspect as std_inspect
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from loguru import logger
from pyprojroot import here
from sqlalchemy import (
    Column,
    Connection,
    Engine,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

from llamabot.components.messages import BaseMessage
from llamabot.utils import find_or_set_db_path, get_object_name

prompt_recorder_var = contextvars.ContextVar("prompt_recorder")
current_span_var = contextvars.ContextVar("current_span", default=None)
current_trace_id_var = contextvars.ContextVar("current_trace_id", default=None)

# Design sqlite database for storing prompts and responses.
# Columns:
# - Python object name from globals()
# - Time stamp
# - Full message log as a JSON string of dictionaries.


Base = declarative_base()


class MessageLog(Base):
    """A log of a message exchange."""

    __tablename__ = "message_log"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(String, index=True)
    object_name = Column(String)
    model_name = Column(String)
    temperature = Column(Float)
    message_log = Column(Text)
    rating = Column(Integer, nullable=True)  # 1 for thumbs up, 0 for thumbs down
    span_id = Column(String, ForeignKey("spans.span_id"), nullable=True, index=True)
    trace_id = Column(String, nullable=True, index=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._message_log_dict = None

    @property
    def message_log_dict(self) -> Dict[str, Any]:
        """
        Returns the message log as a dictionary.

        If the message log is not already parsed, it attempts to parse it from JSON.
        If parsing fails, it returns an empty dictionary.

        :return: The message log as a dictionary.
        """
        if not hasattr(self, "_message_log_dict"):
            try:
                self._message_log_dict = (
                    json.loads(self.message_log) if self.message_log else {}
                )
            except json.JSONDecodeError:
                self._message_log_dict = {}
        return self._message_log_dict


class SpanRecord(Base):
    """A span record in the database."""

    __tablename__ = "spans"

    id = Column(Integer, primary_key=True, index=True)
    trace_id = Column(String, index=True)
    span_id = Column(String, unique=True, index=True)
    parent_span_id = Column(String, ForeignKey("spans.span_id"), nullable=True)
    operation_name = Column(String, index=True)
    start_time = Column(String, index=True)
    end_time = Column(String, nullable=True)
    duration_ms = Column(Float, nullable=True)
    attributes = Column(Text)  # JSON string
    events = Column(Text)  # JSON string
    status = Column(String)  # "started", "completed", "error"
    error_message = Column(Text, nullable=True)


class Runs(Base):
    """A record of an experiment run."""

    __tablename__ = "runs"

    id = Column(Integer, primary_key=True)
    experiment_name = Column(String)
    timestamp = Column(String)
    run_metadata = Column(Text)
    run_data = Column(Text)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._run_data_dict = None

    @property
    def run_data_dict(self) -> Dict[str, Any]:
        """
        Returns the run data as a dictionary.

        If the run data is not already parsed, it attempts to parse it from JSON.
        If parsing fails, it returns an empty dictionary.

        :return: The run data as a dictionary.
        """
        if not hasattr(self, "_run_data_dict"):
            try:
                self._run_data_dict = json.loads(self.run_data) if self.run_data else {}
            except json.JSONDecodeError:
                self._run_data_dict = {}
        return self._run_data_dict


def upgrade_database(engine: Engine):
    """Upgrade the database schema."""
    # Create all tables from Base (now includes Runs and Span)
    Base.metadata.create_all(engine)

    # Add new columns to existing tables if needed
    with engine.connect() as connection:
        inspector = inspect(engine)
        for table in [MessageLog, Prompt, Runs, SpanRecord]:
            if inspector.has_table(table.__tablename__):
                existing_columns = [
                    c["name"] for c in inspector.get_columns(table.__tablename__)
                ]
                for column in table.__table__.columns:
                    if column.name not in existing_columns:
                        try:
                            add_column(connection, table.__tablename__, column)
                        except OperationalError as e:
                            print(f"Error adding column {column.name}: {e}")

    # Add rating column if it doesn't exist
    inspector = inspect(engine)
    if "message_log" in inspector.get_table_names():
        columns = [col["name"] for col in inspector.get_columns("message_log")]
        if "rating" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text("ALTER TABLE message_log ADD COLUMN rating INTEGER")
                )

        # Add span_id and trace_id columns if they don't exist
        if "span_id" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text("ALTER TABLE message_log ADD COLUMN span_id TEXT")
                )
        if "trace_id" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text("ALTER TABLE message_log ADD COLUMN trace_id TEXT")
                )


def add_column(connection: Connection, table_name: str, column: Column):
    """
    Add a new column to an existing table.

    :param connection: SQLAlchemy connection
    :param table_name: Name of the table to modify
    :param column: Column object to add
    """
    column_name = column.name
    column_type = column.type.compile(connection.dialect)
    connection.execute(
        text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
    )


# Example usage:
# To add a new column in the future, you would do:
# add_column(engine, PromptResponseLog.__tablename__, Column('new_column_name', String))


def ensure_db_in_gitignore(db_path: Path) -> None:
    """Ensure the database file is listed in .gitignore.

    If the database is in a .llamabot directory, this function checks if the
    .llamabot/.gitignore file exists and is properly configured.
    Otherwise, it attempts to add the database file to the project's .gitignore.

    :param db_path: Path to the database file
    """
    try:
        # Check if the db_path is inside a .llamabot directory
        if ".llamabot" in db_path.parts:
            llamabot_dir = db_path.parent
            gitignore_path = llamabot_dir / ".gitignore"

            # Ensure the .llamabot/.gitignore exists with proper content
            if not gitignore_path.exists():
                with open(gitignore_path, "w") as f:
                    f.write("# Ignore all files in this directory\n*")
            return

        # For databases not in .llamabot directory, use the original approach
        repo_root = here()
        gitignore_path = repo_root / ".gitignore"
        db_filename = db_path.name

        if gitignore_path.exists():
            with open(gitignore_path, "r+") as f:
                content = f.read()
                if db_filename not in content:
                    f.write(f"\n# SQLite database\n{db_filename}\n")
        else:
            with open(gitignore_path, "w") as f:
                f.write(f"# SQLite database\n{db_filename}\n")
    except Exception as e:
        logger.debug(f"Could not update .gitignore: {e}")


# @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def sqlite_log(
    obj: Any, messages: list[BaseMessage], db_path: Optional[Path] = None
) -> int:
    """Log messages to the sqlite database for further analysis.

    :param obj: The object to log the messages for.
    :param messages: The messages to log.
    :param db_path: The path to the database to use.
        If not specified, defaults to ~/.llamabot/message_log.db
    :return: ID of the created message log entry
    """
    # Set up the database path
    db_path = find_or_set_db_path(db_path)

    # Ensure database is in .gitignore if we're in a git repo
    ensure_db_in_gitignore(db_path)

    # Create the engine and session
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    upgrade_database(engine)  # Add this line to ensure the database is upgraded
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Get the object name
        object_name = get_object_name(obj)

        # Get the current timestamp
        timestamp = datetime.now().isoformat()

        # Convert messages to a JSON string, including prompt_hash
        message_logs = []
        for message in messages:
            logger.debug(f"Processing message with role: {message.role}")

            # Extract basic message info
            message_dict = {
                "role": message.role,
                "content": message.content,
            }

            # Add prompt hash if it exists
            if hasattr(message, "prompt_hash"):
                message_dict["prompt_hash"] = message.prompt_hash
                logger.debug(f"Message has prompt hash: {message.prompt_hash}")
            else:
                message_dict["prompt_hash"] = None
                logger.debug("Message has no prompt hash")

            # Process tool calls if they exist
            if hasattr(message, "tool_calls") and message.tool_calls:
                logger.debug(f"Processing {len(message.tool_calls)} tool calls")
                tool_calls = []
                for tool_call in message.tool_calls:
                    tool_call_dict = {
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        }
                    }
                    logger.debug(
                        f"Tool call: {tool_call.function.name} with args: {tool_call.function.arguments}"
                    )
                    tool_calls.append(tool_call_dict)
                message_dict["tool_calls"] = tool_calls
            else:
                message_dict["tool_calls"] = []
                logger.debug("Message has no tool calls")

            message_logs.append(message_dict)

        # Convert to JSON string
        message_log = json.dumps(message_logs)
        logger.debug(f"Final message log JSON: {message_log}")

        # Get current span if available
        current_span = current_span_var.get(None)
        span_id = current_span.span_id if current_span else None
        trace_id = current_span.trace_id if current_span else None

        # Create a new MessageLog instance
        new_log = MessageLog(
            object_name=object_name,
            timestamp=timestamp,
            message_log=message_log,
            model_name=obj.model_name,
            temperature=obj.temperature if hasattr(obj, "temperature") else None,
            span_id=span_id,
            trace_id=trace_id,
        )

        # Add the new log to the session and commit
        session.add(new_log)
        session.commit()

        # Get the ID before we do anything else
        log_id = new_log.id

        # If we're in an experiment context, add this message log to the run
        from .experiments import current_run

        experiment = current_run.get(None)
        if experiment is not None:
            experiment.add_message_log(log_id)

        return log_id

    finally:
        # Always close the session
        session.close()


class Prompt(Base):
    """A version-controlled prompt template."""

    __tablename__ = "prompts"

    id = Column(Integer, primary_key=True, index=True)
    hash = Column(String, unique=True, index=True)
    function_name = Column(String)
    template = Column(Text)
    previous_version_id = Column(Integer, ForeignKey("prompts.id"), nullable=True)
    previous_version = relationship(
        "Prompt", remote_side=[id], backref="next_version", uselist=False
    )


def hash_template(template: str) -> str:
    """Hash a prompt template."""
    return hashlib.sha256(template.encode()).hexdigest()


def store_prompt_version(
    session, template: str, function_name: str, previous_hash: Optional[str] = None
):
    """Store a new prompt version in the database."""
    logger.debug(f"Storing prompt version for function: {function_name}")
    template_hash = hash_template(template)

    existing_prompt = session.query(Prompt).filter_by(hash=template_hash).first()
    if existing_prompt:
        logger.debug(f"Existing prompt found with hash: {template_hash}")
        return existing_prompt

    previous_version = None
    if previous_hash:
        previous_version = session.query(Prompt).filter_by(hash=previous_hash).first()
        logger.debug(
            f"Previous version found: {previous_version.id if previous_version else None}"
        )

    new_prompt = Prompt(
        hash=template_hash,
        template=template,
        function_name=function_name,
        previous_version=previous_version,
    )
    session.add(new_prompt)
    logger.debug(f"New prompt added to session with hash: {template_hash}")
    session.flush()
    logger.debug(f"Session flushed, new prompt id: {new_prompt.id}")
    return new_prompt


def generate_span_id() -> str:
    """Generate a unique span ID.

    :return: Unique span ID string
    """
    return str(uuid.uuid4())


def get_or_create_trace_id() -> str:
    """Get or create a trace ID for the current context.

    :return: Trace ID string
    """
    trace_id = current_trace_id_var.get(None)
    if trace_id is None:
        trace_id = str(uuid.uuid4())
        current_trace_id_var.set(trace_id)
    return trace_id


def _serialize_value(value: Any, max_size: int = 1000) -> Any:
    """Serialize a value for storage in span attributes.

    :param value: Value to serialize
    :param max_size: Maximum size in bytes before truncation
    :return: Serialized value
    """
    try:
        json_str = json.dumps(value)
        if len(json_str) > max_size:
            return json_str[:max_size] + "... (truncated)"
        return value
    except (TypeError, ValueError):
        str_value = str(value)
        if len(str_value) > max_size:
            return str_value[:max_size] + "... (truncated)"
        return str_value


class Span:
    """Represents a single span in a trace with dictionary-like interface.

    :param operation_name: Name of the operation
    :param trace_id: Trace ID (if None, uses or creates current trace ID)
    :param parent_span_id: Parent span ID for nesting
    :param db_path: Database path for saving span
    :param attributes: Initial span attributes
    """

    def __init__(
        self,
        operation_name: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        db_path: Optional[Path] = None,
        **attributes,
    ):
        self.span_id = generate_span_id()
        self.operation_name = operation_name
        # Get trace_id from context if not provided, but track if we created a new one
        if trace_id is None:
            existing_trace_id = current_trace_id_var.get(None)
            if existing_trace_id is None:
                # Create new trace_id but don't set it in context yet
                self.trace_id = str(uuid.uuid4())
                self._trace_id_was_created = True
            else:
                # Use existing trace_id from context
                self.trace_id = existing_trace_id
                self._trace_id_was_created = False
        else:
            # Explicit trace_id provided
            self.trace_id = trace_id
            self._trace_id_was_created = False
        self.parent_span_id = parent_span_id
        self.start_time = datetime.now()
        self.end_time = None
        self.duration_ms = None
        self.attributes = dict(attributes)
        self.events = []
        self.status = "started"
        self.error_message = None
        self._db_path = db_path
        self._previous_span = None
        self._previous_trace_id = None

    def __enter__(self):
        """Enter span context - sets as current span."""
        # Store previous span and trace_id to restore on exit
        self._previous_span = current_span_var.get(None)
        self._previous_trace_id = current_trace_id_var.get(None)
        current_span_var.set(self)
        # Set trace ID in context
        current_trace_id_var.set(self.trace_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit span context - automatically calculates duration and saves to DB."""
        self.end_time = datetime.now()
        # AUTOMATIC DURATION CALCULATION
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000

        if exc_type is not None:
            self.status = "error"
            self.error_message = str(exc_val)
        else:
            self.status = "completed"

        # Save to database
        self._save_to_db()

        # Restore previous span from context (for nested spans)
        if self._previous_span is not None:
            current_span_var.set(self._previous_span)
        else:
            current_span_var.set(None)

        # Restore previous trace_id from context
        # If we created a new trace_id AND we're a root span (no parent), clear it on exit
        # This ensures root spans clear their trace_id, but nested spans restore it
        # so other nested spans in the same trace can still use it
        if self._trace_id_was_created and self.parent_span_id is None:
            # We created a new trace_id as a root span, so clear it on exit
            # This prevents subsequent manual spans from reusing this trace_id
            current_trace_id_var.set(None)
        elif self._previous_trace_id is not None:
            # Restore the previous trace_id that was in context
            # This allows nested spans to maintain the trace context
            current_trace_id_var.set(self._previous_trace_id)
        else:
            # No previous trace_id, clear it
            current_trace_id_var.set(None)

    def __getitem__(self, key: str) -> Any:
        """Get attribute value using dictionary-like syntax.

        :param key: Attribute key
        :return: Attribute value
        """
        return self.attributes[key]

    def __setitem__(self, key: str, value: Any):
        """Set attribute value using dictionary-like syntax.

        :param key: Attribute key
        :param value: Attribute value
        """
        self.attributes[key] = _serialize_value(value)

    def __delitem__(self, key: str):
        """Delete attribute using dictionary-like syntax.

        :param key: Attribute key
        """
        del self.attributes[key]

    def __contains__(self, key: str) -> bool:
        """Check if attribute exists using dictionary-like syntax.

        :param key: Attribute key
        :return: True if key exists
        """
        return key in self.attributes

    def get(self, key: str, default: Any = None) -> Any:
        """Get attribute value with default.

        :param key: Attribute key
        :param default: Default value if key doesn't exist
        :return: Attribute value or default
        """
        return self.attributes.get(key, default)

    def log(self, event_name: str, **data):
        """Log an event within this span.

        :param event_name: Name of the event
        :param data: Event data
        """
        event = {
            "name": event_name,
            "timestamp": datetime.now().isoformat(),
            "data": data,
        }
        self.events.append(event)

    def set_attribute(self, key: str, value: Any):
        """Set a span attribute (for backward compatibility).

        :param key: Attribute key
        :param value: Attribute value
        """
        self[key] = value

    def span(self, operation_name: str, **attributes):
        """Create a nested child span.

        :param operation_name: Name of the child operation
        :param attributes: Initial attributes for child span
        :return: New Span instance
        """
        return Span(
            operation_name,
            trace_id=self.trace_id,
            parent_span_id=self.span_id,
            db_path=self._db_path,
            **attributes,
        )

    def _save_to_db(self):
        """Save span to database."""
        if self._db_path is None:
            self._db_path = find_or_set_db_path(None)

        engine = create_engine(f"sqlite:///{self._db_path}")
        upgrade_database(engine)
        Session = sessionmaker(bind=engine)
        session = Session()

        try:
            span_record = SpanRecord(
                trace_id=self.trace_id,
                span_id=self.span_id,
                parent_span_id=self.parent_span_id,
                operation_name=self.operation_name,
                start_time=self.start_time.isoformat(),
                end_time=self.end_time.isoformat() if self.end_time else None,
                duration_ms=self.duration_ms,
                attributes=json.dumps(self.attributes),
                events=json.dumps(self.events),
                status=self.status,
                error_message=self.error_message,
            )
            session.add(span_record)
            session.commit()
        except Exception as e:
            logger.error(f"Error saving span to database: {e}")
            session.rollback()
        finally:
            session.close()

    def _repr_html_(self) -> str:
        """Return HTML string for marimo display.

        When a Span object is the last expression in a marimo cell,
        this method is automatically called to display the span visualization.
        Shows only this span and its direct/indirect children, not all spans in the trace.

        **Data Flow:**
        1. Convert `self` (Span object) to dictionary format using `span_to_dict(self)`
        2. Query database for spans that are children of this span (parent_span_id=self.span_id)
           - Recursively collect all descendants
        3. Merge spans from both sources:
           - Database spans: Already dictionaries from `get_spans()`
           - Current span (`self`): Converted from Span object to dict via `span_to_dict()`
        4. Deduplicate by `span_id` (database spans take precedence if duplicate)
        5. Always ensure `self` is included even if not yet saved to database
        6. All spans are now in unified dictionary format for visualization

        **Handles:**
        - Incomplete spans (still in progress, end_time is None)
        - Spans not yet saved to database (includes self even if not saved)
        - HTML escaping for security
        - Pagination for large traces (default 25 spans per page)

        :return: Complete HTML string with embedded CSS and JavaScript
        """
        # Convert self Span object to dict
        self_dict = span_to_dict(self)

        # Get all spans in trace to find children
        try:
            all_trace_spans_objects = get_spans(
                trace_id=self.trace_id, db_path=self._db_path
            )
            # Convert Span objects to dictionaries for processing
            # get_spans() returns SpanList, which is iterable
            all_trace_spans = [span_to_dict(s) for s in all_trace_spans_objects]
        except Exception:
            all_trace_spans = []

        # Build span dict for quick lookup
        span_dict_lookup = {s["span_id"]: s for s in all_trace_spans}
        span_dict_lookup[self.span_id] = self_dict  # Ensure self is included

        # Recursively collect all descendants of self
        def collect_descendants(span_id: str, collected: set) -> None:
            """Recursively collect all descendant spans."""
            if span_id in collected:
                return
            collected.add(span_id)
            # Find all spans that have this span as parent
            for span in all_trace_spans:
                if span.get("parent_span_id") == span_id:
                    collect_descendants(span["span_id"], collected)

        # Collect this span and all its descendants
        relevant_span_ids = set()
        collect_descendants(self.span_id, relevant_span_ids)

        # Filter to only relevant spans (self + descendants)
        relevant_spans = [
            span_dict_lookup[span_id]
            for span_id in relevant_span_ids
            if span_id in span_dict_lookup
        ]

        # Build hierarchical structure
        trace_tree = build_hierarchy(relevant_spans)

        # Generate HTML
        return generate_span_html(
            span_dict=self_dict,
            all_spans=relevant_spans,
            trace_tree=trace_tree,
            current_span_id=self.span_id,
        )


class SpanFactory:
    """Factory that can work as both decorator and context manager."""

    def __init__(self, operation_name, log_return, exclude_args, db_path, attributes):
        self.operation_name = operation_name
        self.log_return = log_return
        self.exclude_args = exclude_args
        self.db_path = db_path
        self.attributes = attributes
        self._span_obj = None

    def __call__(self, func=None):
        """When called with a function, use as decorator."""
        if func is not None:
            return _span_decorator(
                operation_name=self.operation_name or func.__name__,
                log_return=self.log_return,
                exclude_args=self.exclude_args or [],
                db_path=self.db_path,
                **self.attributes,
            )(func)
        # If called without function, return context manager
        return self._get_span()

    def _get_span(self):
        """Get span context manager."""
        trace_id = get_or_create_trace_id()
        return Span(
            self.operation_name or "unnamed",
            trace_id=trace_id,
            db_path=self.db_path,
            **self.attributes,
        )

    def __enter__(self):
        """Enter context manager."""
        self._span_obj = self._get_span()
        return self._span_obj.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if self._span_obj:
            return self._span_obj.__exit__(exc_type, exc_val, exc_tb)


def span(
    operation_name: Optional[Union[str, Callable]] = None,
    log_return: bool = False,
    exclude_args: Optional[List[str]] = None,
    db_path: Optional[Path] = None,
    **attributes,
):
    """Create a span context manager or decorator.

    Can be used in two ways:
    1. As context manager: with span("op_name") as s: ...
    2. As decorator: @span("op_name") or @span() (uses function name)

    :param operation_name: Name of operation. If None and used as decorator, uses function.__name__
        If callable, treated as function to decorate (for @span usage)
    :param log_return: If True (when used as decorator), log return value
    :param exclude_args: List of argument names to exclude from logging
    :param db_path: Database path for saving spans
    :param attributes: Additional span attributes
    :return: Span context manager or decorator function
    """
    # If called as decorator without parentheses: @span
    if callable(operation_name):
        func = operation_name
        return _span_decorator(
            operation_name=func.__name__,
            log_return=log_return,
            exclude_args=exclude_args or [],
            db_path=db_path,
            **attributes,
        )(func)

    # Return factory that works as both decorator and context manager
    return SpanFactory(
        operation_name=operation_name,
        log_return=log_return,
        exclude_args=exclude_args or [],
        db_path=db_path,
        attributes=attributes,
    )


def _span_decorator(
    operation_name: Optional[str] = None,
    log_return: bool = False,
    exclude_args: Optional[List[str]] = None,
    db_path: Optional[Path] = None,
    **attributes,
):
    """Internal decorator factory for span."""
    exclude_args = exclude_args or []

    def decorator(func):
        """Decorator function that wraps the target function with span tracking.

        :param func: Function to wrap
        :return: Wrapped function with span tracking
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """Wrapper function that creates a span around function execution.

            :param args: Positional arguments
            :param kwargs: Keyword arguments
            :return: Function result
            """
            # Determine operation name
            op_name = operation_name or func.__name__

            # Create span
            trace_id = get_or_create_trace_id()
            span_obj = Span(op_name, trace_id=trace_id, db_path=db_path, **attributes)

            # Log function inputs and outputs automatically
            try:
                sig = std_inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()

                # Log all inputs as attributes (coerce to strings)
                for param_name, param_value in bound_args.arguments.items():
                    if param_name not in exclude_args:
                        try:
                            span_obj[f"input_{param_name}"] = str(param_value)
                        except Exception:
                            span_obj[f"input_{param_name}"] = (
                                f"<unable to stringify {type(param_value).__name__}>"
                            )
            except Exception:
                # Fallback if signature inspection fails (built-ins, C extensions, etc.)
                # Log positional args by index
                if args:
                    span_obj["args_count"] = len(args)
                    for i, arg_value in enumerate(args):
                        try:
                            span_obj[f"input_arg_{i}"] = str(arg_value)
                        except Exception:
                            span_obj[f"input_arg_{i}"] = (
                                f"<unable to stringify {type(arg_value).__name__}>"
                            )
                # Log keyword args
                for key, value in kwargs.items():
                    if key not in exclude_args:
                        try:
                            span_obj[f"input_{key}"] = str(value)
                        except Exception:
                            span_obj[f"input_{key}"] = (
                                f"<unable to stringify {type(value).__name__}>"
                            )

            with span_obj:
                try:
                    result = func(*args, **kwargs)
                    # Always log output as attribute (coerce to string)
                    try:
                        span_obj["output"] = str(result)
                    except Exception:
                        span_obj["output"] = (
                            f"<unable to stringify {type(result).__name__}>"
                        )
                    return result
                except Exception as e:
                    span_obj.log(
                        "function_error", error=str(e), error_type=type(e).__name__
                    )
                    raise

        return wrapper

    return decorator


def get_current_span() -> Optional[Span]:
    """Get the currently active span from context.

    :return: Current span object or None if no active span
    """
    return current_span_var.get(None)


def get_spans(
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    operation_name: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    db_path: Optional[Path] = None,
    **attribute_filters,
) -> "SpanList":
    """Query spans from the database.

    :param trace_id: Filter by trace ID
    :param span_id: Filter by specific span ID
    :param operation_name: Filter by operation name
    :param start_time: Filter spans that started after this time
    :param end_time: Filter spans that ended before this time
    :param db_path: Database path. If None, uses find_or_set_db_path(None)
    :param attribute_filters: Additional attribute filters (e.g., model="gpt-4")
    :return: SpanList containing Span objects
    """
    # Resolve database path
    if db_path is None:
        db_path = find_or_set_db_path(None)
    else:
        db_path = Path(db_path)

    # Create engine for this specific database
    engine = create_engine(f"sqlite:///{db_path}")
    upgrade_database(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Build query
        query = session.query(SpanRecord)

        # Apply filters
        if trace_id:
            query = query.filter(SpanRecord.trace_id == trace_id)
        if span_id:
            query = query.filter(SpanRecord.span_id == span_id)
        if operation_name:
            query = query.filter(SpanRecord.operation_name == operation_name)
        if start_time:
            query = query.filter(SpanRecord.start_time >= start_time.isoformat())
        if end_time:
            query = query.filter(SpanRecord.end_time <= end_time.isoformat())

        # Note: Attribute filtering is done in Python after loading spans
        # because SQLite JSON extraction can be unreliable across versions

        # Execute query and convert to dictionaries
        spans = query.all()
        result = []
        for span_record in spans:
            span_dict = {
                "span_id": span_record.span_id,
                "trace_id": span_record.trace_id,
                "parent_span_id": span_record.parent_span_id,
                "operation_name": span_record.operation_name,
                "start_time": span_record.start_time,
                "end_time": span_record.end_time,
                "duration_ms": span_record.duration_ms,
                "attributes": (
                    json.loads(span_record.attributes) if span_record.attributes else {}
                ),
                "events": json.loads(span_record.events) if span_record.events else [],
                "status": span_record.status,
                "error_message": span_record.error_message,
            }
            result.append(span_dict)

        # Apply attribute filters in Python
        if attribute_filters:
            filtered_result = []
            for span_dict in result:
                match = True
                for key, value in attribute_filters.items():
                    if span_dict["attributes"].get(key) != value:
                        match = False
                        break
                if match:
                    filtered_result.append(span_dict)
            result = filtered_result

        # Convert dictionaries to Span objects
        span_objects = []
        for span_dict in result:
            span_obj = dict_to_span(span_dict, db_path=db_path)
            span_objects.append(span_obj)

        return SpanList(span_objects)
    finally:
        session.close()


def get_span_tree(
    trace_id: str,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build hierarchical tree structure of spans for a trace.

    :param trace_id: The trace ID to build tree for
    :param db_path: Database path. If None, uses find_or_set_db_path(None)
    :return: Nested dictionary representing span tree with root span and children
    """
    # Get all spans for trace_id (returns SpanList)
    spans_objects = get_spans(trace_id=trace_id, db_path=db_path)

    if not spans_objects:
        return {}

    # Convert Span objects to dictionaries for tree building
    # SpanList is iterable, so we can iterate over it
    spans = [span_to_dict(s) for s in spans_objects]

    # Build parent-child relationships
    span_dict = {span["span_id"]: span for span in spans}
    root_spans = [span for span in spans if span["parent_span_id"] is None]

    def build_tree(span: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively build tree structure."""
        children = [
            build_tree(span_dict[child_id])
            for child_id, child_span in span_dict.items()
            if child_span["parent_span_id"] == span["span_id"]
        ]
        result = span.copy()
        result["children"] = children
        return result

    # Return first root span (or build tree for all roots)
    if root_spans:
        return build_tree(root_spans[0])
    return {}


def enable_span_recording():
    """Enable automatic span recording globally."""
    # This sets a global flag that bots can check
    # Implementation will be in bot classes
    import llamabot.recorder as recorder_module

    recorder_module._span_recording_enabled = True


def is_span_recording_enabled() -> bool:
    """Check if span recording is enabled.

    :return: True if span recording is enabled
    """
    import llamabot.recorder as recorder_module

    return getattr(recorder_module, "_span_recording_enabled", False)


def span_to_dict(span: Span) -> Dict[str, Any]:
    """Convert Span object to dictionary format matching get_spans() output.

    :param span: Span object to convert
    :return: Dictionary with keys matching get_spans() format
    """
    return {
        "span_id": span.span_id,
        "trace_id": span.trace_id,
        "parent_span_id": span.parent_span_id,
        "operation_name": span.operation_name,
        "start_time": span.start_time.isoformat() if span.start_time else None,
        "end_time": span.end_time.isoformat() if span.end_time else None,
        "duration_ms": span.duration_ms,
        "attributes": span.attributes,
        "events": span.events,
        "status": span.status,
        "error_message": span.error_message,
    }


def dict_to_span(span_dict: Dict[str, Any], db_path: Optional[Path] = None) -> Span:
    """Convert span dictionary to Span object for visualization.

    Creates a Span object from a dictionary (e.g., from get_spans()).
    The returned Span object can be used for visualization via _repr_html_().

    :param span_dict: Span dictionary from get_spans()
    :param db_path: Database path (optional, will be inferred if None)
    :return: Span object that can be displayed
    """
    # Parse ISO timestamps back to datetime objects
    start_time = None
    if span_dict.get("start_time"):
        try:
            start_time = datetime.fromisoformat(
                span_dict["start_time"].replace("Z", "+00:00")
            )
        except (ValueError, AttributeError):
            start_time = None

    end_time = None
    if span_dict.get("end_time"):
        try:
            end_time = datetime.fromisoformat(
                span_dict["end_time"].replace("Z", "+00:00")
            )
        except (ValueError, AttributeError):
            end_time = None

    # Create a new Span object
    span = Span(
        operation_name=span_dict.get("operation_name", ""),
        trace_id=span_dict.get("trace_id"),
        parent_span_id=span_dict.get("parent_span_id"),
        db_path=db_path,
        **span_dict.get("attributes", {}),
    )

    # Manually set all the fields to match the dict
    span.span_id = span_dict.get("span_id", span.span_id)
    span.start_time = start_time or span.start_time
    span.end_time = end_time
    span.duration_ms = span_dict.get("duration_ms")
    span.attributes = span_dict.get("attributes", {})
    span.events = span_dict.get("events", [])
    span.status = span_dict.get("status", "completed")
    span.error_message = span_dict.get("error_message")

    return span


class SpanList:
    """A list-like collection of spans with unified visualization.

    Behaves like a list but displays all spans together in a single
    visualization when used as the last expression in a marimo cell.

    Supports multiple root spans from different traces, showing them
    all in a unified timeline view.
    """

    def __init__(self, spans: List[Span]):
        self._spans = spans

    def __iter__(self):
        """Iterate over spans."""
        return iter(self._spans)

    def __getitem__(self, index):
        """Get span by index or slice."""
        if isinstance(index, slice):
            return SpanList(self._spans[index])
        return self._spans[index]

    def __len__(self):
        """Return number of spans."""
        return len(self._spans)

    def __contains__(self, item):
        """Check if span is in collection."""
        return item in self._spans

    def __repr__(self):
        """String representation."""
        return f"SpanList({len(self._spans)} spans)"

    def __add__(self, other):
        """Concatenate with another SpanList or list of spans."""
        if isinstance(other, SpanList):
            return SpanList(self._spans + other._spans)
        elif isinstance(other, list):
            return SpanList(self._spans + other)
        return NotImplemented

    def _repr_html_(self) -> str:
        """Display all spans together in unified visualization.

        Shows all spans in a single timeline view, handling multiple
        root spans from different traces. Root spans are shown at the
        top level, with their children nested below.

        :return: Complete HTML string with embedded CSS and JavaScript
        """
        if not self._spans:
            return (
                '<div style="padding: 1rem; color: #2E3440;">No spans to display.</div>'
            )

        # Convert all spans to dictionaries
        span_dicts = [span_to_dict(s) for s in self._spans]

        # Find all root spans (spans with no parent)
        root_spans = [s for s in span_dicts if s.get("parent_span_id") is None]

        # If no root spans found, use the first span as the "current" span
        if root_spans:
            # Use the first root span as the current span for highlighting
            current_span_dict = root_spans[0]
            current_span_id = current_span_dict["span_id"]
        else:
            # Fallback to first span if no root spans found
            current_span_dict = span_dicts[0]
            current_span_id = current_span_dict["span_id"]

        # Build hierarchical structure for all spans
        # Since we may have multiple root spans, we'll build a forest
        trace_tree = self._build_forest(span_dicts)

        # Generate HTML visualization
        return generate_span_html(
            span_dict=current_span_dict,
            all_spans=span_dicts,
            trace_tree=trace_tree,
            current_span_id=current_span_id,
        )

    def _build_forest(self, span_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build a forest of trees from multiple root spans.

        Handles the case where spans come from different traces by
        creating a virtual root that contains all root spans as children.

        :param span_dicts: List of span dictionaries
        :return: Tree structure with all root spans
        """
        if not span_dicts:
            return {}

        # Build span dict keyed by span_id
        span_dict = {span["span_id"]: span for span in span_dicts}

        # Find root spans (no parent)
        root_spans = [span for span in span_dicts if span.get("parent_span_id") is None]

        def build_tree(span: Dict[str, Any]) -> Dict[str, Any]:
            """Recursively build tree structure."""
            children = [
                build_tree(span_dict[child_id])
                for child_id, child_span in span_dict.items()
                if child_span.get("parent_span_id") == span["span_id"]
            ]
            result = span.copy()
            result["children"] = children
            return result

        # If we have multiple root spans, create a virtual root
        if len(root_spans) > 1:
            # Create a virtual root that contains all root spans
            virtual_root = {
                "span_id": "__virtual_root__",
                "trace_id": "__virtual_root__",
                "parent_span_id": None,
                "operation_name": f"Multiple Traces ({len(root_spans)} roots)",
                "start_time": min(
                    s.get("start_time", "") for s in root_spans if s.get("start_time")
                ),
                "end_time": max(
                    s.get("end_time", "") for s in root_spans if s.get("end_time")
                ),
                "duration_ms": None,
                "attributes": {},
                "events": [],
                "status": "completed",
                "error_message": None,
                "children": [build_tree(root) for root in root_spans],
            }
            return virtual_root
        elif root_spans:
            # Single root span
            return build_tree(root_spans[0])
        else:
            # No root spans, return empty
            return {}

    def filter(self, **attribute_filters) -> "SpanList":
        """Filter spans by attributes.

        :param attribute_filters: Attribute filters (e.g., category="test")
        :return: New SpanList with filtered spans
        """
        filtered = []
        for span in self._spans:
            match = True
            for key, value in attribute_filters.items():
                if span.attributes.get(key) != value:
                    match = False
                    break
            if match:
                filtered.append(span)
        return SpanList(filtered)

    def group_by_trace(self) -> Dict[str, "SpanList"]:
        """Group spans by trace_id.

        :return: Dictionary mapping trace_id to SpanList
        """
        grouped = {}
        for span in self._spans:
            trace_id = span.trace_id
            if trace_id not in grouped:
                grouped[trace_id] = []
            grouped[trace_id].append(span)
        return {trace_id: SpanList(spans) for trace_id, spans in grouped.items()}


def escape_html(text: str) -> str:
    """Escape HTML special characters for security.

    :param text: Raw text string (may contain HTML special chars)
    :return: HTML-escaped string
    """
    if text is None:
        return ""
    import html

    return html.escape(str(text))


def format_timestamp(timestamp_str: str) -> str:
    """Format timestamps for display.

    :param timestamp_str: ISO format timestamp string
    :return: Human-readable timestamp string
    """
    if not timestamp_str:
        return ""
    from datetime import datetime

    try:
        dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        return dt.strftime("%H:%M:%S.%f")[:-3]  # Show milliseconds
    except (ValueError, AttributeError):
        return str(timestamp_str)


def format_duration(duration_ms: Optional[float], is_in_progress: bool = False) -> str:
    """Format duration (ms, s, etc.), handle None for in-progress spans.

    :param duration_ms: Duration in milliseconds (or None for in-progress spans)
    :param is_in_progress: Whether span is still in progress
    :return: Formatted duration string (e.g., "123ms", "1.5s", "in progress")
    """
    if is_in_progress or duration_ms is None:
        return "in progress"
    if duration_ms < 1000:
        return f"{duration_ms:.0f}ms"
    elif duration_ms < 60000:
        return f"{duration_ms / 1000:.2f}s"
    else:
        minutes = duration_ms / 60000
        return f"{minutes:.2f}min"


def get_span_color(status: str) -> str:
    """Determine color based on status (Nord colors).

    :param status: Status string ("completed", "error", "started")
    :return: Nord color hex code
    """
    color_map = {
        "completed": "#A3BE8C",  # aurora green
        "error": "#BF616A",  # aurora red
        "started": "#5E81AC",  # frost blue
    }
    return color_map.get(status, "#5E81AC")


def calculate_nesting_level(
    span: Dict[str, Any], span_dict: Dict[str, Dict[str, Any]]
) -> int:
    """Calculate indentation level for a span.

    :param span: Span dictionary
    :param span_dict: Dict of all spans keyed by span_id
    :return: Nesting level (0 for root spans, 1+ for nested)
    """
    level = 0
    current_span = span
    visited = set()

    while (
        current_span.get("parent_span_id")
        and current_span["parent_span_id"] not in visited
    ):
        visited.add(current_span["span_id"])
        parent_id = current_span["parent_span_id"]
        if parent_id in span_dict:
            current_span = span_dict[parent_id]
            level += 1
        else:
            break

    return level


def build_hierarchy(spans: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Organize spans into tree structure.

    :param spans: List of span dictionaries (from database or converted from Span objects)
    :return: Hierarchical tree structure with parent-child relationships
    """
    if not spans:
        return {}

    # Build span dict keyed by span_id
    span_dict = {span["span_id"]: span for span in spans}

    # Find root spans (no parent)
    root_spans = [span for span in spans if span.get("parent_span_id") is None]

    def build_tree(span: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively build tree structure."""
        children = [
            build_tree(span_dict[child_id])
            for child_id, child_span in span_dict.items()
            if child_span.get("parent_span_id") == span["span_id"]
        ]
        result = span.copy()
        result["children"] = children
        return result

    # Return first root span (or build tree for all roots)
    if root_spans:
        return build_tree(root_spans[0])
    return {}


def generate_span_html(
    span_dict: Dict[str, Any],
    all_spans: List[Dict[str, Any]],
    trace_tree: Dict[str, Any],
    current_span_id: str,
    page: int = 1,
    per_page: int = 25,
) -> str:
    """Generate complete HTML string with pagination.

    :param span_dict: Current span as dictionary
    :param all_spans: List of all span dictionaries (from database + converted objects)
    :param trace_tree: Hierarchical tree structure
    :param current_span_id: ID of span to highlight
    :param page: Current page number for pagination
    :param per_page: Number of spans per page
    :return: Complete HTML string with embedded CSS and JavaScript
    """
    # Sort spans by start_time
    sorted_spans = sorted(
        all_spans,
        key=lambda s: s.get("start_time") or "",
    )

    # Pagination - we'll render all spans client-side and paginate in JavaScript
    total_spans = len(sorted_spans)
    total_pages = (total_spans + per_page - 1) // per_page

    # Build span dict for quick lookup
    span_dict_lookup = {s["span_id"]: s for s in all_spans}

    # Generate HTML for ALL span list items (will be paginated client-side)
    span_items_html = []
    for idx, span in enumerate(sorted_spans):
        span_id = span["span_id"]
        nesting_level = calculate_nesting_level(span, span_dict_lookup)
        is_current = span_id == current_span_id
        is_in_progress = (
            span.get("status") == "started" and span.get("end_time") is None
        )
        color = get_span_color(span.get("status", "started"))
        operation_name = escape_html(span.get("operation_name", ""))
        timestamp = format_timestamp(span.get("start_time", ""))
        duration = format_duration(span.get("duration_ms"), is_in_progress)

        indent_px = nesting_level * 20
        highlight_class = "current-span" if is_current else ""
        status_class = "span-in-progress" if is_in_progress else ""

        # Initially show spans on first page, hide others
        initial_display = "block" if idx < per_page else "none"

        span_items_html.append(
            f"""
            <div class="span-item {highlight_class} {status_class}"
                 data-span-id="{escape_html(span_id)}"
                 style="display: {initial_display}; margin-left: {indent_px}px; border-left: 3px solid {color};"
                 onclick="selectSpan('{escape_html(span_id)}')">
                <div class="span-header">
                    <span class="span-time">{timestamp}</span>
                    <span class="span-name">{operation_name}</span>
                    <span class="span-duration">{duration}</span>
                </div>
            </div>
            """
        )

    # Generate details panel HTML for current span
    details_html = generate_span_details_html(span_dict)

    # Generate pagination controls
    pagination_html = generate_pagination_html(page, total_pages, total_spans, per_page)

    # Complete HTML with embedded CSS and JavaScript
    html = f"""
    <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 0;
                padding: 0;
            }}
            .span-container {{
                display: grid;
                grid-template-columns: 13fr 7fr;
                gap: 0.5rem;
                background: white;
                padding: 1rem;
                border-radius: 8px;
                max-width: 100%;
                margin: 0;
            }}
            .span-timeline {{
                background: #E5E9F0;
                border-radius: 6px;
                padding: 1rem 1rem 1rem 0.5rem;
                overflow-y: auto;
            }}
            .span-list {{
                display: flex;
                flex-direction: column;
                gap: 0.5rem;
            }}
            .span-item {{
                background: white;
                padding: 0.75rem;
                border-radius: 4px;
                cursor: pointer;
                transition: all 0.2s;
                border-left: 3px solid #5E81AC;
            }}
            .span-item:hover {{
                background: #ECEFF4;
                transform: translateX(2px);
            }}
            .span-item.current-span {{
                background: #D8DEE9;
                font-weight: 600;
            }}
            .span-item.span-in-progress {{
                border-left-style: dashed;
            }}
            .span-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 1rem;
            }}
            .span-time {{
                color: #2E3440;
                font-size: 0.85rem;
                font-family: monospace;
            }}
            .span-name {{
                color: #2E3440;
                font-weight: 500;
                flex: 1;
            }}
            .span-duration {{
                color: #5E81AC;
                font-size: 0.85rem;
                font-weight: 600;
            }}
            .span-details {{
                background: #E5E9F0;
                border-radius: 6px;
                padding: 1rem 1.5rem 1rem 1rem;
                overflow-y: auto;
            }}
            .details-section {{
                margin-bottom: 1.5rem;
            }}
            .details-section h3 {{
                color: #2E3440;
                font-size: 1rem;
                margin-bottom: 0.5rem;
                border-bottom: 2px solid #D8DEE9;
                padding-bottom: 0.25rem;
            }}
            .details-table {{
                width: 100%;
                border-collapse: collapse;
            }}
            .details-table td {{
                padding: 0.5rem;
                border-bottom: 1px solid #D8DEE9;
            }}
            .details-table td:first-child {{
                font-weight: 600;
                color: #2E3440;
                width: 30%;
            }}
            .details-table td:last-child {{
                color: #2E3440;
                font-family: monospace;
                font-size: 0.9rem;
            }}
            .pagination-controls {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-top: 1rem;
                padding-top: 1rem;
                border-top: 1px solid #D8DEE9;
            }}
            .pagination-info {{
                color: #2E3440;
                font-size: 0.9rem;
            }}
            .pagination-buttons {{
                display: flex;
                gap: 0.5rem;
            }}
            .pagination-btn {{
                padding: 0.5rem 1rem;
                background: #5E81AC;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 0.9rem;
            }}
            .pagination-btn:hover {{
                background: #4C6A8A;
            }}
            .pagination-btn:disabled {{
                background: #D8DEE9;
                color: #2E3440;
                cursor: not-allowed;
            }}
            .status-badge {{
                display: inline-block;
                padding: 0.25rem 0.5rem;
                border-radius: 4px;
                font-size: 0.85rem;
                font-weight: 600;
            }}
            .status-completed {{
                background: #A3BE8C;
                color: #2E3440;
            }}
            .status-error {{
                background: #BF616A;
                color: white;
            }}
            .status-started {{
                background: #5E81AC;
                color: white;
            }}
        </style>
        <div class="span-container">
            <div class="span-timeline">
                <div class="span-list">
                    {"".join(span_items_html)}
                </div>
                {pagination_html}
            </div>
            <div class="span-details" id="span-details-panel">
                {details_html}
            </div>
        </div>
        <script>
            // Embed span data for JavaScript access
            const spanData = {json.dumps(all_spans)};
            const perPage = {per_page};
            let currentPage = {page};
            const totalPages = {total_pages};
            const totalSpans = {total_spans};

            function escapeHtml(text) {{
                if (text === null || text === undefined) return "";
                const div = document.createElement('div');
                div.textContent = String(text);
                return div.innerHTML;
            }}

            function formatTimestamp(timestampStr) {{
                if (!timestampStr) return "";
                try {{
                    const dt = new Date(timestampStr.replace('Z', '+00:00'));
                    return dt.toLocaleTimeString('en-US', {{ hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit', fractionalSecondDigits: 3 }});
                }} catch (e) {{
                    return String(timestampStr);
                }}
            }}

            function formatDuration(durationMs, isInProgress) {{
                if (isInProgress || durationMs === null || durationMs === undefined) {{
                    return "in progress";
                }}
                if (durationMs < 1000) {{
                    return Math.round(durationMs) + "ms";
                }} else if (durationMs < 60000) {{
                    return (durationMs / 1000).toFixed(2) + "s";
                }} else {{
                    return (durationMs / 60000).toFixed(2) + "min";
                }}
            }}

            function generateDetailsHtml(span) {{
                const operationName = escapeHtml(span.operation_name || "");
                const status = span.status || "started";
                const statusClass = `status-${{status}}`;
                const statusBadge = `<span class="status-badge ${{statusClass}}">${{escapeHtml(status)}}</span>`;

                const startTime = formatTimestamp(span.start_time || "");
                const endTime = span.end_time ? formatTimestamp(span.end_time) : "in progress";
                const duration = formatDuration(span.duration_ms, status === "started");

                let attributesHtml = "";
                if (span.attributes && Object.keys(span.attributes).length > 0) {{
                    const attrsRows = Object.entries(span.attributes).map(([key, value]) => {{
                        return `<tr><td>${{escapeHtml(String(key))}}</td><td>${{escapeHtml(String(value))}}</td></tr>`;
                    }}).join("");
                    attributesHtml = `
                    <div class="details-section">
                        <h3>Attributes</h3>
                        <table class="details-table">
                            ${{attrsRows}}
                        </table>
                    </div>
                    `;
                }}

                let eventsHtml = "";
                if (span.events && span.events.length > 0) {{
                    const eventsRows = span.events.map(event => {{
                        const eventName = escapeHtml(event.name || "");
                        const eventTime = formatTimestamp(event.timestamp || "");
                        const eventData = escapeHtml(JSON.stringify(event.data || {{}}));
                        return `<tr><td>${{eventName}}</td><td>${{eventTime}}<br>${{eventData}}</td></tr>`;
                    }}).join("");
                    eventsHtml = `
                    <div class="details-section">
                        <h3>Events</h3>
                        <table class="details-table">
                            ${{eventsRows}}
                        </table>
                    </div>
                    `;
                }}

                let errorHtml = "";
                if (span.error_message) {{
                    const errorMsg = escapeHtml(span.error_message);
                    errorHtml = `
                    <div class="details-section">
                        <h3>Error</h3>
                        <div style="color: #BF616A; font-family: monospace; font-size: 0.9rem;">${{errorMsg}}</div>
                    </div>
                    `;
                }}

                return `
                    <div class="details-section">
                        <h3>Operation</h3>
                        <div style="font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">${{operationName}}</div>
                        ${{statusBadge}}
                    </div>
                    <div class="details-section">
                        <h3>Timing</h3>
                        <table class="details-table">
                            <tr><td>Start Time</td><td>${{startTime}}</td></tr>
                            <tr><td>End Time</td><td>${{endTime}}</td></tr>
                            <tr><td>Duration</td><td>${{duration}}</td></tr>
                        </table>
                    </div>
                    ${{attributesHtml}}
                    ${{eventsHtml}}
                    ${{errorHtml}}
                `;
            }}

            function selectSpan(spanId) {{
                // Update all span items
                document.querySelectorAll('.span-item').forEach(item => {{
                    item.classList.remove('current-span');
                }});

                // Highlight selected span
                const selectedItem = document.querySelector(`[data-span-id="${{spanId}}"]`);
                if (selectedItem) {{
                    selectedItem.classList.add('current-span');
                    selectedItem.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                }}

                // Find span data and update details panel
                const span = spanData.find(s => s.span_id === spanId);
                if (span) {{
                    const detailsPanel = document.getElementById('span-details-panel');
                    if (detailsPanel) {{
                        detailsPanel.innerHTML = generateDetailsHtml(span);
                    }}
                }}
            }}

            function renderSpans(page) {{
                const startIdx = (page - 1) * perPage;
                const endIdx = startIdx + perPage;

                // Show/hide spans based on current page
                const spanItems = document.querySelectorAll('.span-item');
                spanItems.forEach((item, index) => {{
                    if (index >= startIdx && index < endIdx) {{
                        item.style.display = 'block';
                    }} else {{
                        item.style.display = 'none';
                    }}
                }});

                // Update pagination controls
                const startDisplay = startIdx + 1;
                const endDisplay = Math.min(endIdx, totalSpans);
                document.getElementById('pagination-info').textContent =
                    `Showing ${{startDisplay}}-${{endDisplay}} of ${{totalSpans}} spans`;
                document.getElementById('page-info').textContent =
                    `Page ${{page}} of ${{totalPages}}`;

                // Update button states
                const prevBtn = document.getElementById('prev-btn');
                const nextBtn = document.getElementById('next-btn');
                if (prevBtn) {{
                    prevBtn.disabled = page === 1;
                    prevBtn.onclick = () => changePage(page - 1);
                }}
                if (nextBtn) {{
                    nextBtn.disabled = page === totalPages;
                    nextBtn.onclick = () => changePage(page + 1);
                }}
            }}

            function changePage(newPage) {{
                if (newPage < 1 || newPage > totalPages) return;
                currentPage = newPage;
                renderSpans(currentPage);
            }}

            // Initialize pagination on load
            renderSpans(currentPage);
        </script>
    </div>
    """
    return html


def generate_span_details_html(span_dict: Dict[str, Any]) -> str:
    """Generate HTML for span details panel.

    :param span_dict: Span dictionary
    :return: HTML string for details panel
    """
    operation_name = escape_html(span_dict.get("operation_name", ""))
    status = span_dict.get("status", "started")
    status_class = f"status-{status}"
    status_badge = (
        f'<span class="status-badge {status_class}">{escape_html(status)}</span>'
    )

    start_time = format_timestamp(span_dict.get("start_time", ""))
    end_time = (
        format_timestamp(span_dict.get("end_time", ""))
        if span_dict.get("end_time")
        else "in progress"
    )
    duration = format_duration(span_dict.get("duration_ms"), status == "started")

    # Attributes section
    attributes_html = ""
    if span_dict.get("attributes"):
        attrs_rows = []
        for key, value in span_dict["attributes"].items():
            key_escaped = escape_html(str(key))
            value_escaped = escape_html(str(value))
            attrs_rows.append(
                f"<tr><td>{key_escaped}</td><td>{value_escaped}</td></tr>"
            )
        attributes_html = f"""
        <div class="details-section">
            <h3>Attributes</h3>
            <table class="details-table">
                {"".join(attrs_rows)}
            </table>
        </div>
        """

    # Events section
    events_html = ""
    if span_dict.get("events"):
        events_rows = []
        for event in span_dict["events"]:
            event_name = escape_html(event.get("name", ""))
            event_time = format_timestamp(event.get("timestamp", ""))
            event_data = escape_html(str(event.get("data", {})))
            events_rows.append(
                f"<tr><td>{event_name}</td><td>{event_time}<br>{event_data}</td></tr>"
            )
        events_html = f"""
        <div class="details-section">
            <h3>Events</h3>
            <table class="details-table">
                {"".join(events_rows)}
            </table>
        </div>
        """

    # Error section
    error_html = ""
    if span_dict.get("error_message"):
        error_msg = escape_html(span_dict["error_message"])
        error_html = f"""
        <div class="details-section">
            <h3>Error</h3>
            <div style="color: #BF616A; font-family: monospace; font-size: 0.9rem;">{error_msg}</div>
        </div>
        """

    return f"""
        <div class="details-section">
            <h3>Operation</h3>
            <div style="font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">{operation_name}</div>
            {status_badge}
        </div>
        <div class="details-section">
            <h3>Timing</h3>
            <table class="details-table">
                <tr><td>Start Time</td><td>{start_time}</td></tr>
                <tr><td>End Time</td><td>{end_time}</td></tr>
                <tr><td>Duration</td><td>{duration}</td></tr>
            </table>
        </div>
        {attributes_html}
        {events_html}
        {error_html}
    """


def generate_pagination_html(
    page: int, total_pages: int, total_spans: int, per_page: int = 25
) -> str:
    """Generate pagination controls HTML.

    :param page: Current page number
    :param total_pages: Total number of pages
    :param total_spans: Total number of spans
    :param per_page: Number of spans per page
    :return: HTML string for pagination controls
    """
    if total_pages <= 1:
        return ""

    prev_disabled = "disabled" if page == 1 else ""
    next_disabled = "disabled" if page == total_pages else ""

    start_idx = (page - 1) * per_page + 1
    end_idx = min(page * per_page, total_spans)

    return f"""
    <div class="pagination-controls" id="pagination-controls">
        <div class="pagination-info" id="pagination-info">
            Showing {start_idx}-{end_idx} of {total_spans} spans
        </div>
        <div class="pagination-buttons">
            <button class="pagination-btn" id="prev-btn" {prev_disabled} onclick="changePage(currentPage - 1)">Previous</button>
            <span style="padding: 0.5rem; color: #2E3440;" id="page-info">Page {page} of {total_pages}</span>
            <button class="pagination-btn" id="next-btn" {next_disabled} onclick="changePage(currentPage + 1)">Next</button>
        </div>
    </div>
    """
