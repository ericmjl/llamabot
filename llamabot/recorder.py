"""Prompt recorder class definition."""

import contextvars
import functools
import json
import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Dict, List, Union, Callable

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
from sqlalchemy.orm import sessionmaker, relationship

from llamabot.components.messages import BaseMessage
from llamabot.utils import find_or_set_db_path, get_object_name
from loguru import logger

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
        self.trace_id = trace_id or get_or_create_trace_id()
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

    def __enter__(self):
        """Enter span context - sets as current span."""
        # Store previous span to restore on exit
        self._previous_span = current_span_var.get(None)
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

            # Add function arguments as attributes (excluding sensitive ones)
            if args:
                span_obj["args_count"] = len(args)
            for key, value in kwargs.items():
                if key not in exclude_args:
                    span_obj[f"arg_{key}"] = _serialize_value(value)

            with span_obj:
                try:
                    result = func(*args, **kwargs)
                    if log_return:
                        span_obj.log("function_return", result=_serialize_value(result))
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
) -> List[Dict[str, Any]]:
    """Query spans from the database.

    :param trace_id: Filter by trace ID
    :param span_id: Filter by specific span ID
    :param operation_name: Filter by operation name
    :param start_time: Filter spans that started after this time
    :param end_time: Filter spans that ended before this time
    :param db_path: Database path. If None, uses find_or_set_db_path(None)
    :param attribute_filters: Additional attribute filters (e.g., model="gpt-4")
    :return: List of span dictionaries
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
            return filtered_result

        return result
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
    # Get all spans for trace_id
    spans = get_spans(trace_id=trace_id, db_path=db_path)

    if not spans:
        return {}

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
