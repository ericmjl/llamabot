"""Prompt recorder class definition."""

import contextvars
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Dict

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
    # Create all tables from Base (now includes Runs)
    Base.metadata.create_all(engine)

    # Add new columns to existing tables if needed
    with engine.connect() as connection:
        inspector = inspect(engine)
        for table in [MessageLog, Prompt, Runs]:
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

        # Create a new MessageLog instance
        new_log = MessageLog(
            object_name=object_name,
            timestamp=timestamp,
            message_log=message_log,
            model_name=obj.model_name,
            temperature=obj.temperature if hasattr(obj, "temperature") else None,
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
