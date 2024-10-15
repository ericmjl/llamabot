"""Prompt recorder class definition."""

import contextvars
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pyprojroot import here
from sqlalchemy import (
    Column,
    Connection,
    Engine,
    Float,
    Integer,
    String,
    Text,
    ForeignKey,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

from llamabot.components.messages import BaseMessage
from llamabot.utils import get_object_name

prompt_recorder_var = contextvars.ContextVar("prompt_recorder")


class PromptRecorder:
    """Prompt recorder to support recording of prompts and responses."""

    def __init__(self):
        self.prompts_and_responses = []

    def __enter__(self):
        """Enter the context manager.

        :returns: The prompt recorder.
        """
        prompt_recorder_var.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager.

        :param exc_type: The exception type.
        :param exc_val: The exception value.
        :param exc_tb: The exception traceback.
        """
        prompt_recorder_var.set(None)

    def log(self, prompt: str, response: str):
        """Log the prompt and response in chat history.

        :param prompt: The human prompt.
        :param response: A the response from the bot.
        """
        self.prompts_and_responses.append({"prompt": prompt, "response": response})

    def __repr__(self):
        """Return a string representation of the prompt recorder.

        :return: A string form of the prompts and responses as a dataframe.
        """
        import pandas as pd

        return pd.DataFrame(self.prompts_and_responses).__str__()

    def _repr_html_(self):
        """Return an HTML representation of the prompt recorder.

        :return: We delegate to the _repr_html_ method of the pandas DataFrame class.
        """
        return self.dataframe()._repr_html_()

    def dataframe(self):
        """Return a pandas DataFrame representation of the prompt recorder.

        :return: A pandas DataFrame representation of the prompt recorder.
        """
        import pandas as pd

        return pd.DataFrame(self.prompts_and_responses)

    def save(self, path: Path):
        """Save the prompt recorder to a path.

        :param path: The path to save the prompt recorder to.
        """
        path = Path(path)  # coerce to pathlib.Path
        with path.open("w+") as f:
            for prompt_and_response in self.prompts_and_responses:
                f.write(
                    f"**{prompt_and_response['prompt']}**\n\n{prompt_and_response['response']}\n\n"
                )

    def panel(self):
        """Return a panel representation of the prompt recorder.

        :return: A panel representation of the prompt recorder.
        """
        import panel as pn

        global index
        index = 0
        pn.extension()

        next_button = pn.widgets.Button(name=">")
        prev_button = pn.widgets.Button(name="<")

        buttons = pn.Row(prev_button, next_button)

        prompt_header = pn.pane.Markdown("# Prompt")
        prompt_display = pn.pane.Markdown(self.prompts_and_responses[index]["prompt"])

        prompt = pn.Column(prompt_header, prompt_display)

        response_header = pn.pane.Markdown("# Response")
        response_display = pn.pane.Markdown(
            self.prompts_and_responses[index]["response"]
        )
        response = pn.Column(response_header, response_display)

        display = pn.Row(prompt, response)

        def update_objects(index):
            """Update the prompt and response Markdown panes.

            :param index: The index of the prompt and response to update.
            """
            prompt_display.object = self.prompts_and_responses[index]["prompt"]
            response_display.object = self.prompts_and_responses[index]["response"]

        def next_button_callback(event):
            """Callback function for the next button.

            :param event: The click event.
            """
            global index
            index += 1
            if index > len(self.prompts_and_responses) - 1:
                index = len(self.prompts_and_responses) - 1

            update_objects(index)

        def prev_button_callback(event):
            """Callback function for the previous button.

            :param event: The click event.
            """
            global index
            index -= 1
            if index < 0:
                index = 0
            update_objects(index)

        next_button.on_click(next_button_callback)
        prev_button.on_click(prev_button_callback)

        return pn.Column(buttons, display)


def autorecord(prompt: str, response: str):
    """Record a prompt and response.

    This is intended to be called within every bot.
    If we are within a prompt recorder context,
    then the prompt recorder will record the prompt and response
    as specified in the function.

    :param prompt: The human prompt.
    :param response: A the response from the bot.
    """
    # Log the response.
    prompt_recorder: Optional[PromptRecorder] = prompt_recorder_var.get(None)
    if prompt_recorder:
        prompt_recorder.log(prompt, response)


# Design sqlite database for storing prompts and responses.
# Columns:
# - Python object name from globals()
# - Time stamp
# - Full message log as a JSON string of dictionaries.


Base = declarative_base()


class MessageLog(Base):
    """A log of a message exchange."""

    __tablename__ = "message_log"

    id = Column(Integer, primary_key=True)
    object_name = Column(String)
    timestamp = Column(String)
    message_log = Column(Text)
    model_name = Column(String)
    temperature = Column(Float, nullable=True)


def upgrade_database(engine: Engine):
    """Upgrade the database schema."""
    Base.metadata.create_all(engine)

    with engine.connect() as connection:
        inspector = inspect(engine)
        for table in [MessageLog, Prompt]:
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


def sqlite_log(
    obj: Any,
    messages: list[BaseMessage],
    prompt_hash: Optional[str] = None,
    db_path: Optional[Path] = None,
):
    """Log messages to the sqlite database for further analysis.

    :param obj: The object to log the messages for.
    :param messages: The messages to log.
    :param prompt_hash: The hash of the prompt.
    :param db_path: The path to the database to use.
        If not specified, defaults to ~/.llamabot/message_log.db
    """
    # Set up the database path
    if db_path is None:
        try:
            repo_root = here()
            db_path = repo_root / "message_log.db"

            # Ensure message_log.db is in .gitignore
            gitignore_path = repo_root / ".gitignore"
            if gitignore_path.exists():
                with open(gitignore_path, "r+") as f:
                    content = f.read()
                    if "message_log.db" not in content:
                        f.write("\n# SQLite database\nmessage_log.db\n")
            else:
                with open(gitignore_path, "w") as f:
                    f.write("# SQLite database\nmessage_log.db\n")
        except Exception:
            # If we're not in a git repo, use the home directory
            db_path = Path.home() / ".llamabot" / "message_log.db"

    # Create the engine and session
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    upgrade_database(engine)  # Add this line to ensure the database is upgraded
    Session = sessionmaker(bind=engine)
    session = Session()

    # Get the object name
    object_name = get_object_name(obj)

    # Get the current timestamp
    timestamp = datetime.now().isoformat()

    # Convert messages to a JSON string, including prompt_hash
    message_log = json.dumps(
        [
            {
                "role": message.role,
                "content": message.content,
                "prompt_hash": message.prompt_hash,
            }
            for message in messages
        ]
    )

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

    # Close the session
    session.close()


class Prompt(Base):
    """A version-controlled prompt template."""

    __tablename__ = "prompts"

    id = Column(Integer, primary_key=True)
    hash = Column(String, unique=True, index=True)
    template = Column(Text)
    function_name = Column(String)  # Add this line
    previous_version_id = Column(Integer, ForeignKey("prompts.id"), nullable=True)
    previous_version = relationship("Prompt", remote_side=[id])


def hash_template(template: str) -> str:
    """Hash a prompt template."""
    return hashlib.sha256(template.encode()).hexdigest()


def store_prompt_version(
    session, template: str, function_name: str, previous_hash: Optional[str] = None
):
    """Store a new prompt version in the database."""
    template_hash = hash_template(template)

    existing_prompt = session.query(Prompt).filter_by(hash=template_hash).first()
    if existing_prompt:
        return existing_prompt

    if previous_hash:
        previous_version = session.query(Prompt).filter_by(hash=previous_hash).first()
        previous_version_id = previous_version.id if previous_version else None
    else:
        previous_version_id = None

    new_prompt = Prompt(
        hash=template_hash,
        template=template,
        function_name=function_name,  # Add this line
        previous_version_id=previous_version_id,
    )
    session.add(new_prompt)
    session.commit()
    return new_prompt
