"""
This module provides a set of tests for the PromptRecorder
and autorecord functions in the llamabot.recorder module.
"""

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from llamabot.recorder import (
    MessageLog,
    Base,
    upgrade_database,
    add_column,
    sqlite_log,
)
from llamabot.components.messages import BaseMessage
from unittest.mock import Mock


@pytest.fixture
def engine():
    """Create a temporary in-memory SQLite database for testing."""
    return create_engine("sqlite:///:memory:")


@pytest.fixture
def session(engine):
    """Create a new session for testing."""
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


def test_message_log_creation(session):
    """Test that a MessageLog instance can be created and saved to the database."""
    log = MessageLog(
        object_name="test_object",
        timestamp="2023-01-01T00:00:00",
        message_log='{"message": "test"}',
    )
    session.add(log)
    session.commit()

    retrieved_log = session.query(MessageLog).first()
    assert retrieved_log.object_name == "test_object"
    assert retrieved_log.timestamp == "2023-01-01T00:00:00"
    assert retrieved_log.message_log == '{"message": "test"}'


def test_upgrade_database(engine):
    """Test that upgrade_database creates the MessageLog table if it doesn't exist."""
    # Drop the table if it exists
    MessageLog.__table__.drop(engine, checkfirst=True)

    # Run upgrade_database
    upgrade_database(engine)

    # Check if the table now exists
    inspector = inspect(engine)
    assert inspector.has_table("message_log")

    # Check if all columns exist
    columns = inspector.get_columns("message_log")
    column_names = [c["name"] for c in columns]
    expected_columns = ["id", "object_name", "timestamp", "message_log"]
    for col in expected_columns:
        assert col in column_names


def test_add_column(engine):
    """Test that add_column adds a new column to an existing table."""
    from sqlalchemy import Column, String

    # Ensure the table exists
    Base.metadata.create_all(engine)

    # Add a new column
    new_column = Column("new_test_column", String)
    with engine.connect() as connection:
        add_column(connection, MessageLog.__tablename__, new_column)

    # Check if the new column exists
    inspector = inspect(engine)
    columns = inspector.get_columns(MessageLog.__tablename__)
    assert any(col["name"] == "new_test_column" for col in columns)


def test_sqlite_log(engine, monkeypatch, tmp_path):
    """Test that sqlite_log correctly logs messages to the database in a temporary directory."""
    # Create a temporary directory for the test database
    temp_db_path = tmp_path / "test_message_log.db"

    # Mock the get_object_name function to return a predictable name
    def mock_get_object_name(obj):
        return "test_object"

    monkeypatch.setattr("llamabot.recorder.get_object_name", mock_get_object_name)

    # Create a mock test object with model_name and temperature attributes
    test_obj = Mock()
    test_obj.model_name = "test_model"
    test_obj.temperature = 0.7

    test_messages = [
        BaseMessage(role="user", content="Hello"),
        BaseMessage(role="assistant", content="Hi there!"),
    ]

    # Call sqlite_log with the temporary database path
    sqlite_log(test_obj, test_messages, db_path=temp_db_path)

    # Create an engine for the temporary database
    temp_engine = create_engine(f"sqlite:///{temp_db_path}")
    Session = sessionmaker(bind=temp_engine)
    session = Session()

    # Query the database to check if the log was saved
    log_entry = session.query(MessageLog).first()

    assert log_entry is not None
    assert log_entry.object_name == "test_object"
    assert "Hello" in log_entry.message_log
    assert "Hi there!" in log_entry.message_log
    assert log_entry.model_name == "test_model"
    assert log_entry.temperature == 0.7

    # Close the session and dispose of the engine
    session.close()
    temp_engine.dispose()

    # Verify that the database file was created
    assert temp_db_path.exists()

    # The temporary directory and its contents will be automatically cleaned up after the test
