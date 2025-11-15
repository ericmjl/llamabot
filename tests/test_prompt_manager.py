"""Tests for the prompt decorator."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from llamabot.components.messages import BaseMessage
from llamabot.prompt_manager import prompt, version_prompt
from llamabot.recorder import hash_template, Base, Prompt
import logging
from llamabot import user
from llamabot.components.messages import HumanMessage, ImageMessage

# Setup a test database
TEST_DB_PATH = "sqlite:////tmp/test_message_log.db"


@pytest.fixture(scope="function")
def db_session():
    """
    Create a new database session for a test.

    This fixture sets up a new in-memory SQLite database for each test function,
    creates all tables, yields a session for the test to use, and then tears down
    the database after the test is complete.

    :yield: An SQLAlchemy session connected to an in-memory test database.
    """
    engine = create_engine(TEST_DB_PATH)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    Base.metadata.drop_all(engine)


def test_prompt_no_args():
    """Test that the decorated prompt function returns the correct string."""

    @prompt(role="system")
    def test_func():
        """This is a test function."""

    assert test_func() == BaseMessage(
        role="system",
        content=test_func.__doc__,
        prompt_hash=hash_template(test_func.__doc__),
    )


def test_prompt_with_args():
    """Test that the decorated prompt function returns the correct string with args."""

    @prompt(role="system")
    def test_func(a, b):
        """This function takes two arguments: {{a}} and {{b}}. # noqa: DAR101"""

    assert test_func(1, 2) == BaseMessage(
        role="system",
        content="This function takes two arguments: 1 and 2. # noqa: DAR101",
        prompt_hash=hash_template(test_func.__doc__),
    )


def test_prompt_with_kwargs():
    """Test that the decorated prompt function
    returns the correct string with kwargs."""

    @prompt(role="system")
    def test_func(a=1, b=2):
        """This function takes two arguments: {{a}} and {{b}}. # noqa: DAR101"""

    assert test_func(a=3, b=4) == BaseMessage(
        role="system",
        content="This function takes two arguments: 3 and 4. # noqa: DAR101",
        prompt_hash=hash_template(test_func.__doc__),
    )


def test_prompt_with_args_and_kwargs():
    """Test that the decorated prompt function
    returns the correct string with args and kwargs."""

    @prompt(role="system")
    def test_func(a, b=2):
        """This function takes two arguments: {{a}} and {{b}}. # noqa: DAR101"""

    assert test_func(1, b=3) == BaseMessage(
        role="system",
        content="This function takes two arguments: 1 and 3. # noqa: DAR101",
        prompt_hash=hash_template(test_func.__doc__),
    )


def test_prompt_with_missing_kwargs():
    """Test that the decorated prompt function
    raises an error when kwargs are missing."""

    @prompt(role="system")
    def test_func(a=1, b=2):
        """This function takes two arguments: {{a}}, {{b}}, and {{c}}. # noqa: DAR101"""

    with pytest.raises(ValueError):
        test_func(a=3)


def test_version_prompt(caplog):
    """
    Test the version_prompt function for creating and updating prompt versions.

    This test covers three scenarios:
    1. Creating a new prompt
    2. Updating an existing prompt
    3. Attempting to create a duplicate prompt (which should return the existing one)

    :param caplog: PyTest's caplog fixture for capturing logs.
    """
    caplog.set_level(logging.DEBUG)

    # Test case 1: New prompt
    template1 = "This is a test prompt template"
    function_name = "test_function"

    hash1 = version_prompt(template1, function_name, db_path=TEST_DB_PATH)
    print(f"Generated hash: {hash1}")
    print("Captured logs:")
    print(caplog.text)

    # Create a session to verify the database contents
    engine = create_engine(TEST_DB_PATH)
    Base.metadata.create_all(engine)  # Create tables
    Session = sessionmaker(bind=engine)
    session = Session()

    # Check if a new prompt was created
    prompt1 = session.query(Prompt).filter_by(hash=hash1).first()
    assert prompt1 is not None, "Prompt was not created in the database"
    assert (
        prompt1.function_name == function_name
    ), f"Expected function name {function_name}, got {prompt1.function_name}"
    assert (
        prompt1.template == template1
    ), f"Expected template {template1}, got {prompt1.template}"
    assert prompt1.previous_version is None, "Expected previous_version to be None"

    # Test case 2: Updated prompt
    template2 = "This is an updated test prompt template"

    hash2 = version_prompt(template2, function_name, db_path=TEST_DB_PATH)
    print(f"Generated hash for updated prompt: {hash2}")

    # Check if a new version was created
    prompt2 = session.query(Prompt).filter_by(hash=hash2).first()
    assert prompt2 is not None, "Updated prompt was not created in the database"
    assert (
        prompt2.function_name == function_name
    ), f"Expected function name {function_name}, got {prompt2.function_name}"
    assert (
        prompt2.template == template2
    ), f"Expected template {template2}, got {prompt2.template}"
    assert (
        prompt2.previous_version.id == prompt1.id
    ), f"Expected previous_version.id to be {prompt1.id}, got {prompt2.previous_version.id if prompt2.previous_version else None}"

    # Test case 3: Same prompt (no change)
    hash3 = version_prompt(template2, function_name, db_path=TEST_DB_PATH)
    print(f"Generated hash for duplicate prompt: {hash3}")

    # Check if no new version was created
    assert hash3 == hash2, f"Expected hash3 ({hash3}) to be equal to hash2 ({hash2})"
    prompt_count = session.query(Prompt).count()
    assert prompt_count == 2, f"Expected 2 prompts in the database, got {prompt_count}"

    # Print all prompts in the database for debugging
    all_prompts = session.query(Prompt).all()
    for individual_prompt in all_prompts:
        print(
            f"Prompt: id={individual_prompt.id}, hash={individual_prompt.hash}, function_name={individual_prompt.function_name}, template={individual_prompt.template}"
        )

    session.close()


def test_version_prompt_error_handling(db_session, monkeypatch):
    """
    Test error handling in the version_prompt function.

    This test simulates a database error to ensure that the function
    properly raises an exception when encountering database issues.

    :param db_session: The database session fixture.
    :param monkeypatch: PyTest's monkeypatch fixture for mocking.
    """

    # Mock the create_engine function to use our test database
    def mock_create_engine(url):
        return create_engine(TEST_DB_PATH)

    monkeypatch.setattr("sqlalchemy.create_engine", mock_create_engine)

    # Test case: Database error
    def mock_query_error(*args, **kwargs):
        raise Exception("Database error")

    monkeypatch.setattr("sqlalchemy.orm.Session.query", mock_query_error)

    with pytest.raises(Exception):
        version_prompt("Test template", "test_function")


def test_user_message_with_string():
    """Test that user() correctly creates a HumanMessage from a string.

    :param None: No parameters needed.
    """
    message = user("Hello, world!")
    assert isinstance(message, HumanMessage)
    assert message.content == "Hello, world!"


def test_user_message_with_text_file(tmp_path):
    """Test that user() correctly creates a HumanMessage from a text file.

    :param tmp_path: Pytest fixture providing a temporary directory path.
    """
    # Create a temporary text file
    text_file = tmp_path / "test.txt"
    text_file.write_text("Hello from file!")

    message = user(text_file)
    assert isinstance(message, HumanMessage)
    assert message.content == "Hello from file!"


def test_user_message_with_image_file(tmp_path):
    """Test that user() correctly creates an ImageMessage from an image file.

    :param tmp_path: Pytest fixture providing a temporary directory path.
    """
    # Create a dummy image file
    image_file = tmp_path / "test.png"
    image_file.write_bytes(b"fake image content")

    message = user(image_file)
    assert isinstance(message, ImageMessage)
    assert message.content == "ZmFrZSBpbWFnZSBjb250ZW50"
