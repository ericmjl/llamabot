"""Tests for the prompt decorator."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from llamabot.components.messages import BaseMessage
from llamabot.prompt_manager import prompt, version_prompt
from llamabot.recorder import hash_template, Base, Prompt

# Setup a test database
TEST_DB_PATH = "sqlite:///:memory:"


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
        """This function takes two arguments: {{a}} and {{b}}. # noqa: DAR101"""

    with pytest.raises(ValueError):
        test_func(a=3)


def test_version_prompt(db_session, monkeypatch):
    """
    Test the version_prompt function for creating and updating prompt versions.

    This test covers three scenarios:
    1. Creating a new prompt
    2. Updating an existing prompt
    3. Attempting to create a duplicate prompt (which should return the existing one)

    :param db_session: The database session fixture.
    :param monkeypatch: PyTest's monkeypatch fixture for mocking.
    """

    # Mock the create_engine function to use our test database
    def mock_create_engine(url):
        return create_engine(TEST_DB_PATH)

    monkeypatch.setattr("sqlalchemy.create_engine", mock_create_engine)

    # Test case 1: New prompt
    template1 = "This is a test prompt template"
    function_name = "test_function"

    hash1 = version_prompt(template1, function_name)

    # Check if a new prompt was created
    prompt1 = db_session.query(Prompt).filter_by(hash=hash1).first()
    assert prompt1 is not None
    assert prompt1.function_name == function_name
    assert prompt1.template == template1
    assert prompt1.previous_version is None

    # Test case 2: Updated prompt
    template2 = "This is an updated test prompt template"

    hash2 = version_prompt(template2, function_name)

    # Check if a new version was created
    prompt2 = db_session.query(Prompt).filter_by(hash=hash2).first()
    assert prompt2 is not None
    assert prompt2.function_name == function_name
    assert prompt2.template == template2
    assert prompt2.previous_version.id == prompt1.id

    # Test case 3: Same prompt (no change)
    hash3 = version_prompt(template2, function_name)

    # Check if no new version was created
    assert hash3 == hash2
    prompt_count = db_session.query(Prompt).count()
    assert prompt_count == 2  # Still only two prompts in the database


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
