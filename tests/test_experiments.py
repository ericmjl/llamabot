"""Tests for the experiments module.

This module contains tests for the experiment tracking functionality,
including metric recording, message logging, and prompt tracking.
"""

from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from llamabot.prompt_manager import prompt
from llamabot.experiments import Experiment, metric
from llamabot.experiments import ExperimentRun, Base
from llamabot.recorder import sqlite_log
from llamabot.components.messages import BaseMessage
from llamabot.bot.simplebot import SimpleBot


@pytest.fixture
def db_path(tmp_path) -> Path:
    """Create a temporary database path.

    :param tmp_path: pytest fixture for temporary directory
    :return: Path to test database
    """
    return tmp_path / "test.db"


@pytest.fixture
def engine(db_path):
    """Create a test database engine.

    :param db_path: Path to test database
    :return: SQLAlchemy engine
    """
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def session(engine):
    """Create a test database session.

    :param engine: SQLAlchemy engine
    :return: SQLAlchemy session
    """
    Session = sessionmaker(bind=engine)
    return Session()


def test_metric_recording(db_path):
    """Test that metrics are properly recorded in experiment runs."""

    @metric
    def test_metric(x: int) -> float:
        """Test metric that returns a constant.

        :param x: Input value (unused)
        :return: Constant value of 42.0
        """
        return 42.0

    with Experiment("test_metrics", db_path=db_path) as exp:
        value = test_metric(0)
        assert value == 42.0

        # Check that the metric was recorded
        assert exp.run is not None
        assert "test_metric" in exp.run.run_data["metrics"]
        assert exp.run.run_data["metrics"]["test_metric"]["value"] == 42.0


def test_message_logging(db_path):
    """Test that message logs are properly linked to experiment runs."""

    # Create a SimpleBot instance for testing
    test_bot = SimpleBot(system_prompt="Test system prompt")

    with Experiment("test_messages", db_path=db_path) as exp:
        messages = [
            BaseMessage(role="user", content="Hello"),
            BaseMessage(role="assistant", content="Hi there!"),
        ]

        # Log messages using the test_bot instead of object()
        log_id = sqlite_log(test_bot, messages, db_path)

        # Check that the message log was linked
        assert exp.run is not None
        assert log_id in exp.run.run_data["message_log_ids"]


def test_prompt_tracking(db_path):
    """Test that prompts are properly tracked in experiment runs."""

    @prompt("system")
    def test_prompt(x: str) -> str:
        """Test prompt with parameter {{ x }}."""
        return f"Test prompt with {x}"

    with Experiment("test_prompts", db_path=db_path) as exp:
        _ = test_prompt("hello")

        # Check that the prompt was tracked
        assert exp.run is not None
        assert len(exp.run.run_data["prompts"]) == 1
        prompt_info = exp.run.run_data["prompts"][0]
        assert "hash" in prompt_info
        assert isinstance(prompt_info["hash"], str)


def test_experiment_metadata(db_path):
    """Test that experiment metadata is properly stored."""
    metadata = {"test_key": "test_value"}

    with Experiment("test_metadata", metadata=metadata, db_path=db_path) as exp:
        assert exp.run is not None
        assert exp.run.run_data["metadata"] == metadata


def test_multiple_runs(db_path):
    """Test that multiple runs are properly tracked."""
    engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)

    # First run
    with Experiment("multiple_runs", db_path=db_path):
        pass

    # Second run
    with Experiment("multiple_runs", db_path=db_path):
        pass

    # Check that both runs were recorded
    session = Session()
    runs = (
        session.execute(
            select(ExperimentRun).where(
                ExperimentRun.experiment_name == "multiple_runs"
            )
        )
        .scalars()
        .all()
    )

    assert len(runs) == 2
    session.close()


def test_experiment_cleanup(db_path):
    """Test that experiment resources are properly cleaned up."""
    exp = Experiment("cleanup_test", db_path=db_path)

    with exp:
        assert exp.run is not None
        assert exp.session is not None

    # After context exit
    assert exp.run is None

    # Try to use the session (should raise an error)
    with pytest.raises(Exception):
        exp.session.query(ExperimentRun).all()


def test_experiment_exception_handling(db_path):
    """Test that experiments handle exceptions properly."""
    exp = Experiment("exception_test", db_path=db_path)

    with pytest.raises(ValueError):
        with exp:
            raise ValueError("Test exception")

    # Session should be closed
    with pytest.raises(Exception):
        exp.session.query(ExperimentRun).all()


def test_run_data_structure(db_path):
    """Test that the run_data JSON structure is correct."""
    with Experiment("structure_test", db_path=db_path) as exp:
        assert exp.run is not None
        run_data = exp.run.run_data

        # Check structure
        assert "metadata" in run_data
        assert "message_log_ids" in run_data
        assert "prompts" in run_data
        assert "metrics" in run_data

        # Check types
        assert isinstance(run_data["metadata"], dict)
        assert isinstance(run_data["message_log_ids"], list)
        assert isinstance(run_data["prompts"], list)
        assert isinstance(run_data["metrics"], dict)


def test_concurrent_experiments(db_path):
    """Test that nested experiment contexts work properly."""
    # Create a SimpleBot instance for testing
    test_bot = SimpleBot(system_prompt="Test system prompt")

    with Experiment("outer", db_path=db_path) as outer_exp:
        with Experiment("inner", db_path=db_path) as inner_exp:
            assert outer_exp.run is not None
            assert inner_exp.run is not None

            # Log something in inner experiment
            messages = [
                BaseMessage(role="user", content="Hello"),
                BaseMessage(role="assistant", content="Hi!"),
            ]
            log_id = sqlite_log(test_bot, messages, db_path)

            # Check that the log is associated with the inner experiment
            assert log_id in inner_exp.run.run_data["message_log_ids"]
            assert log_id not in outer_exp.run.run_data["message_log_ids"]
