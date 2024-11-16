"""Tests for the experiments module.

This module contains tests for the experiment tracking functionality,
including metric recording, bot tracking, and prompt template tracking.
The tests verify that:

- Metric decorators properly validate and record scalar metrics
- Experiment context managers correctly track bot configurations
- Prompt templates are properly registered within experiments
"""

import pytest
from typing import Any

from llamabot import SimpleBot, prompt, Experiment, metric


def test_metric_decorator():
    """Test that the metric decorator properly validates and records metrics."""

    @metric
    def valid_metric(x: int) -> float:
        """Return a constant value for testing.

        :param x: An integer input that is not used.
        :returns: A constant float value of 42.0.
        """

        return 42.0

    # Test that non-numeric return values raise ValueError
    with pytest.raises(ValueError):
        # Define this without the decorator to avoid type checking
        def bad_metric(x: Any) -> Any:
            """Return a non-numeric value to test metric validation.

            :param x: An input parameter that is not used.
            :returns: A string value that should trigger a ValueError when used with @metric.
            """
            return "not a number"

        metric(bad_metric)(0)

    # Test valid metric outside experiment context
    assert valid_metric(0) == 42.0

    # Test metric recording within experiment
    with Experiment("test_metrics") as exp:
        _ = valid_metric(0)
        assert exp.current_run is not None
        assert "valid_metric" in exp.current_run.metrics
        assert exp.current_run.metrics["valid_metric"] == 42.0


def test_experiment_tracks_bots():
    """Test that the experiment context properly tracks bot creation and usage."""
    with Experiment("test_bots") as exp:
        bot = SimpleBot(
            system_prompt="You are a test bot.",
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            mock_response="aloha aloha!",
        )
        bot("hello!")

        assert exp.current_run is not None
        assert "bot" in exp.current_run.bots
        bot_config = exp.current_run.bots["bot"]
        assert bot_config["class_name"] == "SimpleBot"
        assert bot_config["model_name"] == "gpt-3.5-turbo"
        assert bot_config["temperature"] == 0.7


def test_experiment_tracks_prompts():
    """Test that the experiment context properly tracks prompt templates."""

    @prompt("system")
    def test_prompt(x: str) -> str:
        """This is a test prompt with {{ x }}"""
        return f"Test prompt with {x}"

    with Experiment("test_prompts") as exp:
        _ = test_prompt("hello")

        assert exp.current_run is not None
        assert "test_prompt" in exp.current_run.prompt_templates
        template_info = exp.current_run.prompt_templates["test_prompt"]
        assert template_info["docstring"] == "This is a test prompt with {{ x }}"


def test_experiment_multiple_runs():
    """Test that experiment properly handles multiple runs and metrics."""
    exp = Experiment("multiple_runs")

    @metric
    def count_items(x: list) -> int:
        """Count the number of items in a list.

        :param x: List of items to count
        :returns: Number of items in the list
        """
        return len(x)

    # First run
    with exp:
        items = [1, 2, 3]
        count_items(items)

    # Second run
    with exp:
        items = [1, 2, 3, 4]
        count_items(items)

    assert len(exp.runs) == 2
    assert exp.runs[0].metrics["count_items"] == 3
    assert exp.runs[1].metrics["count_items"] == 4


def test_experiment_nested_contexts():
    """Test that nested experiment contexts work properly."""
    with Experiment("outer") as outer_exp:
        with Experiment("inner") as inner_exp:

            @metric
            def test_metric(x: int) -> int:
                """Test metric function that always returns 1.

                :param x: An integer input that is not used.
                :returns: Always returns 1
                """
                return 1

            _ = test_metric(0)

            assert inner_exp.current_run is not None
            assert "test_metric" in inner_exp.current_run.metrics

        # After inner context
        assert outer_exp.current_run is not None
        assert inner_exp.current_run is None


def test_experiment_context_cleanup():
    """Test that experiment context is properly cleaned up after exit."""
    exp = Experiment("cleanup_test")

    with exp:
        assert exp.current_run is not None
        assert Experiment.get_current() is exp

    assert exp.current_run is None
    assert Experiment.get_current() is None


def test_experiment_with_exception():
    """Test that experiment handles exceptions properly."""
    exp = Experiment("exception_test")

    with pytest.raises(ValueError):
        with exp:
            raise ValueError("Test exception")

    assert exp.current_run is None
    assert len(exp.runs) == 0
