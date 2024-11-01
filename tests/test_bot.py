"""Tests for llamabot."""

import pytest
from llamabot import SimpleBot


@pytest.mark.xfail(reason="Test is flaky due to model availability.")
def test_ollama_bot():
    """Test that SimpleBot works with Ollama."""
    bot = SimpleBot("You are a simple bot", model_name="ollama/gemma2:2b")
    bot("Hello there!")
