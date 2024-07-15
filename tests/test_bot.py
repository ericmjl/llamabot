"""Tests for llamabot."""

from llamabot import SimpleBot


def test_ollama_bot():
    """Test that SimpleBot works with Ollama."""
    bot = SimpleBot("You are a simple bot", model_name="ollama/phi3")
    bot("Hello there!")
