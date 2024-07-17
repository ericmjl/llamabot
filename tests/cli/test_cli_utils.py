"""
This module contains tests for the get_valid_input function
from the llamabot.utils module.
"""

from unittest.mock import patch

from llamabot.cli.utils import get_valid_input


def test_get_valid_input_y():
    """Test that get_valid_input returns 'y' when the user enters 'y'."""
    with patch("builtins.input", return_value="y"):
        assert get_valid_input("Enter 'y' for yes or 'n' for no: ") == "y"


def test_get_valid_input_n():
    """Test that get_valid_input returns 'n' when the user enters 'n'."""
    with patch("builtins.input", return_value="n"):
        assert get_valid_input("Enter 'y' for yes or 'n' for no: ") == "n"


def test_get_valid_input_invalid_then_y():
    """Test that get_valid_input returns 'y'
    when the user enters 'y' after entering an invalid input."""
    with patch("builtins.input", side_effect=["invalid", "y"]):
        assert get_valid_input("Enter 'y' for yes or 'n' for no: ") == "y"


def test_get_valid_input_invalid_then_n():
    """Test that get_valid_input returns 'n'
    when the user enters 'n' after entering an invalid input."""
    with patch("builtins.input", side_effect=["invalid", "n"]):
        assert get_valid_input("Enter 'y' for yes or 'n' for no: ") == "n"
