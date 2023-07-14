"""Paths to configuration files and directories."""
import os
from pathlib import Path

from dotenv import load_dotenv

llamabotrc_path = Path.home() / ".llamabot/.llamabotrc"

llamabot_config_dir = Path.home() / ".llamabot"


def default_language_model():
    """Return the default language model to be used.

    :return: The default language model to be used.
    """

    load_dotenv(llamabotrc_path)

    return os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4-32k")
