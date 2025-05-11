"""Utility functions."""

from pathlib import Path
from typing import Optional

from pyprojroot import here


def get_object_name(obj):
    """
    Get the name of the object as it's defined in the current namespace.

    :param obj: The object whose name we want to find.
    :return: The name of the object as a string, or None if not found.
    """
    for name, value in globals().items():
        if value is obj:
            return name
    return None


def find_or_set_db_path(db_path: Optional[Path] = None) -> Path:
    """Find or set the database path for message logging.

    If no path is provided, attempts to create a `.llamabot` directory in the current project root
    and places the database there. Falls back to user's home directory if project root cannot be determined.

    :param db_path: Optional path to the database file. If None, uses default locations.
    :return: Path to the database file.
    """
    if db_path is None:
        try:
            # Create .llamabot directory in the project root and place the database there
            llamabot_dir = here() / ".llamabot"
            llamabot_dir.mkdir(parents=True, exist_ok=True)

            # Create a .gitignore file in the .llamabot directory if it doesn't exist
            gitignore_path = llamabot_dir / ".gitignore"
            if not gitignore_path.exists():
                with open(gitignore_path, "w") as f:
                    f.write("# Ignore all files in this directory\n*")

            db_path = llamabot_dir / "message_log.db"
        except Exception:
            # Fall back to user home directory if we can't determine project root
            db_path = Path.home() / ".llamabot" / "message_log.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path
