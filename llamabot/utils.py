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

    If no path is provided, attempts to create the database in the current project root.
    Falls back to user's home directory if project root cannot be determined.

    :param db_path: Optional path to the database file. If None, uses default locations.
    :return: Path to the database file.
    """
    if db_path is None:
        try:
            db_path = here() / "message_log.db"
        except Exception:
            db_path = Path.home() / ".llamabot" / "message_log.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path
