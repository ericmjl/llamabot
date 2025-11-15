"""Utility functions."""

from pathlib import Path
from typing import Optional, Dict, List, Tuple

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


def categorize_globals(globals_dict: Dict) -> Dict[str, List[Tuple[str, str]]]:
    """Safely categorize variables from globals_dict.

    This function safely categorizes variables without triggering __getitem__
    on objects like Polars DataFrames that override attribute access.

    :param globals_dict: Dictionary of global variables
    :return: Dictionary with keys 'dataframes', 'callables', 'other' containing
        lists of (name, class_name) tuples
    """
    dataframes = []
    callables = []
    other = []

    for name, value in globals_dict.items():
        if value is None:
            continue

        # Safely get class name without triggering __getitem__
        try:
            class_name = type(value).__name__
        except Exception:
            class_name = "unknown"

        # Check if it's a DataFrame (pandas or polars)
        # Use hasattr to avoid triggering __getitem__
        is_dataframe = False
        try:
            # Check for pandas DataFrame
            if (
                class_name == "DataFrame"
                and hasattr(value, "shape")
                and hasattr(value, "columns")
            ):
                is_dataframe = True
            # Check for polars DataFrame
            elif (
                class_name == "DataFrame"
                and hasattr(value, "shape")
                and hasattr(value, "schema")
            ):
                is_dataframe = True
        except Exception:
            # Intentionally ignore exceptions to avoid issues with objects
            # that override attribute access (e.g., custom DataFrames).
            pass

        if is_dataframe:
            dataframes.append((name, class_name))
        # Check if callable using Python's callable() function
        # This avoids accessing __call__ attribute directly
        elif callable(value):
            callables.append((name, class_name))
        else:
            other.append((name, class_name))

    return {
        "dataframes": dataframes,
        "callables": callables,
        "other": other,
    }


def find_or_set_db_path(db_path: Optional[Path] = None) -> Path:
    """Find or set the database path for message logging.

    If no path is provided, attempts to create the database in a .llamabot
    directory within the current project root.
    Falls back to user's home directory if project root cannot be determined.

    :param db_path: Optional path to the database file. If None, uses default locations.
    :return: Path to the database file.
    """
    if db_path is None:
        try:
            # Attempt to use project-specific .llamabot directory
            project_db_dir = here() / ".llamabot"
            project_db_dir.mkdir(
                parents=True, exist_ok=True
            )  # Ensure .llamabot directory exists
            db_path = project_db_dir / "message_log.db"
        except Exception:
            # Fallback to user's home directory if here() fails or other issues
            home_db_dir = Path.home() / ".llamabot"
            home_db_dir.mkdir(
                parents=True, exist_ok=True
            )  # Ensure ~/.llamabot directory exists
            db_path = home_db_dir / "message_log.db"
    else:
        # If db_path is provided, ensure its parent directory exists
        # Convert to Path object if it's a string, to be safe
        if isinstance(db_path, str):
            db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path
