"""Utility functions."""
from functools import wraps


def capture_errors(func):
    """Decorator to capture errors and print them to the console.

    :param func: Function to decorate.
    :return: Decorated function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        """Wrapper function for the decorator.

        :param args: Positional arguments to pass to the function.
        :param kwargs: Keyword arguments to pass to the function.
        :return: Return value of the function.
        :raises: Exception: If an error occurs.
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print("An error occurred:", str(e))

    return wrapper
