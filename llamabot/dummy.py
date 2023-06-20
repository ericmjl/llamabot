"""Dummy class for experimentation purposes."""
from dataclasses import dataclass


@dataclass
class Dummy:
    """
    A simple data class representing a dummy object.

    .. code-block:: python

        dummy = Dummy(well=42, done=True)

    :param well: An integer value
    :param done: A boolean value indicating the status
    """

    well: int
    done: bool

    def __repr__(self):
        """Return a string representation of the object.

        :return: A string representation of the object.
        """
        return f"Dummy(well={self.well}, done={self.done})"
