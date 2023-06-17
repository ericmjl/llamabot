"""Utilities for llamabot."""


def get_valid_input(prompt):
    """
    This function prompts the user for input and validates it.

    .. code-block:: python

        user_choice = get_valid_input("Enter 'y' for yes or 'n' for no: ")

    :param prompt: The prompt to display to the user.
    :return: The validated user input, either 'y' or 'n'.
    """
    while True:
        user_input = input(prompt).lower()
        if user_input == "y" or user_input == "n":
            return user_input
        else:
            print("Invalid input. Please enter 'y' or 'n'.")
