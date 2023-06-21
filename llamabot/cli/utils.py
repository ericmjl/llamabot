"""CLI utility functions."""


def get_valid_input(prompt, valid_inputs=("y", "n")):
    """
    This function prompts the user for input and validates it.

    .. code-block:: python

        user_choice = get_valid_input("Enter 'y' for yes or 'n' for no: ")

    :param prompt: The prompt to display to the user.
    :param valid_inputs: A tuple of valid inputs.
    :return: The validated user input, either 'y' or 'n'.
    """
    msg = f"{prompt} {valid_inputs}"
    while True:
        user_input = input(msg).lower()
        if user_input in valid_inputs:
            return user_input
        else:
            print(f"Invalid input. Please enter one of { '/'.join(valid_inputs) }.")
