"""CLI utility functions."""

from pathlib import Path


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


def configure_environment_variable(env_var: str, env_value: str):
    """Configure environment variable in ~/.llamabotrc.

    :param env_var: The name of the environment variable.
        Will be converted to upper case.
    :param env_value: The API key.
    """
    config_file = Path.home() / ".llamabot/.llamabotrc"
    env_var_line = f'export {env_var.upper()}="{env_value}"'

    if config_file.exists():
        with open(config_file, "r") as file:
            content = file.readlines()

        found_env_var = False
        with open(config_file, "w") as file:
            for line in content:
                if f"export {env_var.upper()}=" in line:
                    print(f"found env var line {env_var_line}")
                    file.write(env_var_line + "\n")
                    found_env_var = True
                else:
                    file.write(line)

        if not found_env_var:
            with open(config_file, "a") as file:
                file.write(env_var_line + "\n")
    else:
        with open(config_file, "w") as file:
            file.write(env_var_line + "\n")
