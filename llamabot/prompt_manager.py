"""Decorators for turning Python functions into Jinja2-templated prompts.

This module contains decorators for turning Python functions' docstrings
into Jinja2-templated functions that accept function arguments
and return the Jinja2 templated docstrings as strings when called.

Inspired from the Outlines library.
"""

from functools import wraps
import jinja2
from jinja2 import meta
import inspect
from textwrap import dedent
from llamabot.recorder import store_prompt_version
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pyprojroot import here
from llamabot.recorder import Base, upgrade_database
from llamabot.components.messages import BaseMessage
from typing import Literal


def version_prompt(template: str, function_name: str) -> str:
    """Version a prompt template and return its hash.

    :param template: The prompt template to version.
    :param function_name: The name of the function being decorated.
    :return: The hash of the prompt template.
    """
    engine = create_engine(f"sqlite:///{here() / 'message_log.db'}")
    Base.metadata.create_all(engine)
    upgrade_database(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        stored_prompt = store_prompt_version(session, template, function_name)
        return stored_prompt.hash
    finally:
        session.close()


def prompt(role: Literal["system", "user", "assistant"] = "system"):
    """Wrap a Python function into a Jinja2-templated prompt with version control.

    :param role: The role of the prompt.
    :return: The prompt decorator.
    """

    def decorator(func):
        """Decorator function.

        :param func: The function to wrap.
        :return: The wrapped function.
        """

        @wraps(func)
        def wrapper(*args, **kwargs) -> BaseMessage:
            """Wrapper function.

            :param args: Positional arguments to the function.
            :param kwargs: Keyword arguments to the function.
            :return: A BaseMessage with the Jinja2-templated docstring as content.
            :raises ValueError: If a variable in the docstring
                is not passed into the function.
            """
            # get the function's docstring.
            docstring = func.__doc__

            # map args and kwargs onto func's signature.
            signature = inspect.signature(func)
            kwargs = signature.bind(*args, **kwargs).arguments

            # create a Jinja2 environment
            env = jinja2.Environment()

            # parse the docstring
            parsed_content = env.parse(docstring)

            # get all variables in the docstring
            variables = meta.find_undeclared_variables(parsed_content)

            # check if all variables are in kwargs
            for var in variables:
                if var not in kwargs:
                    raise ValueError(
                        f"Variable '{var}' was not passed into the function"
                    )

            # Version the prompt template
            prompt_hash = version_prompt(docstring, func.__name__)

            # interpolate docstring with args and kwargs
            template = jinja2.Template(docstring)
            string = template.render(**kwargs)

            # dedent the string
            # Split the string into lines
            lines = string.split("\n")

            # Dedent each line
            dedented_lines = [dedent(line) for line in lines]

            # Join the lines back into a single string
            dedented_string = "\n".join(dedented_lines).strip()

            # Return a BaseMessage with the specified role
            return BaseMessage(
                role=role, content=dedented_string, prompt_hash=prompt_hash
            )

        return wrapper

    return decorator
