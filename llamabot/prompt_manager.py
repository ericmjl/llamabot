"""Decorators for turning Python functions into Jinja2-templated prompts.

This module contains decorators for turning Python functions' docstrings
into Jinja2-templated functions that accept function arguments
and return the Jinja2 templated docstrings as strings when called.

Inspired from the Outlines library.
"""

import inspect
import logging
from functools import wraps
from pathlib import Path
from textwrap import dedent
from typing import Callable, Literal, Optional

import jinja2
from jinja2 import meta
from pyprojroot import here
from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker

from llamabot.components.messages import BaseMessage
from llamabot.recorder import Base, Prompt, store_prompt_version, upgrade_database

logger = logging.getLogger(__name__)


def version_prompt(
    template: str, function_name: str, db_path: Optional[Path] = None
) -> tuple[str, int]:
    """Version a prompt template and return its hash and ID.

    :param template: The prompt template to version.
    :param function_name: The name of the function being decorated.
    :param db_path: The path to the database file. Defaults to 'message_log.db' in the project root.
    :return: A tuple of (hash, id) for the prompt template.
    """
    logger.debug(f"Versioning prompt for function: {function_name}")
    if db_path is None:
        db_path = here() / "message_log.db"
    if str(db_path).startswith("sqlite:///"):
        db_path = Path(str(db_path).replace("sqlite:///", ""))
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    upgrade_database(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        logger.debug("Checking for existing prompts")
        # Check for the latest entry with the same function name but different hash
        latest_prompt = (
            session.query(Prompt)
            .filter(Prompt.function_name == function_name)
            .order_by(desc(Prompt.id))
            .first()
        )

        previous_hash = None
        if latest_prompt and latest_prompt.template != template:
            previous_hash = latest_prompt.hash
            logger.debug(f"Found previous version with hash: {previous_hash}")

        logger.debug("Storing new prompt version")
        stored_prompt = store_prompt_version(
            session, template, function_name, previous_hash
        )
        logger.debug(f"Stored prompt with hash: {stored_prompt.hash}")
        session.commit()
        logger.debug("Session committed")

        # Verify that the prompt was actually stored
        verification = session.query(Prompt).filter_by(hash=stored_prompt.hash).first()
        if verification:
            logger.debug(f"Verified prompt storage: {verification.hash}")
        else:
            logger.error("Failed to verify prompt storage")

        return stored_prompt.hash, stored_prompt.id
    except Exception as e:
        logger.error(f"Error in version_prompt: {str(e)}")
        raise
    finally:
        session.close()
        logger.debug("Session closed")


def prompt(role: Literal["system", "user", "assistant"] = "system"):
    """Wrap a Python function into a Jinja2-templated prompt with version control.

    :param role: The role of the prompt.
    :return: The prompt decorator.
    """

    def decorator(func) -> Callable:
        """Decorator function.

        :param func: The function to wrap.
        :return: The wrapped function.
        """
        # get the function's docstring and version it immediately
        docstring = func.__doc__
        prompt_hash, prompt_id = version_prompt(docstring, func.__name__)

        @wraps(func)
        def wrapper(*args, **kwargs) -> BaseMessage:
            """Wrapper function.

            :param args: Positional arguments to the function.
            :param kwargs: Keyword arguments to the function.
            :return: A BaseMessage with the Jinja2-templated docstring as content.
            :raises ValueError: If a variable in the docstring
                is not passed into the function.
            """
            # Get current experiment if one exists
            from .experiments import current_run

            experiment = current_run.get(None)
            if experiment is not None:
                experiment.add_prompt(prompt_hash, prompt_id)

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

        # Set attributes on the wrapper function
        wrapper._prompt_hash = prompt_hash
        wrapper._prompt_id = prompt_id
        wrapper._prompt_template = docstring
        wrapper._decorator_name = "prompt"

        return wrapper

    return decorator
