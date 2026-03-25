# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "marimo",
#     "llamabot>=0.17.0",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Build a bot that checks that docstring descriptions match function source

    In this notebook, we are going to build an LLM-powered bot
    that checks that docstring descriptions match function source.
    We will use the LlamaBot's StructuredBot and Pydantic to make this happen.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Define the behaviour of our bot

    The bot's ideal behaviour will look like this:

    1. It will be given function's source code.
    2. It will then be asked to return a boolean judgment call:
        1. If the docstring matches the function source, the answer will be "True".
        2. If the docstring doesn't match, the answer will be "False" along with a list of reasons.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Define Pydantic model

    The desired behaviour above means we need the following Pydantic model:
    """
    )
    return


@app.cell
def _():
    from pydantic import BaseModel, Field

    class DocstringDescribesFunction(BaseModel):
        docs_match_source: bool = Field(
            default=False,
            description="Whether or not the docstring matches the function source.",
        )
        reasons: list[str] = Field(
            default_factory=list,
            description="Reasons why the docstring doesn't match the function source.",
        )

    return (DocstringDescribesFunction,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Define the StructuredBot's behaviour

    We will now design the prompt for StructuredBot,
    particularly focusing in on the system prompt for the StructuredBot.

    The system prompt is an opportunity for us to steer the behaviour of StructuredBot.
    Here, we leave instructions for the bot to follow.
    Doing so here allows us to ensure that the bot's `__call__` method
    only needs to be concerned with receiving a function's source (as a string).
    """
    )
    return


@app.cell
def _():
    from llamabot import prompt

    @prompt
    def docstringbot_sysprompt() -> str:
        """You are an expert at documenting functions.

        You will be given a docstring and a function source.
        Your job is to determine if the docstring matches the function source.

        If it does match, respond with no reasons and respond with "True".

        If it doesn't match,
        respond with a list of reasons why the docstring doesn't match the function source.
        Be specific about the reasons, such as:

        - "The docstring is mismatched with the function. The function does <something>,
          but the docstring says <something_else>."
        - "The docstring is completely missing."
        """

    return (docstringbot_sysprompt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Next up, let's create the StructuredBot.
    Upon initializing, we provide the system prompt and the Pydantic model
    that it needs to reference.
    """
    )
    return


@app.cell
def _(DocstringDescribesFunction, docstringbot_sysprompt):
    from llamabot import StructuredBot

    docstringbot = StructuredBot(
        docstringbot_sysprompt(), pydantic_model=DocstringDescribesFunction
    )
    return (docstringbot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Test docstringbot on different functions
    """
    )
    return


@app.function
def fibbonacci(n: int) -> int:
    """Return the nth Fibonacci number.

    Mathematically, the nth Fibonnaci number is defined as
    the sum of the (n-1)th and (n-2)th Fibonacci numbers.

    As such, this is what is returned.

    :param n: The position of the Fibonacci number to return.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer.")
    elif n == 1:
        return 0
    elif n == 2:
        return 1
    else:
        return fibbonacci(n - 1) + fibbonacci(n - 2)


@app.cell
def _(docstringbot):
    from inspect import getsource

    _source_code = getsource(fibbonacci)
    docstringbot(_source_code)
    return (getsource,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Let's try now an example where the docstring is completely missing.
    """
    )
    return


@app.cell
def _(docstringbot, getsource):
    def fibbonacci_1(n: int) -> int:
        if n <= 0:
            raise ValueError("n must be a positive integer.")
        elif n == 1:
            return 0
        elif n == 2:
            return 1
        else:
            return fibbonacci(n - 1) + fibbonacci(n - 2)

    _source_code = getsource(fibbonacci_1)
    docstringbot(_source_code)
    return (fibbonacci_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    And now let's try an example where the docstring doesn't match the function source.
    """
    )
    return


@app.cell
def _(docstringbot, fibbonacci_1, getsource):
    def fibbonacci_2(n: int) -> int:
        """This function bakes a cake of the Fibonacci sequence."""
        if n <= 0:
            raise ValueError("n must be a positive integer.")
        elif n == 1:
            return 0
        elif n == 2:
            return 1
        else:
            return fibbonacci_1(n - 1) + fibbonacci_1(n - 2)

    _source_code = getsource(fibbonacci_2)
    docstringbot(_source_code)
    return


if __name__ == "__main__":
    app.run()
