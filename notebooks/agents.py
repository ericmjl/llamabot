# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "llamabot[all]==0.12.0",
#     "marimo",
#     "pydantic==2.11.4",
# ]
#
# [tool.uv.sources]
# llamabot = { path = "../", editable = true }
# ///

import marimo
import llamabot as lmb
from pydantic import BaseModel, Field
from llamabot.components.tools import (
    search_internet_and_summarize,
    write_and_execute_script,
)

__generated_with = "0.13.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# LlamaBot Agents""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    This notebook demonstrates how to use LlamaBot's agent capabilities to perform autonomous tasks.
    """
    )
    return


@app.cell
def _():
    @lmb.prompt("system")
    def mistral_tool_calling_system_prompt():
        """You are an expert LLM prompt writer.

        You will be given a prompt, rewrite it to be optimal for you.
        Return for me only the prompt without the preamble or postamble.
        """

    bot = lmb.SimpleBot(
        model_name="ollama_chat/mistral-small3.1",
        system_prompt=mistral_tool_calling_system_prompt(),
    )
    response = bot(
        "Download the red wine quality dataset from https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv. Use the provided column headers. Train a random forest classifier with 100 trees and a maximum depth of 5 to predict the 'quality' column using the other features. Perform 5-fold cross-validation and return the mean and standard deviation of the accuracy."
    )
    return BaseModel, Field, bot, lmb, response


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Using AgentBot with Tools

    LlamaBot provides an `AgentBot` class that can use tools to perform autonomous tasks.
    Let's set up an agent with some built-in tools.
    """
    )
    return


@app.cell
def _():
    agent = lmb.AgentBot(
        system_prompt="You are a helpful bot that writes scripts to accomplish tasks.",
        tools=[write_and_execute_script, search_internet_and_summarize],
        model_name="openai/mistral-small3.1",
        api_base="http://localhost:11434/v1",
    )
    return agent, search_internet_and_summarize, write_and_execute_script


@app.cell
def _(agent, response):
    result = agent(response.content)
    return result


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The agent successfully:
    1. Downloaded the red wine quality dataset
    2. Trained a Random Forest Classifier with the specified parameters
    3. Performed 5-fold cross-validation
    4. Returned the mean and standard deviation of the accuracy scores

    The results show:
    - Mean accuracy score: ~0.584
    - Standard deviation: ~0.029
    """
    )
    return


@app.cell
def _():
    return (mo,)


if __name__ == "__main__":
    app.run()
