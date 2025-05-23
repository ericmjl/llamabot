# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "llamabot[all]==0.12.3",
#     "marimo",
#     "pydantic==2.11.4",
#     "ipython",
#     "litellm==1.70.4",
# ]
#
# [tool.uv.sources]
# llamabot = { path = "../", editable = true }
# ///

import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import llamabot as lmb

    return (lmb,)


@app.cell
def _(lmb):
    from llamabot.components.tools import (
        search_internet_and_summarize,
        write_and_execute_script,
    )

    agent_only_scripting = lmb.AgentBot(
        tools=[write_and_execute_script],
        model_name="gpt-4.1",
    )
    return (
        agent_only_scripting,
        search_internet_and_summarize,
        write_and_execute_script,
    )


@app.cell
def _(agent_only_scripting):
    response_agent_only_scripting = agent_only_scripting(
        "Summarize for me the latest ratings of Taylor Swift's latest album."
    )
    return


@app.cell
def _(lmb, search_internet_and_summarize, write_and_execute_script):
    agent = lmb.AgentBot(
        tools=[write_and_execute_script, search_internet_and_summarize],
        model_name="gpt-4.1",
    )
    return (agent,)


@app.cell
def _(agent):
    response_taylor_swift = agent(
        "Summarize for me the latest ratings of Taylor Swift's latest album."
    )
    return


@app.cell
def _(agent):
    response_wine = agent(
        "Download the red wine quality dataset from https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv. Use the provided column headers. Train a random forest classifier with 100 trees and a maximum depth of 5 to predict the 'quality' column using the other features. Perform 5-fold cross-validation and return the mean and standard deviation of the accuracy."
    )
    return (response_wine,)


@app.cell
def _(response_wine):
    print(response_wine.content)
    return


@app.cell
def _(agent):
    response_mufc = agent(
        "What are the predictions for Man Utd's europa league final game?"
    )
    return (response_mufc,)


@app.cell
def _(response_mufc):
    print(response_mufc.content)
    return


@app.cell
def _(agent):
    response_mufc1999 = agent(
        "What were the predictions for Man Utd's 1999 Champions League final?"
    )
    return (response_mufc1999,)


@app.cell
def _(response_mufc1999):
    print(response_mufc1999)
    return


if __name__ == "__main__":
    app.run()
