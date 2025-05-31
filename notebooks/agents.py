# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "llamabot[all]==0.12.3",
#     "marimo",
#     "pydantic==2.11.4",
#     "ipython",
#     "litellm==1.70.4",
#     "loguru==0.7.3",
# ]
#
# [tool.uv.sources]
# llamabot = { path = "../", editable = true }
# ///

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    """Setup and imports."""
    import llamabot as lmb
    from llamabot.components.tools import (
        search_internet_and_summarize,
        write_and_execute_script,
        add,
    )
    from llamabot.experiments import Experiment, metric
    from llamabot.bot.agentbot import planner_bot_system_prompt
    from loguru import logger

    # Enable debug mode to see detailed logs
    lmb.set_debug_mode(True)

    return (
        Experiment,
        add,
        lmb,
        logger,
        metric,
        planner_bot_system_prompt,
        search_internet_and_summarize,
        write_and_execute_script,
    )


@app.cell
def _(add, lmb, search_internet_and_summarize, write_and_execute_script):
    """Create an AgentBot with multiple tools."""
    agent = lmb.AgentBot(
        tools=[write_and_execute_script, search_internet_and_summarize, add],
        model_name="gpt-4.1",
    )
    return (agent,)


@app.cell
def _(agent):
    """Test the agent with a simple query about tech events."""
    response = agent("What are upcoming tech events happening in Boston?")
    print(response.content)
    return


@app.cell
def _(lmb, planner_bot_system_prompt):
    """Set up the planner bot for complex task decomposition."""
    planner_bot = lmb.SimpleBot(
        system_prompt=planner_bot_system_prompt(), model_name="gpt-4.1"
    )
    return (planner_bot,)


@app.cell
def _(lmb, metric):
    """Define metrics for measuring agent performance."""

    @metric
    def num_iterations(agent: lmb.AgentBot) -> int:
        """Return the number of iterations the agent took to complete its task.

        :param agent: The AgentBot instance
        :return: Number of iterations used
        """
        return agent.run_meta["current_iteration"]

    @metric
    def total_duration(agent: lmb.AgentBot) -> float:
        """Return the total duration in seconds that the agent took to complete its task.

        :param agent: The AgentBot instance
        :return: Duration in seconds
        """
        return agent.run_meta["duration"]

    @metric
    def tool_usage_count(agent: lmb.AgentBot) -> int:
        """Return the total number of tool calls made by the agent.

        :param agent: The AgentBot instance
        :return: Total number of tool calls
        """
        return sum(usage["calls"] for usage in agent.run_meta["tool_usage"].values())

    @metric
    def message_counts(agent: lmb.AgentBot) -> int:
        """Return the total number of messages exchanged during the agent's execution.

        :param agent: The AgentBot instance
        :return: Total number of messages (user + assistant + tool)
        """
        counts = agent.run_meta["message_counts"]
        return sum(counts.values())

    @metric
    def planning_time(agent: lmb.AgentBot) -> float:
        """Return the time taken for planning in seconds, if planning was used.

        :param agent: The AgentBot instance
        :return: Planning time in seconds, or 0 if no planning was done
        """
        if agent.run_meta["planning_metrics"]:
            return agent.run_meta["planning_metrics"].get("plan_time", 0)
        return 0

    @metric
    def returned_within_iteration_budget(passed: bool) -> int:
        """Return 1 if the agent returned within the iteration budget, 0 otherwise."""
        return 1 if passed else 0

    return (
        message_counts,
        num_iterations,
        planning_time,
        returned_within_iteration_budget,
        tool_usage_count,
        total_duration,
    )


@app.cell
def _(
    Experiment,
    lmb,
    logger,
    message_counts,
    num_iterations,
    planner_bot,
    planning_time,
    returned_within_iteration_budget,
    search_internet_and_summarize,
    tool_usage_count,
    total_duration,
    write_and_execute_script,
):
    """Benchmark different models on the same task."""
    model_names = [
        "gpt-4.1",
        "ollama_chat/command-r:latest",
        "ollama_chat/smollm2:latest",
    ]
    with Experiment("benchmarking agentbot with different models") as exp:
        for model_name in model_names:
            try:
                agent = lmb.AgentBot(
                    tools=[write_and_execute_script, search_internet_and_summarize],
                    model_name=model_name,
                    planner_bot=planner_bot,
                )
                response = agent(
                    "What is the weather going to be like this weekend in Boston?"
                )

                # Execute metrics
                total_duration(agent)
                num_iterations(agent)
                tool_usage_count(agent)
                message_counts(agent)
                planning_time(agent)
                returned_within_iteration_budget(True)
            except Exception as e:
                logger.error(f"Error in experiment {exp.name}: {e}")
                returned_within_iteration_budget(False)
    return (agent,)


@app.cell
def _(agent):
    """Test the agent with a data analysis task."""
    response = agent(
        "Download the red wine quality dataset from https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv. Use the provided column headers. Train a random forest classifier with 100 trees and a maximum depth of 5 to predict the 'quality' column using the other features. Perform 5-fold cross-validation and return the mean and standard deviation of the accuracy."
    )
    print(response.content)
    return


@app.cell
def _(agent):
    """Test the agent with a sports analysis task."""
    response = agent("What are the predictions for Man Utd's europa league final game?")
    print(response.content)
    return


@app.cell
def _(agent):
    """Test the agent with a historical analysis task."""
    response = agent(
        "What were the predictions for Man Utd's 1999 Champions League final?"
    )
    print(response.content)
    return


if __name__ == "__main__":
    app.run()
