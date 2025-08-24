# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "llamabot==0.13.6",
#     "loguru==0.7.3",
#     "marimo",
#     "numpy==2.3.2",
#     "pandas==2.3.1",
#     "matplotlib",
#     "seaborn",
#     "scikit-learn",
# ]
#
# [tool.uv.sources]
# llamabot = { path = "../", editable = true }
#
# ///

import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import llamabot as lmb
    from llamabot.components.tools import (
        search_internet_and_summarize,
        write_and_execute_script,
        add,
    )
    from llamabot.experiments import Experiment, metric
    from llamabot.bot.agentbot import planner_bot_system_prompt
    from loguru import logger
    from typing import Callable, List
    import marimo as mo

    # Enable debug mode to see detailed logs
    lmb.set_debug_mode(True)
    return logger, mo


@app.cell
def _():
    from llamabot.components.chat_memory import ChatMemory
    from llamabot.bot.toolbot import ToolBot, toolbot_sysprompt

    return ChatMemory, ToolBot, toolbot_sysprompt


@app.cell
def _(mo):
    mo.md(r"""## First DataFrame on Personnel""")
    return


@app.cell(hide_code=True)
def _():
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate fake data
    n_rows = 100

    fake_df = pd.DataFrame(
        {
            "name": [f"Person_{i}" for i in range(1, n_rows + 1)],
            "age": np.random.randint(18, 80, n_rows),
            "city": np.random.choice(
                [
                    "New York",
                    "Los Angeles",
                    "Chicago",
                    "Houston",
                    "Phoenix",
                    "Philadelphia",
                ],
                n_rows,
            ),
            "salary": np.random.normal(65000, 15000, n_rows).round(2),
            "department": np.random.choice(
                ["Engineering", "Marketing", "Sales", "HR", "Finance"], n_rows
            ),
            "start_date": [
                datetime(2020, 1, 1) + timedelta(days=np.random.randint(0, 1460))
                for _ in range(n_rows)
            ],
            "performance_score": np.random.uniform(1, 10, n_rows).round(2),
            "is_remote": np.random.choice([True, False], n_rows),
        }
    )

    fake_df
    return fake_df, pd


@app.cell
def _(mo):
    mo.md(r"""## Second DataFrame on Departments""")
    return


@app.cell(hide_code=True)
def _(fake_df, pd):
    # Get unique departments from fake_df to ensure consistency
    departments = fake_df["department"].unique()

    # Create department information dataframe
    dept_info = pd.DataFrame(
        {
            "department": departments,
            "budget": [
                2500000,  # Engineering
                850000,  # Marketing
                1200000,  # Sales
                450000,  # HR
                750000,  # Finance
            ],
            "location": [
                "Building A - Floor 3",  # Engineering
                "Building B - Floor 2",  # Marketing
                "Building A - Floor 1",  # Sales
                "Building C - Floor 1",  # HR
                "Building B - Floor 3",  # Finance
            ],
            "manager": [
                "Sarah Johnson",  # Engineering
                "Mike Chen",  # Marketing
                "Lisa Rodriguez",  # Sales
                "David Kim",  # HR
                "Emma Thompson",  # Finance
            ],
            "team_size": [
                len(fake_df[fake_df["department"] == dept]) for dept in departments
            ],
            "established_year": [2015, 2018, 2010, 2012, 2016],
            "office_type": [
                "Open Office",
                "Creative Space",
                "Collaborative",
                "Private Offices",
                "Traditional",
            ],
        }
    )

    dept_info
    return


@app.cell
def _():
    from llamabot.components.tools import write_and_execute_code

    return (write_and_execute_code,)


@app.cell
def _():
    # f = write_and_execute_code(globals_dict=globals())
    # f.json_schema
    return


@app.cell
def _(ChatMemory, ToolBot, toolbot_sysprompt, write_and_execute_code):
    import json

    # passing in globals() is important! If it isn't present, ToolBot can't write code

    bot = ToolBot(
        system_prompt=toolbot_sysprompt(globals_dict=globals()),
        model_name="gpt-4.1",
        tools=[write_and_execute_code(globals_dict=globals())],
        chat_memory=ChatMemory(),
        temperature=0.7,
    )
    return bot, json


@app.cell
def _(bot, json, logger, mo):
    def model(messages):
        logger.debug(messages[-1].content)
        # prompt = toolbot_userprompt(messages[-1].content)
        tools = bot(messages[-1].content)

        responses = []
        for tool in tools:
            function = tool.function
            logger.debug(f"Function name: {function.name}")
            logger.debug(
                f"Function arguments returned: {json.loads(function.arguments).keys()}"
            )
            try:
                result = bot.name_to_tool_map[function.name](
                    **json.loads(function.arguments)
                )
                if isinstance(result, str):
                    result = mo.md(result)
                responses.append(result)
            except Exception as e:
                responses.append(f"Error: {e}")
        return responses

    return (model,)


@app.cell
def _():
    # result = model(
    #     [
    #         lmb.user(
    #             "Calculate the salary-to-budget ratio for each department and identify employees whose salaries are above or below the department average, considering when each department was established. Return for me as a pandas dataframe, and also return the code that you wrote to make it happen."
    #         )
    #     ]
    # )
    # result
    return


@app.cell
def _(mo, model):
    starter_prompts = [
        "Compare the average salary per employee against the total department budget. Which departments are most cost-efficient in terms of salary spending relative to their budget?",
        "Analyze employee performance scores by office type (Open Office, Creative Space, etc.). Create a visualization showing how different work environments correlate with performance.",
        "Show me which department managers oversee the highest-performing remote workers. Include the manager name, department location, and average performance score of their remote team members.",
        "Calculate the salary-to-budget ratio for each department and identify employees whose salaries are above or below the department average, considering when each department was established.",
    ]

    chat = mo.ui.chat(model, prompts=starter_prompts)
    chat
    return


@app.cell
def _():
    # m1 = mo.md("Hey there!")
    # m2 = mo.md("What's up?")

    # [m1, m2]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
