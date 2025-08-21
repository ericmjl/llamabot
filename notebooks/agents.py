# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "llamabot[all]==0.13.0",
#     "marimo",
#     "pydantic==2.11.4",
#     "ipython",
#     "loguru==0.7.3",
#     "anthropic==0.64.0",
#     "numpy==2.3.2",
#     "pandas==2.3.1",
#     "matplotlib==3.10.5",
#     "jinja2==3.1.6",
#     "statsmodels==0.14.5",
#     "seaborn==0.13.2",
# ]
#
# [tool.uv.sources]
# llamabot = { path = "../", editable = true }
# ///

import marimo

__generated_with = "0.14.17"
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
    from typing import Callable, List

    # Enable debug mode to see detailed logs
    lmb.set_debug_mode(True)
    return (
        Callable,
        Experiment,
        List,
        lmb,
        logger,
        metric,
        planner_bot_system_prompt,
        search_internet_and_summarize,
        write_and_execute_script,
    )


@app.cell
def _(Callable, List, lmb, logger):
    from llamabot.components.tools import respond_to_user, today_date
    from llamabot.bot.simplebot import (
        SimpleBot,
        extract_content,
        extract_tool_calls,
        make_response,
        stream_chunks,
    )
    from llamabot.components.chat_memory import ChatMemory

    class ToolBot(SimpleBot):
        def __init__(
            self,
            system_prompt: str,
            model_name: str,
            tools: List[Callable] = None,
            chat_memory=ChatMemory(),
            **completion_kwargs,
        ):
            super().__init__(
                system_prompt=system_prompt,
                model_name=model_name,
                **completion_kwargs,
            )
            all_tools = [today_date, respond_to_user]
            if tools is not None:
                all_tools.extend([f for f in tools])
            self.tools = [f.json_schema for f in all_tools]
            self.name_to_tool_map = {f.__name__: f for f in all_tools}
            self.chat_memory = chat_memory

        def __call__(self, message):
            message = lmb.user(message)
            # Convert messages to a list of UserMessage objects
            message_list = [self.system_prompt]
            if self.chat_memory:
                message_list.extend(self.chat_memory.retrieve(message.content))
            message_list.extend([message])

            # Execute the plan
            stream = self.stream_target != "none"
            logger.debug("Message list: {}", message_list)
            response = make_response(self, message_list, stream=stream)
            response = stream_chunks(response, target=self.stream_target)
            logger.debug("Response: {}", response)
            tool_calls = extract_tool_calls(response)
            content = extract_content(response)

            from llamabot.components.messages import AIMessage

            self.chat_memory.append(message, AIMessage(content=content))

            return tool_calls

    return (ToolBot,)


@app.cell(hide_code=True)
def _(lmb):
    @lmb.prompt("system")
    def toolbot_sysprompt():
        """
        You are a ToolBot, an intelligent agent designed to analyze user requests and determine the most appropriate tool or function to execute.

        Your primary responsibilities:
        1. **Analyze the user's request** to understand what they want to accomplish
        2. **Select the most appropriate tool** from your available function toolkit
        3. **Extract or infer the necessary arguments** for the selected function
        4. **Return a single function call** with the proper arguments to execute

        ## Decision Process:
        When you receive a user request:
        - Break down what the user is asking for
        - Identify the core action or information needed
        - Map this to one of your available tools
        - Determine the required parameters/arguments
        - Make the function call with appropriate arguments

        ## Available Tools:
        You have access to several tools through function calling:

        ### Core Tools:
        - Use `today_date()` when users need current date/time information
        - Use `respond_to_user()` when you don't think there's code to write (e.g., greetings, general questions, explanations, or when the user just needs a conversational response)

        ### Code Execution Tool:
        - Use `write_and_execute_code()` for any task that requires custom Python code generation and execution
        - This tool takes a `placeholder_function` parameter (complete Python function as string)
        - This tool has access to ALL globals in the current runtime environment (variables, dataframes, functions, etc.)
        - Perfect for: data analysis, calculations, transformations, visualizations, custom algorithms
        - You should generate complete, self-contained Python code with imports inside the function body
        - Can execute the generated function immediately with provided keyword arguments

        ## Code Generation Guidelines:
        When using `write_and_execute_code()`:

        1. **Write self-contained Python functions** with ALL imports inside the function body
        2. **Place all imports at the beginning of the function**: import statements must be the first lines inside the function
        3. **Include all required libraries**: pandas, numpy, matplotlib, etc. - import everything the function needs
        4. **Leverage existing global variables**: Can reference `fake_df`, `dept_info`, and other variables that exist in the runtime
        5. **Include proper error handling** and docstrings
        6. **Provide keyword arguments** when the function requires parameters
        7. **Make functions reusable** - they will be stored globally for future use
        8. **ALWAYS RETURN A VALUE**: Every function must explicitly return something - never just print, display, or show results without returning them. Even for plotting functions, return the figure/axis object.

        ## Function Arguments Handling:
        **CRITICAL**: When calling `write_and_execute_code()`, you MUST match the function signature with the kwargs:

        - **If your function takes NO parameters** (e.g., `def analyze_data():`), then pass an **empty dictionary**: `{}`
        - **If your function takes parameters** (e.g., `def filter_data(min_age, department):`), then pass the required arguments as a dictionary: `{"min_age": 30, "department": "Engineering"}`
        - **Never pass kwargs that don't match the function signature** - this will cause execution errors

        ## Code Structure Example:
        ```python
        # Function with NO parameters - use empty dict {}
        def analyze_departments():
            '''Analyze department performance.'''
            import pandas as pd
            import numpy as np

            result = fake_df.groupby('department')['salary'].mean()
            return result

        # Function WITH parameters - pass matching kwargs
        def filter_employees(min_age, department):
            '''Filter employees by criteria.'''
            import pandas as pd

            filtered = fake_df[(fake_df['age'] >= min_age) & (fake_df['department'] == department)]
            return filtered
        ```

        ## Function Call Structure:
        When calling `write_and_execute_code()`, provide:
        - `placeholder_function`: Complete Python function code as a string
        - **kwargs**: Dictionary matching the function parameters
            - Empty function `def func():` → `{}`
            - Function with params `def func(a, b):` → `{"a": value1, "b": value2}`
            - NEVER return a dictionary that has keys that are NOT present in the function signature.

        ## Return Value Requirements:
        - **Data analysis functions**: Return the computed results (numbers, DataFrames, lists, dictionaries)
        - **Plotting functions**: Return the figure or axes object (e.g., `return fig` or `return plt.gca()`)
        - **Filter/transformation functions**: Return the processed data
        - **Calculation functions**: Return the calculated values
        - **Utility functions**: Return relevant output (status, processed data, etc.)
        - **Never return None implicitly** - always have an explicit return statement

        ## Examples:
        - User: "What's today's date?" → Call `today_date()`
        - User: "Hello" or "How are you?" → Call `respond_to_user(message="Hello! How can I help you today?")`
        - User: "Calculate average salary by department" → Call `write_and_execute_code(placeholder_function="def calc_avg_salary():\\n    import pandas as pd\\n    return fake_df.groupby('department')['salary'].mean()", **{})` ← Note empty dict
        - User: "Find employees older than 30" → Call `write_and_execute_code(placeholder_function="def filter_employees(min_age):\\n    import pandas as pd\\n    return fake_df[fake_df['age'] > min_age]", min_age=30)`

        ## Code Access Capabilities:
        The generated code will have access to:
        - All global variables and dataframes in the current session (like `fake_df`, `dept_info`, etc.)
        - Any previously defined functions
        - The ability to import any standard Python libraries within the function
        - The ability to create new reusable functions that will be stored globally

        Remember: You are a function selector and executor with powerful code generation capabilities. For any computational, analytical, or data processing task, generate complete, self-contained Python functions where ALL imports are placed at the beginning of the function body, leverage the current runtime environment's global variables, and MOST IMPORTANTLY:
        1. Always return meaningful results rather than just printing or displaying them
        2. **Match function signatures with kwargs exactly** - empty functions get `{}`, parameterized functions get matching argument dictionaries
        """

    return (toolbot_sysprompt,)


@app.cell
def _():
    def something():
        raise NotImplementedError()

    something.__code__
    return


@app.cell
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
def _(lmb):
    import ast

    @lmb.tool
    def write_and_execute_code(placeholder_function: str, kwargs: dict):
        """Write and execute `placeholder_function` with the passed in `kwargs`. `placeholder_function` should contain all of the imports that it needs."""
        print("------")
        print(placeholder_function)
        print("------")

        # Parse the code to extract the function name
        try:
            tree = ast.parse(placeholder_function)
            function_name = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    function_name = node.name
                    break

            if function_name is None:
                raise ValueError("No function definition found in the code")

        except Exception as e:
            return f"Error parsing function name: {str(e)}"

        print(f"Found function name: {function_name}")

        ns = dict(globals())
        compiled = compile(placeholder_function, "<llm>", "exec")
        exec(compiled, dict(globals()), ns)

        return ns[function_name](**kwargs)

    return (write_and_execute_code,)


@app.cell
def _(ToolBot, toolbot_sysprompt, write_and_execute_code):
    import json

    bot = ToolBot(
        system_prompt=toolbot_sysprompt(),
        model_name="gpt-4.1",
        tools=[write_and_execute_code],
        # temperature=1,
        # stream_target="none",
    )

    def toolbot_userprompt(request):
        prompt_parts = ["You have access to the following global variables:\n"]

        # Callable Functions
        callables = [(k, v) for k, v in globals().items() if callable(v)]
        if callables:
            prompt_parts.append("## Callable Functions:")
            for k, v in callables:
                prompt_parts.append(f"- {k}: {v.__class__.__name__}")
            prompt_parts.append("")

        # DataFrames
        dataframes = [
            (k, v)
            for k, v in globals().items()
            if hasattr(v, "shape") and hasattr(v, "columns")
        ]
        if dataframes:
            prompt_parts.append("## DataFrames:")
            for k, v in dataframes:
                prompt_parts.append(f"- {k}: DataFrame with shape {v.shape}")
                prompt_parts.append(f"  Columns: {list(v.columns)}")
                if v.shape[0] > 0:
                    prompt_parts.append("  Sample data:")
                    prompt_parts.append(f"  {v.head(3).to_string()}")
            prompt_parts.append("")

        # Other Variables
        others = [
            (k, v)
            for k, v in globals().items()
            if not callable(v) and not (hasattr(v, "shape") and hasattr(v, "columns"))
        ]
        if others:
            prompt_parts.append("## Other Variables:")
            for k, v in others:
                prompt_parts.append(f"- {k}: {v.__class__.__name__}")
            prompt_parts.append("")

        prompt_parts.append("---\n")
        prompt_parts.append(f"This is the user's request:\n\n{request}")

        return "\n".join(prompt_parts)

    tool = bot(toolbot_userprompt("help me plot the ECDF of performance by city"))
    tool
    return bot, json, tool, toolbot_userprompt


@app.cell
def _(toolbot_userprompt):
    prompt = toolbot_userprompt(
        "plot for me the empirical cumulative distribution of the data in fake_df"
    )
    print(prompt)
    return (prompt,)


@app.cell
def _(bot, prompt):
    bot(prompt)
    return


@app.cell
def _(json, tool):
    function = tool[0].function
    json.loads(function.arguments)
    return (function,)


@app.cell
def _(function):
    function.name
    return


@app.cell
def _(bot, json, toolbot_userprompt):
    def model(messages):
        print(messages[-1].content)
        prompt = toolbot_userprompt(messages[-1].content)
        tool = bot(prompt)
        print(tool)
        function = tool[0].function
        return bot.name_to_tool_map[function.name](**json.loads(function.arguments))

    return (model,)


@app.cell
def _(lmb, model):
    model(
        [
            lmb.user(
                "Calculate the salary-to-budget ratio for each department and identify employees whose salaries are above or below the department average, considering when each department was established."
            )
        ]
    )
    return


@app.cell
def _(bot):
    bot.chat_memory.retrieve("something")
    return


@app.cell
def _(model):
    import marimo as mo

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
                    tools=[
                        write_and_execute_script,
                        search_internet_and_summarize,
                    ],
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
