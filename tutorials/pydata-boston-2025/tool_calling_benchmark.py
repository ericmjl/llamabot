# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "llamabot[all]",
#     "marimo>=0.17.0",
#     "pandas==2.3.3",
#     "pydantic",
# ]
#
# [tool.uv.sources]
# llamabot = { path = "../../../", editable = true }
# ///

import marimo

__generated_with = "0.18.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo

    # Ensure we're using the local editable llamabot installation
    repo_root = Path(__file__).parent.parent.parent
    llamabot_path = repo_root / "llamabot"
    if llamabot_path.exists() and str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Tool Calling Benchmark

    This notebook benchmarks tool calling accuracy across multiple LLM models.
    We test how well each model selects the correct tool from a growing set of available tools (1-4 tools).

    ## Models Being Evaluated

    - `qwen2.5:32b`
    - `deepseek-r1:32b`
    - `llama3.1:70b`
    - `qwen2.5:72b-q4_K_M`
    - `gpt-4.1` (baseline for comparison)

    ## Methodology

    1. Create 4 distinct back-office tools
    2. Generate test prompts (3-5 per tool) that should trigger specific tools
    3. Test each model with 1, 2, 3, and 4 tools available
    4. Compare selected tool vs expected tool
    5. Calculate accuracy metrics and compare to gpt-4.1 baseline
    """)
    return


@app.cell
def _():
    import llamabot as lmb
    from llamabot.components.pocketflow import nodeify
    from llamabot.components.tools import tool

    return lmb, nodeify, tool


@app.cell
def _(nodeify, tool):
    @nodeify(loopback_name="decide")
    @tool
    def check_expense(amount: float, category: str, _globals_dict: dict = None) -> str:
        """Check if an expense is within budget for a given category.

        Use this tool when the user asks about expense validation, budget checks,
        or whether a specific expense amount is allowed.

        :param amount: The expense amount to check (as a float)
        :param category: The expense category (e.g., "office supplies", "travel", "meals")
        :param _globals_dict: Internal parameter - automatically injected by AgentBot
        :return: A message indicating whether the expense is within budget
        """
        # Simple budget limits for demonstration
        budget_limits = {
            "office supplies": 1000.0,
            "travel": 2000.0,
            "meals": 500.0,
            "equipment": 5000.0,
        }

        limit = budget_limits.get(category.lower(), 1000.0)
        if amount <= limit:
            return f"Expense of ${amount:.2f} for {category} is within budget (limit: ${limit:.2f})"
        else:
            return f"Expense of ${amount:.2f} for {category} exceeds budget (limit: ${limit:.2f})"

    return (check_expense,)


@app.cell
def _(nodeify, tool):
    @nodeify(loopback_name="decide")
    @tool
    def generate_report(
        report_type: str, date_range: str, _globals_dict: dict = None
    ) -> str:
        """Generate a financial or operational report.

        Use this tool when the user asks to create, generate, or produce a report.
        This includes financial reports, quarterly reports, monthly summaries, etc.

        :param report_type: Type of report to generate (e.g., "financial", "quarterly", "monthly", "annual")
        :param date_range: The date range for the report (e.g., "Q1 2025", "January 2025", "2024")
        :param _globals_dict: Internal parameter - automatically injected by AgentBot
        :return: A message indicating the report was generated
        """
        return f"Generated {report_type} report for {date_range}. Report contains summary data and key metrics."

    return (generate_report,)


@app.cell
def _(nodeify, tool):
    @nodeify(loopback_name="decide")
    @tool
    def lookup_employee(employee_id: str, _globals_dict: dict = None) -> str:
        """Look up employee information by employee ID.

        Use this tool when the user asks to find, look up, or retrieve information about an employee.
        This includes employee details, contact information, or status.

        :param employee_id: The employee ID to look up (as a string)
        :param _globals_dict: Internal parameter - automatically injected by AgentBot
        :return: Employee information as a formatted string
        """
        # Mock employee data for demonstration
        employees = {
            "12345": {
                "name": "John Doe",
                "department": "Engineering",
                "status": "Active",
            },
            "67890": {
                "name": "Jane Smith",
                "department": "Sales",
                "status": "Active",
            },
            "11111": {
                "name": "Bob Johnson",
                "department": "HR",
                "status": "On Leave",
            },
        }

        emp = employees.get(employee_id, None)
        if emp:
            return f"Employee {employee_id}: {emp['name']}, Department: {emp['department']}, Status: {emp['status']}"
        else:
            return f"Employee {employee_id} not found in the system"

    return (lookup_employee,)


@app.cell
def _(nodeify, tool):
    @nodeify(loopback_name="decide")
    @tool
    def calculate_budget(
        department: str, quarter: str, _globals_dict: dict = None
    ) -> str:
        """Calculate the budget for a department for a specific quarter.

        Use this tool when the user asks about budget calculations, department budgets,
        or quarterly budget information.

        :param department: The department name (e.g., "Engineering", "Sales", "Marketing")
        :param quarter: The quarter (e.g., "Q1", "Q2", "Q3", "Q4")
        :param _globals_dict: Internal parameter - automatically injected by AgentBot
        :return: Budget calculation result as a formatted string
        """
        # Mock budget data for demonstration
        base_budgets = {
            "engineering": 500000.0,
            "sales": 300000.0,
            "marketing": 200000.0,
            "hr": 100000.0,
        }

        base = base_budgets.get(department.lower(), 250000.0)
        # Simple calculation: base budget with slight variation by quarter
        quarter_multipliers = {"Q1": 1.0, "Q2": 1.1, "Q3": 1.05, "Q4": 1.15}
        multiplier = quarter_multipliers.get(quarter.upper(), 1.0)
        budget = base * multiplier

        return f"Budget for {department} in {quarter}: ${budget:,.2f}"

    return (calculate_budget,)


@app.cell
def _(mo):
    mo.md("""
    ## Tools Created

    We've created 4 simple back-office tools:

    1. **check_expense**: Validates expenses against budget limits
    2. **generate_report**: Generates financial/operational reports
    3. **lookup_employee**: Retrieves employee information by ID
    4. **calculate_budget**: Calculates department budgets for quarters

    Each tool is decorated with `@tool` and `@nodeify(loopback_name="decide")` to work with AgentBot.
    """)
    return


@app.cell
def _(lmb):
    @lmb.prompt("system")
    def tool_calling_sysprompt():
        """You are a helpful back-office assistant that uses tools to help users with their requests.

        Your primary responsibility is to analyze user requests and select the most appropriate tool
        to execute. You have access to various tools that can help with expense checking, report generation,
        employee lookups, and budget calculations.

        **CRITICAL**: You MUST always select a tool when the user's request matches one of the available tools.
        Never return empty tool calls when a tool is clearly needed.

        **Tool Selection Guidelines**:
        - If the user asks about expenses, budgets, or expense validation → use check_expense
        - If the user asks to create, generate, or produce a report → use generate_report
        - If the user asks to find, look up, or retrieve employee information → use lookup_employee
        - If the user asks about budget calculations or department budgets → use calculate_budget

        **Task Completion Detection**:
        After a tool executes, you must determine if the task is complete:

        - **Task is COMPLETE** if:
          * The tool result directly answers the user's original query
          * The tool result contains the information the user requested
          * No additional processing or tools are needed to satisfy the query
          * Example: User asks "Is $500 within budget?" → check_expense returns "Expense is within budget" → This directly answers the question → Use respond_to_user

        - **Task is INCOMPLETE** if:
          * The tool result is an intermediate step in a multi-step process
          * The tool result indicates you need to call another tool (e.g., "Use X()" or "Call Y()")
          * The tool stores data in globals that needs to be returned via another tool
          * The user's request has multiple parts and you've only completed one part
          * Example: User asks "Process this receipt and generate a report" → After process_receipt, you still need generate_report → Continue with more tools

        **Decision Process After Tool Execution**:
        1. Look at the user's original query
        2. Look at the tool result that was just returned
        3. Ask: "Does this tool result directly and completely answer the user's query?"
        4. If YES → Use respond_to_user to return the tool result to the user
        5. If NO → Continue with additional tools as needed

        **When to use respond_to_user**:
        - After a tool result directly answers the user's query (most common case)
        - After completing ALL steps in a multi-step workflow
        - When you need to ask the user for clarification
        - After a single tool fully satisfies a simple request

        **When to continue with more tools**:
        - The tool result is an intermediate step (e.g., "File processed, now generate report")
        - The tool result tells you to call another tool
        - The tool stores data that needs to be returned via return_object_to_user
        - The user's request has multiple parts and you've only completed one

        Read the tool docstrings carefully to understand when and how to use each tool.
        Extract the necessary parameters from the user's request and call the appropriate tool.
        """

    return (tool_calling_sysprompt,)


@app.cell
def _(mo):
    mo.md("""
    ## System Prompt

    We've created a generic system prompt that guides the agent to select appropriate tools
    based on user queries. The prompt emphasizes tool calling behavior and provides clear
    guidelines for when to use each tool.
    """)
    return


@app.cell
def _():
    # Test prompt suite: 3-5 prompts per tool, each designed to trigger exactly one specific tool
    test_prompts = [
        # check_expense prompts
        {
            "prompt": "Is a $500 office supplies expense within budget?",
            "expected_tool": "check_expense",
            "tool_index": 0,
        },
        {
            "prompt": "Check if $1500 for travel expenses is allowed",
            "expected_tool": "check_expense",
            "tool_index": 0,
        },
        {
            "prompt": "Can I spend $300 on meals?",
            "expected_tool": "check_expense",
            "tool_index": 0,
        },
        {
            "prompt": "Validate an expense of $2500 for equipment",
            "expected_tool": "check_expense",
            "tool_index": 0,
        },
        # generate_report prompts
        {
            "prompt": "Create a Q1 financial report",
            "expected_tool": "generate_report",
            "tool_index": 1,
        },
        {
            "prompt": "Generate a monthly report for January 2025",
            "expected_tool": "generate_report",
            "tool_index": 1,
        },
        {
            "prompt": "Produce an annual report for 2024",
            "expected_tool": "generate_report",
            "tool_index": 1,
        },
        {
            "prompt": "I need a quarterly report for Q2",
            "expected_tool": "generate_report",
            "tool_index": 1,
        },
        # lookup_employee prompts
        {
            "prompt": "Find information for employee ID 12345",
            "expected_tool": "lookup_employee",
            "tool_index": 2,
        },
        {
            "prompt": "Look up employee 67890",
            "expected_tool": "lookup_employee",
            "tool_index": 2,
        },
        {
            "prompt": "What's the status of employee 11111?",
            "expected_tool": "lookup_employee",
            "tool_index": 2,
        },
        {
            "prompt": "Retrieve details for employee ID 12345",
            "expected_tool": "lookup_employee",
            "tool_index": 2,
        },
        # calculate_budget prompts
        {
            "prompt": "What's the budget for Engineering in Q2?",
            "expected_tool": "calculate_budget",
            "tool_index": 3,
        },
        {
            "prompt": "Calculate the Sales department budget for Q1",
            "expected_tool": "calculate_budget",
            "tool_index": 3,
        },
        {
            "prompt": "What is the Marketing budget for Q3?",
            "expected_tool": "calculate_budget",
            "tool_index": 3,
        },
        {
            "prompt": "Get the HR budget for Q4",
            "expected_tool": "calculate_budget",
            "tool_index": 3,
        },
    ]
    return (test_prompts,)


@app.cell
def _(mo, test_prompts):
    mo.md(f"""
    ## Test Prompt Suite

    We've created {len(test_prompts)} test prompts, with 4 prompts per tool.
    Each prompt is designed to trigger exactly one specific tool.

    The prompts are stored with:
    - `prompt`: The user query text
    - `expected_tool`: The tool that should be called
    - `tool_index`: Index of the tool in our tools list (0-3)
    """)
    return


@app.cell
def _():
    from typing import Any, Dict, List

    from llamabot import AgentBot

    return AgentBot, Any, Dict, List


@app.cell
def _(calculate_budget, check_expense, generate_report, lookup_employee):
    # All available tools
    all_tools = [check_expense, generate_report, lookup_employee, calculate_budget]

    # Model configurations with api_base for Ollama models
    api_base = "https://ericmjl--ollama-service-ollamaservice-server.modal.run"

    models = [
        {
            "name": "qwen2.5:32b",
            "model_name": "ollama_chat/qwen2.5:32b",
            "api_base": api_base,
        },
        {
            "name": "deepseek-r1:32b",
            "model_name": "ollama_chat/deepseek-r1:32b",
            "api_base": api_base,
        },
        {
            "name": "llama3.1:70b",
            "model_name": "ollama_chat/llama3.1:70b",
            "api_base": api_base,
        },
        {
            "name": "qwen2.5:72b-q4_K_M",
            "model_name": "ollama_chat/qwen2.5:72b-q4_K_M",
            "api_base": api_base,
        },
        {"name": "gpt-4.1", "model_name": "gpt-4.1"},  # No api_base for gpt-4.1
    ]
    return all_tools, models


@app.cell
def _(mo):
    mo.md("""
    ## Evaluation Framework Setup

    We'll test each model with 1, 2, 3, and 4 tools available.
    For each configuration, we'll run all test prompts and record which tool was called.

    **Retry Mechanism**: Agents get up to 3 tries to select the correct tool.
    - If the correct tool is called within 3 attempts → logged as success
    - If the correct tool is not called within 3 attempts → logged as failure

    **Loop Detection**: AgentBot is configured with `max_iterations=3` to prevent infinite loops.
    If an agent exceeds 3 tool calls, it will automatically terminate by calling `respond_to_user`,
    preventing the agent from getting stuck in a loop. This ensures evaluations complete in a
    reasonable time and provides a fair comparison across models.

    To track tool calls, we'll need to inspect the AgentBot's execution flow.
    Since AgentBot uses PocketFlow, we can check the shared state to see which tools were executed.
    """)
    return


@app.cell
def _(AgentBot, Any, Dict):
    def extract_called_tools_with_retries(
        bot: AgentBot,
        query: str,
        tools_to_use: list,
        expected_tool: str,
        max_tries: int = 3,
    ) -> Dict[str, Any]:
        """Extract tool calls from AgentBot and check if expected tool is called within max_tries.

        Only considers tools from the provided tools_to_use list (our 4 evaluation tools),
        ignoring DEFAULT_TOOLS like respond_to_user.

        :param bot: The AgentBot instance
        :param query: The user query
        :param tools_to_use: List of tools we're evaluating (to filter results)
        :param expected_tool: The expected tool name to check for
        :param max_tries: Maximum number of tool calls to check (default: 3)
        :return: Dictionary with 'tools_called', 'found_expected', 'attempt_number', and 'first_called_tool'
        """
        try:
            # Reset shared state
            bot.shared = {"memory": [], "globals_dict": {}}

            # Execute the query
            bot(query, globals_dict={})

            # Debug output to diagnose looping issues
            if (
                len(bot.shared.get("memory", [])) > 5
            ):  # Only print if we suspect looping
                print(f"\n=== Debug Info for query: '{query}' ===")
                print(f"Memory entries: {len(bot.shared.get('memory', []))}")
                print(f"Iteration count: {bot.shared.get('iteration_count', 0)}")
                print(f"Result: {bot.shared.get('result', 'None')}")
                print("Last 5 memory entries:")
                for i, entry in enumerate(bot.shared.get("memory", [])[-5:], 1):
                    print(f"  {i}. {str(entry)[:100]}...")
                print("==================\n")

            # Check the shared state memory for tool execution traces
            # AgentBot's DecideNode.post() appends "Chosen Tool: {tool_name}" to memory
            memory = bot.shared.get("memory", [])

            # Create a set of our tool names for fast lookup
            our_tool_names = {tool.__name__ for tool in tools_to_use}

            # Collect up to max_tries tool calls from our evaluation tools
            tools_called = []
            found_expected = False
            attempt_number = None
            first_called_tool = None

            for entry in memory:
                if isinstance(entry, str):
                    # Check for "Chosen Tool:" pattern
                    if "Chosen Tool:" in entry:
                        # Extract tool name after the colon
                        parts = entry.split("Chosen Tool:", 1)
                        if len(parts) > 1:
                            tool_name = parts[-1].strip()
                            # Check if it's one of our evaluation tools
                            if tool_name in our_tool_names:
                                if first_called_tool is None:
                                    first_called_tool = tool_name

                                tools_called.append(tool_name)

                                # Check if this is the expected tool and we haven't found it yet
                                if not found_expected and tool_name == expected_tool:
                                    found_expected = True
                                    attempt_number = len(tools_called)

                                # Stop if we've reached max_tries
                                if len(tools_called) >= max_tries:
                                    break

            return {
                "tools_called": tools_called,
                "found_expected": found_expected,
                "attempt_number": attempt_number,
                "first_called_tool": first_called_tool,
            }
        except Exception as e:
            print(f"Error extracting tool calls: {e}")
            return {
                "tools_called": [],
                "found_expected": False,
                "attempt_number": None,
                "first_called_tool": None,
            }

    return (extract_called_tools_with_retries,)


@app.cell
def _(
    AgentBot,
    Any,
    Dict,
    List,
    extract_called_tools_with_retries,
    tool_calling_sysprompt,
):
    def run_evaluation(
        model_config: dict, num_tools: int, test_prompts: list, all_tools: list
    ) -> List[Dict[str, Any]]:
        """Run evaluation for a specific model and number of tools.

        Agents get up to 3 tries to select the correct tool. Success is logged if
        the correct tool is called within 3 attempts, failure otherwise.

        :param model_config: Model configuration dict with 'name', 'model_name', and optionally 'api_base'
        :param num_tools: Number of tools to include (1-4)
        :param test_prompts: List of test prompts to evaluate
        :param all_tools: List of all available tools
        :return: List of result dictionaries
        """
        results = []

        # Select tools for this evaluation
        tools_to_use = all_tools[:num_tools]

        # Filter test prompts to only those that expect tools we have available
        available_tool_names = [tool.__name__ for tool in tools_to_use]
        relevant_prompts = [
            p for p in test_prompts if p["expected_tool"] in available_tool_names
        ]

        # Prepare completion kwargs (include api_base if provided)
        completion_kwargs = {}
        if "api_base" in model_config:
            completion_kwargs["api_base"] = model_config["api_base"]

        # Create AgentBot with selected tools and max_iterations=3 to prevent infinite loops
        try:
            bot = AgentBot(
                tools=tools_to_use,
                system_prompt=tool_calling_sysprompt(),
                model_name=model_config["model_name"],
                max_iterations=3,
                **completion_kwargs,
            )
        except Exception as e:
            print(f"Error creating bot for {model_config['name']}: {e}")
            return results

        # Run each test prompt
        for test_prompt in relevant_prompts:
            try:
                expected_tool = test_prompt["expected_tool"]

                # Extract tool calls with retry tracking (max 3 tries)
                tool_info = extract_called_tools_with_retries(
                    bot,
                    test_prompt["prompt"],
                    tools_to_use,
                    expected_tool,
                    max_tries=3,
                )

                # Success if expected tool was found within 3 tries
                is_correct = tool_info["found_expected"]
                first_called_tool = tool_info["first_called_tool"]

                results.append(
                    {
                        "model": model_config["name"],
                        "num_tools": num_tools,
                        "prompt": test_prompt["prompt"],
                        "expected_tool": expected_tool,
                        "called_tool": first_called_tool,
                        "tools_called": tool_info["tools_called"],
                        "attempt_number": tool_info["attempt_number"],
                        "is_correct": is_correct,
                    }
                )
            except Exception as e:
                print(f"Error running prompt '{test_prompt['prompt']}': {e}")
                results.append(
                    {
                        "model": model_config["name"],
                        "num_tools": num_tools,
                        "prompt": test_prompt["prompt"],
                        "expected_tool": test_prompt["expected_tool"],
                        "called_tool": None,
                        "tools_called": [],
                        "attempt_number": None,
                        "is_correct": False,
                    }
                )

        return results

    return (run_evaluation,)


@app.cell
def _(mo):
    mo.md("""
    ## Running Evaluations

    The evaluation will test each model with 1, 2, 3, and 4 tools.
    This may take a while, so we'll use `mo.stop(True)` to prevent automatic execution.
    """)
    return


@app.cell
def _(mo):
    # Prevent automatic execution - uncomment to run evaluations
    mo.stop(True)
    return


@app.cell
def _(all_tools, models, run_evaluation, test_prompts):
    # Run all evaluations
    all_results = []

    for model_config in models:
        print(f"Evaluating model: {model_config['name']}")
        for num_tools in [1, 2, 3, 4]:
            print(f"  Testing with {num_tools} tool(s)...")
            results = run_evaluation(model_config, num_tools, test_prompts, all_tools)
            all_results.extend(results)

    all_results
    return (all_results,)


@app.cell
def _():
    import pandas as pd

    return (pd,)


@app.cell
def _(all_results, pd):
    # Convert results to DataFrame
    if all_results:
        results_df = pd.DataFrame(all_results)
    else:
        # Create empty DataFrame with expected columns
        results_df = pd.DataFrame(
            columns=[
                "model",
                "num_tools",
                "prompt",
                "expected_tool",
                "called_tool",
                "tools_called",
                "attempt_number",
                "is_correct",
            ]
        )

    results_df
    return (results_df,)


@app.cell
def _(mo):
    mo.md("""
    ## Results Analysis

    Now we'll aggregate the results to calculate accuracy metrics.
    """)
    return


@app.cell
def _(pd, results_df):
    # Calculate accuracy metrics
    if not results_df.empty:
        summary = (
            results_df.groupby(["model", "num_tools"])
            .agg(
                total_prompts=("is_correct", "count"),
                correct_calls=("is_correct", "sum"),
            )
            .reset_index()
        )
        summary["accuracy"] = (
            summary["correct_calls"] / summary["total_prompts"] * 100
        ).round(2)

        # Calculate difference from gpt-4.1 baseline
        gpt41_baseline = summary[summary["model"] == "gpt-4.1"].set_index("num_tools")[
            "accuracy"
        ]

        def calc_vs_baseline(row):
            if row["model"] == "gpt-4.1":
                return 0.0
            baseline_acc = gpt41_baseline.get(row["num_tools"], None)
            if baseline_acc is not None:
                return round(row["accuracy"] - baseline_acc, 2)
            return None

        summary["vs_gpt4_1"] = summary.apply(calc_vs_baseline, axis=1)

        # Reorder columns
        summary = summary[
            [
                "model",
                "num_tools",
                "total_prompts",
                "correct_calls",
                "accuracy",
                "vs_gpt4_1",
            ]
        ]
    else:
        summary = pd.DataFrame(
            columns=[
                "model",
                "num_tools",
                "total_prompts",
                "correct_calls",
                "accuracy",
                "vs_gpt4_1",
            ]
        )

    summary
    return (summary,)


@app.cell
def _(mo):
    mo.md("""
    ## Summary Results

    The summary table shows:
    - **model**: The model being evaluated
    - **num_tools**: Number of tools available (1-4)
    - **total_prompts**: Total number of test prompts evaluated
    - **correct_calls**: Number of correct tool selections (within 3 tries)
    - **accuracy**: Percentage of correct tool selections (within 3 tries)
    - **vs_gpt4_1**: Difference from gpt-4.1 baseline (positive = better, negative = worse)

    **Note**: Success is determined by whether the correct tool is called within 3 attempts.
    Models that find the correct tool faster (in fewer attempts) are better, but any success
    within 3 tries counts as a correct call for accuracy calculation.
    """)
    return


@app.cell
def _(mo, summary):
    # Display summary table
    if not summary.empty:
        display_result = mo.ui.dataframe(summary)
    else:
        display_result = mo.md("No results available. Run the evaluation cells above.")
    display_result
    return


@app.cell
def _(mo):
    mo.md("""
    ## Interpretation

    - **Higher accuracy** indicates better tool selection performance
    - **Positive vs_gpt4_1** values indicate the model performed better than gpt-4.1
    - **Negative vs_gpt4_1** values indicate the model performed worse than gpt-4.1
    - Models that maintain high accuracy as more tools are added demonstrate better tool discrimination

    The model with the highest accuracy and best performance relative to gpt-4.1 across all tool counts
    is the best choice for tool calling tasks.
    """)
    return


if __name__ == "__main__":
    app.run()
