# Script metadata for marimo/llamabot local dev
# Requires llamabot to be installed locally (editable install recommended)
# Requires: pip: llamabot

# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "llamabot[all]",
#     "matplotlib",
#     "polars",
#     "numpy",
#     "marimo>=0.17.0",
#     "pyzmq",
#     "statsmodels",
#     "tabulate",
#     "pymc",
# ]
#
# [tool.uv.sources]
# llamabot = { path = "../", editable = true }
# ///

import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import llamabot as lmb
    from llamabot.components.pocketflow import nodeify
    from llamabot.components.tools import (
        search_internet_and_summarize,
        tool,
        write_and_execute_code,
    )

    return (
        lmb,
        mo,
        nodeify,
        search_internet_and_summarize,
        tool,
        write_and_execute_code,
    )


@app.cell
def _(mo):
    mo.md(
        """
    # Experiment Design Consultation Agent

    An inquisitive statistics agent that helps design robust experiments through
    probing questions, power calculations, and critical evaluation. This agent is
    designed for both lab scientists developing experiments and human statisticians
    seeking a second opinion.

    The agent will:
    - Ask probing questions about experiment goals, constraints, and assumptions
    - Perform power calculations using reasonable priors on effect sizes
    - Generate realistic sample data tables to clarify metadata collection needs
    - Critique designs across various life sciences assay types
    - Search the literature for relevant design considerations
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ---
    """
    )
    return


@app.cell
def _(lmb):
    @lmb.prompt("system")
    def critique_experiment_design_sysprompt():
        """You are an expert experimental design consultant and statistician
        specializing in life sciences research.

        Your goal is to help researchers achieve the best possible experiment design.
        This requires being:
        - **Gentle but firm**: Be supportive and understanding, but don't shy away
          from pointing out important issues that need to be addressed
        - **Friendly and collaborative**: Work WITH the researcher, not against them.
          Frame suggestions as improvements rather than criticisms
        - **Inquisitive to uncover latent objections**: Ask probing questions to
          understand unstated concerns, constraints, or assumptions

        When critiquing an experiment design, provide a comprehensive evaluation that:
        - Identifies potential flaws, biases, and weaknesses
        - Considers biological, statistical, and practical constraints
        - Suggests concrete improvements
        - Asks clarifying questions about unstated assumptions or constraints
        - Is constructive and collaborative in tone
        """

    critique_bot = lmb.SimpleBot(
        system_prompt=critique_experiment_design_sysprompt(),
        # model_name="ollama_chat/gemma3n:latest",
    )
    return (critique_bot,)


@app.cell
def _(critique_bot, nodeify, tool):
    @nodeify(loopback_name="decide")
    @tool
    def critique_experiment_design(design: str) -> str:
        """Critique an experiment design comprehensively, identifying potential
        flaws, biases, weaknesses, and areas for improvement.

        This tool provides a thorough statistical and practical evaluation of
        experimental designs, considering multiple angles including biological,
        statistical, and practical constraints.

        :param design: Description of the proposed experiment design
        :return: Comprehensive critique with identified issues, questions, and
            suggestions for improvement
        """
        result = critique_bot(design)
        return result.content

    return (critique_experiment_design,)


@app.cell
def _(nodeify, search_internet_and_summarize, tool):
    @nodeify(loopback_name="decide")
    @tool
    def search_literature(search_query: str, max_results: int = 5) -> str:
        """Search the scientific literature and web for information about
        experimental design best practices, assay-specific considerations, or
        domain-specific knowledge.

        Use this tool when you need to:
        - Understand best practices for specific assay types
        - Find information about typical effect sizes in a field
        - Learn about special considerations for particular experimental designs
        - Get domain-specific knowledge about experimental constraints

        :param search_query: Search query describing what to look for (e.g.,
            "plate-based assay edge effects", "power analysis for RNA-seq
            experiments", "blocking strategies for agricultural field trials")
        :param max_results: Maximum number of search results to return (default: 5)
        :return: Summarized search results with relevant information
        """
        try:
            summaries = search_internet_and_summarize(search_query, max_results)
            if not summaries:
                return f"No results found for query: {search_query}"

            result_parts = [f"## Search Results for: {search_query}\n"]
            for url, summary in summaries.items():
                result_parts.append(f"### {url}\n{summary}\n")
            return "\n".join(result_parts)
        except Exception as e:
            return f"Error searching literature: {str(e)}"

    return


@app.cell
def _(mo):
    mo.md(
        """
    ---
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## AgentBot Integration

    Create the AgentBot with all experiment design consultation tools integrated.
    """
    )
    return


@app.cell
def _():
    from llamabot import AgentBot

    return (AgentBot,)


@app.cell
def _(lmb):
    @lmb.prompt("system")
    def experiment_design_decision_sysprompt(
        globals_dict: dict = {}, categorized_vars: dict = None
    ) -> str:
        """Custom system prompt for experiment design agent decision-making.

        You are an inquisitive experimental design consultant. Your role is to
        ASK QUESTIONS and PROBE multiple angles before providing recommendations.

        **CRITICAL**: You MUST always select a tool. Never return empty tool calls.
        Every user query requires a tool to be executed.

        **CRITICAL**: After a tool executes successfully and returns results, you
        MUST use `respond_to_user` to return those results to the user. Tool
        execution is incomplete until you respond to the user.

        **Your Collaborative Approach**:
        Your goal is to help researchers achieve the best possible experiment design.
        This requires being:
        - **Gentle but firm**: Be supportive and understanding, but don't shy away
          from pointing out important issues that need to be addressed
        - **Friendly and collaborative**: Work WITH the researcher, not against them.
          Frame suggestions as improvements rather than criticisms
        - **Inquisitive to uncover latent objections**: Ask probing questions to
          understand unstated concerns, constraints, or assumptions. Researchers may
          have practical limitations, budget constraints, or prior experiences that
          influence their design choices - uncover these through thoughtful questioning

        **Your Inquisitive Process**:
        - Start conversations by asking clarifying questions about experiment goals,
          constraints, and assumptions
        - Probe multiple angles: biological, statistical, practical
        - Don't immediately jump to critiques or calculations - understand first,
          then evaluate
        - Ask questions to uncover latent concerns: "What are you most worried about
          with this design?" "What constraints are you working under?" "Have you
          encountered issues with similar experiments before?"
        - Use questioning concurrently with or BEFORE running power calculations
          and generating sample data tables - understanding the full context helps
          you provide more relevant calculations and examples
        - Gather information through conversation before making recommendations

        ## Tool Selection Guidelines:

        **When to use `critique_experiment_design`:**
        - The user provides a complete or partial experiment design and wants
          feedback
        - You need to evaluate a design for flaws, biases, or weaknesses
        - Use this after gathering context about goals and constraints

        **When to use `write_and_execute_code_wrapper`:**
        - **Power calculations**: When sample size or statistical power questions arise,
          FIRST ask clarifying questions about effect sizes, variability, and constraints.
          Then use `write_and_execute_code_wrapper` to generate and execute Python code
          using statistical libraries (statsmodels.stats.power, scipy.stats) to perform
          power calculations. Estimate reasonable effect sizes based on domain knowledge
          and literature when not provided, but always explain your assumptions.
          **After execution**: Consider using `return_object_to_user` with a structured
          dictionary to return both the code (in markdown) and the results together.
        - You need to perform calculations or data manipulations
        - You want to create visualizations or summaries
        - You need to analyze or process data
        - **Generate sample data tables**: When you want to help visualize what
          metadata should be collected, FIRST ask questions about the experimental
          structure, blocking factors, and what metadata is feasible to collect.
          Then use `write_and_execute_code_wrapper` to generate realistic sample data
          tables. The sample data should:
          * Reflect the structure of the proposed experiment
          * Include all necessary metadata columns (treatment groups, blocks,
            replicates, dates, operators, equipment, etc.)
          * Have realistic-looking values appropriate for the assay type
          * Show the expected data structure and format
          * Help clarify what metadata should be collected
          * Be clearly synthetic but realistic (use patterns that indicate it's
            example data)
          * Store the result in a variable like `sample_experiment_data` in globals
          * Use appropriate libraries (polars, numpy, datetime) to generate the data
          * Set random seeds for reproducibility (e.g., `np.random.seed(42)`)
          * Return the DataFrame so it can be displayed
          **After execution**: Consider using `return_object_to_user` with a structured
          dictionary to return both the code (in markdown) and the resulting DataFrame
          together, so the user can see both the code and the data.

        **When to use `respond_to_user`:**
        - After gathering sufficient information through tools
        - To ask clarifying questions (you can ask questions directly in your response)
        - To provide recommendations or summaries
        - After completing a multi-step analysis
        - For simple text-only responses

        **When to use `return_object_to_user` (for multiple outputs):**
        - When you want to return BOTH markdown text (with code blocks) AND Python objects
          (DataFrames, plots, etc.) in a single response
        - After executing code that creates objects you want to display alongside explanations
        - When you want to show code examples AND the resulting objects together

        **How to use `return_object_to_user` for multiple outputs:**
        Create a dictionary with this structure and store it in globals, then return it:

        ```python
        # First, create the structured response dictionary
        response_dict = {
            "markdown": "Here's my analysis:\n\n```python\n# Code example\nimport pandas as pd\n```\n\nThis code calculates...",
            "code": "optional_code_to_execute = 'some code'",  # Optional: code to execute
            "objects": {
                "power_calculation_result": result_dataframe,  # Objects to display
                "sample_data_table": sample_df
            }
        }

        # Store in globals
        agent_response = response_dict

        # Then use return_object_to_user
        return_object_to_user("agent_response")
        ```

        The dictionary structure:
        - `markdown`: Markdown text (can include code blocks for display). This will be shown
          in the chat interface.
        - `code`: (Optional) Python code string to execute. This code will be executed
          automatically and any variables created will be stored in globals.
        - `objects`: (Optional) Dictionary of objects (DataFrames, plots, etc.) to store
          in globals. Keys are variable names, values are the objects.

        **Example workflow:**
        1. Use `write_and_execute_code_wrapper` to generate code and create objects
        2. Store the code string and result objects
        3. Create a structured dictionary with markdown explanation, code, and objects
        4. Store the dictionary in globals (e.g., `agent_response`)
        5. Use `return_object_to_user("agent_response")` to return it

        **Note**: The UI will automatically:
        - Display the markdown (including code blocks)
        - Execute the code if provided
        - Store objects in globals for later use

        ## Multi-Step Consultation Process:

        For complex consultations, you may need to:
        1. Ask initial clarifying questions (via `respond_to_user`) - uncover goals,
           constraints, assumptions, and latent concerns
        2. Continue probing with follow-up questions to understand the full context
           - Ask about practical constraints, budget, timeline, prior experiences
           - Understand what the researcher is most concerned about
           - Identify unstated assumptions or potential issues
        3. Search literature for domain-specific knowledge (via `search_literature`)
           when needed for context
        4. Generate sample data tables (via `write_and_execute_code_wrapper`) AFTER
           understanding the experimental structure through questioning
        5. Perform power calculations (via `write_and_execute_code_wrapper` using
           statistical libraries) AFTER understanding effect size expectations and
           constraints through questioning
        6. Provide comprehensive critique (via `critique_experiment_design`) that
           is constructive and collaborative
        7. Synthesize everything into recommendations (via `respond_to_user`) that
           are actionable and consider the researcher's constraints

        **Critical**: Steps 4 and 5 (calculations and data tables) should happen
        CONCURRENTLY WITH or AFTER thorough questioning (steps 1-2). Understanding
        the full context makes your calculations and examples more relevant and useful.

        Remember: Be gentle but firm, friendly and collaborative, and inquisitive to
        uncover latent objections. Your goal is to help scientists design better
        experiments through thoughtful questioning and guidance, working WITH them
        as a collaborative partner.

        ## Available Global Variables:

        {% if categorized_vars %}
        The available dataframes are:

        {% for name, class_name in categorized_vars.dataframes %}
        - {{ name }}: {{ class_name }}
        {% endfor %}

        The available callables are:

        {% for name, class_name in categorized_vars.callables %}
        - {{ name }}: {{ class_name }}
        {% endfor %}

        The available other variables are:

        {% for name, class_name in categorized_vars.other %}
        - {{ name }}: {{ class_name }}
        {% endfor %}
        {% else %}
        No global variables are currently available.
        {% endif %}
        """

    return (experiment_design_decision_sysprompt,)


@app.cell
def _(
    AgentBot,
    critique_experiment_design,
    experiment_design_decision_sysprompt,
    write_and_execute_code,
):
    # Create AgentBot with experiment design consultation tools
    experiment_design_agent = AgentBot(
        tools=[
            critique_experiment_design,
            write_and_execute_code(globals()),
        ],
        system_prompt=experiment_design_decision_sysprompt(),
    )
    return (experiment_design_agent,)


@app.cell
def _(mo):
    mo.md(
        """
    ---
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Chat Interface

    Chat with the experiment design consultation agent. The agent will ask
    probing questions to understand your experiment goals, constraints, and design
    before providing recommendations.
    """
    )
    return


@app.cell
def _():
    example_prompts = [
        "I'm planning a cell viability assay with 3 treatment groups. Help me design it.",
        "What sample size do I need for a two-group comparison with moderate effect size?",
        "I'm doing a plate-based ELISA. What should I be worried about?",
        "Help me critique this experiment design: [paste your design]",
        "What metadata should I collect for my agricultural field trial?",
        "I'm planning a multiplexed assay. What are the key design considerations?",
        "Can you help me understand if my experiment has enough power?",
        "What are best practices for blocking in greenhouse experiments?",
    ]
    return (example_prompts,)


@app.cell
def _(mo):
    def reformat_result(result, _globals):
        """Reformat a dictionary result into a displayable marimo component.

        Handles:
        - Executing code if present
        - Storing objects in globals
        - Extracting matplotlib figures
        - Formatting other items as markdown
        - Creating a display structure with mo.vstack

        :param result: Dictionary result from agent
        :param _globals: Globals dictionary to update
        :return: Formatted result (mo component or original dict)
        """
        if not isinstance(result, dict):
            return result

        # Execute code if present
        if "code" in result and result["code"]:
            try:
                exec(result["code"], _globals)
            except Exception as e:
                error_msg = f"Error executing code: {str(e)}"
                if "markdown" in result:
                    result["markdown"] += f"\n\n⚠️ {error_msg}"
                else:
                    result["error"] = error_msg

        # Store objects in globals if present
        if "objects" in result and isinstance(result["objects"], dict):
            for name, obj in result["objects"].items():
                _globals[name] = obj

        # Extract matplotlib figures and format other items
        try:
            from matplotlib.figure import Figure
        except ImportError:
            Figure = None

        figures = []
        figure_keys = []
        other_items = {}

        for key, value in result.items():
            # Check if it's a matplotlib figure
            if Figure is not None and isinstance(value, Figure):
                figures.append(value)
                figure_keys.append(key)
                # Store in globals for later access
                _globals[f"agent_{key}"] = value
            else:
                other_items[key] = value

        # If we have figures, create a display structure
        if figures:
            display_parts = []

            # Add markdown if present, or format other items
            if "markdown" in other_items:
                display_parts.append(mo.md(other_items["markdown"]))
            elif other_items:
                # Format other items as markdown
                import json

                formatted = "**Results:**\n\n"
                for key, val in other_items.items():
                    if key != "markdown" and key not in figure_keys:
                        if hasattr(val, "to_markdown"):
                            formatted += f"### {key}\n\n{val.to_markdown()}\n\n"
                        elif isinstance(val, (dict, list)):
                            formatted += (
                                f"### {key}\n\n```json\n"
                                f"{json.dumps(val, indent=2, default=str)}\n```\n\n"
                            )
                        else:
                            formatted += f"**{key}:** {val}\n\n"
                if formatted.strip() != "**Results:**\n\n":
                    display_parts.append(mo.md(formatted))

            # Add figures (marimo will display them automatically)
            for fig in figures:
                display_parts.append(fig)

            # Return a vertical stack of all components
            return (
                mo.vstack(display_parts) if len(display_parts) > 1 else display_parts[0]
            )
        else:
            # No figures, return dict as-is (marimo will display it)
            return result

    return (reformat_result,)


@app.cell
def _(experiment_design_agent, reformat_result):
    def chat_turn(messages, config):
        user_message = messages[-1].content
        result = experiment_design_agent(user_message, globals())

        result = reformat_result(result, globals())

        return result

    return (chat_turn,)


@app.cell
def _(chat_turn, example_prompts, mo):
    chat = mo.ui.chat(chat_turn, max_height=600, prompts=example_prompts)
    return (chat,)


@app.cell(hide_code=True)
def _(chat, mo):
    mo.vstack(
        [
            mo.md("# Experiment Design Consultation Agent"),
            mo.md(
                "Chat with an inquisitive statistics agent that helps design robust "
                "experiments through probing questions, power calculations, and "
                "critical evaluation. The agent will ask questions to understand your "
                "experiment goals, constraints, and assumptions before providing "
                "recommendations."
            ),
            chat,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
