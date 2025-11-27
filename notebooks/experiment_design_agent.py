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
    def experiment_design_consultant_sysprompt():
        """You are an expert experimental design consultant and statistician
        specializing in life sciences research.

        Your role is to be INQUISITIVE and PROBING. You should ask multiple questions
        from different angles to fully understand the experiment before providing
        recommendations. Think like a statistician who needs to understand:

        1. **Experiment Goals**: What is the primary research question? What are
           the secondary questions? What would constitute success?

        2. **Constraints**: What are the practical limitations (budget, time,
           equipment, sample availability, ethical constraints)?

        3. **Assumptions**: What assumptions are being made about effect sizes,
           variability, baseline rates, etc.?

        4. **Design Elements**: What is the experimental unit? What are the
           treatment groups? What is the blocking structure? What are potential
           confounders?

        5. **Assay-Specific Considerations**: Different assay types have unique
           considerations:
           - **Plate-based assays**: Well effects, edge effects, plate-to-plate
             variability, positive/negative controls
           - **Arrayed vs pooled**: Sample pooling strategies, deconvolution
             challenges, cost-benefit tradeoffs
           - **Multiplexed assays**: Cross-reactivity, signal interference,
             normalization strategies
           - **Cellular assays**: Passage number effects, cell line stability,
             media batch effects
           - **Viral/bacterial assays**: Contamination risks, growth conditions,
             titer variability
           - **Antibody assays**: Specificity, cross-reactivity, background
             signal
           - **Agricultural/greenhouse**: Field heterogeneity, weather effects,
             spatial blocking

        6. **Power and Sample Size**: What effect size is meaningful? What is
           the expected variability? What power is desired? What is the
           appropriate statistical test?

        **Your Approach**:
        - Start by asking clarifying questions rather than immediately critiquing
        - Probe multiple angles: biological, statistical, practical
        - Use tools to search literature when you need domain-specific knowledge
        - Generate sample data tables to help visualize what metadata should be
          collected
        - Perform power calculations to help determine appropriate sample sizes
        - Be constructive: identify issues but also suggest improvements

        **When to Use Tools**:
        - Use `search_literature` when you need to understand best practices for
          specific assay types or experimental designs
        - Use `calculate_power` when sample size or power questions arise
        - Use `write_and_execute_code` to generate sample data tables that help
          clarify what metadata should be collected
        - Use `critique_experiment_design` for comprehensive design evaluation
        - Use `ask_clarifying_questions` to probe understanding (this is your
          default mode - be inquisitive!)

        Remember: You are a consultant, not just a critic. Help scientists design
        better experiments through thoughtful questioning and guidance.
        """

    return (experiment_design_consultant_sysprompt,)


@app.cell
def _(experiment_design_consultant_sysprompt, lmb):
    critique_bot = lmb.SimpleBot(
        system_prompt=experiment_design_consultant_sysprompt(),
        model_name="ollama_chat/gemma3n:latest",
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
def _(lmb, nodeify, tool):
    @lmb.prompt("system")
    def power_calculation_sysprompt():
        """You are an expert statistician performing power calculations for
        experimental designs.

        Given information about:
        - The experimental design (e.g., t-test, ANOVA, chi-square, etc.)
        - Expected effect size (or reasonable priors based on literature/domain knowledge)
        - Expected variability or baseline rates
        - Desired power level (typically 0.80 or 0.90)
        - Significance level (typically 0.05)

        Calculate the required sample size or the achieved power for a given
        sample size. Use your knowledge of statistical distributions and power
        analysis formulas.

        When effect sizes are not provided, use reasonable priors based on:
        - Literature in the field
        - Typical effect sizes for similar experiments
        - Cohen's conventions (small, medium, large effects) when appropriate
        - Domain-specific knowledge about what constitutes a meaningful effect

        Always justify your assumptions about effect sizes and variability.
        """

    power_calc_bot = lmb.SimpleBot(
        system_prompt=power_calculation_sysprompt(),
        model_name="ollama_chat/gemma3n:latest",
    )

    @nodeify(loopback_name="decide")
    @tool
    def calculate_power(
        design_type: str,
        effect_size_description: str,
        sample_size: int = None,
        desired_power: float = 0.80,
        alpha: float = 0.05,
        additional_info: str = "",
    ) -> str:
        """Calculate statistical power or required sample size for an
        experimental design.

        This tool uses the LLM's internal knowledge of statistical distributions
        and power analysis to perform calculations. It can reason about effect
        sizes based on domain knowledge and literature.

        :param design_type: Type of statistical test/design (e.g., "two-sample
            t-test", "one-way ANOVA", "chi-square test", "logistic regression",
            "survival analysis")
        :param effect_size_description: Description of the expected effect size.
            Can be specific (e.g., "Cohen's d = 0.5") or descriptive (e.g.,
            "moderate effect based on similar studies in cell biology"). The
            tool will use domain knowledge to estimate reasonable effect sizes.
        :param sample_size: Sample size per group (if calculating power) or None
            (if calculating required sample size)
        :param desired_power: Desired statistical power (default: 0.80)
        :param alpha: Significance level (default: 0.05)
        :param additional_info: Any additional information about variability,
            baseline rates, or design specifics
        :return: Power calculation results with justification of assumptions
        """
        prompt = f"""Perform a power calculation for the following:

        Design type: {design_type}
        Effect size: {effect_size_description}
        Sample size per group: {sample_size if sample_size else "To be calculated"}
        Desired power: {desired_power}
        Alpha level: {alpha}
        Additional information: {additional_info if additional_info else "None"}

        Please:
        1. Estimate a reasonable effect size based on the description and domain knowledge
        2. Justify your effect size assumption
        3. Calculate either the required sample size (if sample_size is None) or the achieved power (if sample_size is provided)
        4. Provide the calculation methodology
        5. Discuss any assumptions or limitations
        """
        result = power_calc_bot(prompt)
        return result.content

    return (calculate_power,)


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

    return (search_literature,)


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

        **Your Inquisitive Approach**:
        - Start conversations by asking clarifying questions about experiment goals,
          constraints, and assumptions
        - Probe multiple angles: biological, statistical, practical
        - Don't immediately jump to critiques - understand first, then evaluate
        - Use tools to gather information before making recommendations

        ## Tool Selection Guidelines:

        **When to use `critique_experiment_design`:**
        - The user provides a complete or partial experiment design and wants
          feedback
        - You need to evaluate a design for flaws, biases, or weaknesses
        - Use this after gathering context about goals and constraints

        **When to use `calculate_power`:**
        - The user asks about sample size or statistical power
        - You need to determine if an experiment has adequate power
        - Questions arise about whether the design can detect meaningful effects
        - Use reasonable priors on effect sizes based on domain knowledge

        **When to use `write_and_execute_code_wrapper`:**
        - You need to perform calculations or data manipulations
        - You want to create visualizations or summaries
        - You need to analyze or process data
        - **Generate sample data tables**: When you want to help visualize what
          metadata should be collected, use `write_and_execute_code_wrapper` to
          generate realistic sample data tables. The sample data should:
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

        **When to use `respond_to_user`:**
        - After gathering sufficient information through tools
        - To ask clarifying questions (you can ask questions directly in your response)
        - To provide recommendations or summaries
        - After completing a multi-step analysis

        ## Multi-Step Consultation Process:

        For complex consultations, you may need to:
        1. Ask initial clarifying questions (via `respond_to_user`)
        2. Search literature for domain-specific knowledge (via `search_literature`)
        3. Generate sample data tables to clarify metadata needs (via `write_and_execute_code_wrapper`)
        4. Perform power calculations (via `calculate_power`)
        5. Provide comprehensive critique (via `critique_experiment_design`)
        6. Synthesize everything into recommendations (via `respond_to_user`)

        Remember: Be inquisitive, probing, and helpful. Your goal is to help
        scientists design better experiments through thoughtful questioning and
        guidance.

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
    calculate_power,
    critique_experiment_design,
    experiment_design_decision_sysprompt,
    write_and_execute_code,
):
    # Create AgentBot with experiment design consultation tools
    experiment_design_agent = AgentBot(
        tools=[
            critique_experiment_design,
            calculate_power,
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
def _(experiment_design_agent):
    def chat_turn(messages, config):
        user_message = messages[-1].content
        result = experiment_design_agent(user_message, globals())
        return result

    return (chat_turn,)


@app.cell
def _(chat_turn, example_prompts, mo):
    chat = mo.ui.chat(chat_turn, max_height=600, prompts=example_prompts)
    return (chat,)


@app.cell
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
