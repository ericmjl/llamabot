# Script metadata for marimo/llamabot local dev
# Requires llamabot to be installed locally (editable install recommended)
# Requires: pip: llamabot

# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "llamabot[all]>=0.17.1",
#     "matplotlib==3.10.7",
#     "polars==1.35.2",
#     "statsmodels==0.14.5",
# ]
#
# [tool.uv.sources]
# llamabot = { path = "../", editable = true }
# ///

import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import llamabot as lmb
    import polars as pl
    import io
    from pathlib import Path
    from caseconverter import snakecase
    from llamabot.components.tools import write_and_execute_code
    from llamabot.components.tools import tool
    from llamabot.components.pocketflow import nodeify

    return (
        Path,
        io,
        lmb,
        mo,
        nodeify,
        pl,
        snakecase,
        tool,
        write_and_execute_code,
    )


@app.cell
def _(mo):
    mo.md(
        """
    # Experiment Design Agent

    Chat with a statistics agent for a first-pass critique of your experiment designs
    before bringing it to a human statistician. Paste in your experiment design written
    up as a methods paragraph in a paper or grant proposal, and the bot will help
    identify potential flaws, biases, or weaknesses in the design.

    The agent can also analyze uploaded CSV data files to identify experiment design issues.
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
def _(mo):
    mo.md(
        """
    ## Tool 1: Critique Experiment Design

    This tool critiques experiment designs by identifying potential flaws, biases, or weaknesses.
    It uses an expert statistician persona to provide constructive feedback on proposed designs.
    """
    )
    return


@app.cell
def _(lmb):
    @lmb.prompt("system")
    def experiment_design_critique_sysprompt():
        """You are an expert statistician that will help poke holes
        in proposed experiment designs to help make them robust.

        You will be provided with a proposed experiment design.
        Your task is to identify potential flaws, biases, or weaknesses in the design,
        and to ask questions back of the propose to clarify any ambiguous aspects.
        If you are given back a list of things to talk through, go through them one by one.
        """

    return (experiment_design_critique_sysprompt,)


@app.cell
def _(experiment_design_critique_sysprompt, lmb):
    critique_bot = lmb.SimpleBot(
        system_prompt=experiment_design_critique_sysprompt(),
        model_name="ollama_chat/gemma3n:latest",
    )
    return (critique_bot,)


@app.cell
def _(critique_bot, nodeify, tool):
    @nodeify(loopback_name="decide")
    @tool
    def critique_experiment_design(design: str) -> str:
        """Critique an experiment design and identify potential flaws, biases, or weaknesses.

        :param design: Description of the proposed experiment design
        :return: Critique of the experiment design with identified issues and questions, as well as suggestions for improvement.
        """
        result = critique_bot(design)
        return result.content

    return (critique_experiment_design,)


@app.cell
def _(mo):
    critique_test_button = mo.ui.run_button(label="Run Critique Test")
    return (critique_test_button,)


@app.cell(hide_code=True)
def _(critique_test_button, mo):
    mo.vstack(
        [
            mo.md(
                """
    ### Test: Critique Experiment Design Tool

    Test the critique tool with a sample experiment design:
    """
            ),
            critique_test_button,
        ]
    )
    return


@app.cell
def _(critique_experiment_design, critique_test_button, mo):
    display_critique_test_result = None
    if critique_test_button.value:
        test_design = """MCF-7 breast cancer cells were grown in RPMI-1640 medium with 10% FBS and maintained at 37°C with 5% CO2. Cells were seeded at 2 × 10^5 cells per well in 6-well plates and allowed to attach overnight. Three treatment groups were established: control (vehicle only), treatment A (1 μM tamoxifen), and treatment B (5 μM tamoxifen). After 48 hours of treatment, cells were washed twice with ice-cold PBS and lysed in RIPA buffer. All control samples (n=4) were processed on day 1, followed by all treatment A samples (n=4) on day 2, and all treatment B samples (n=4) on day 3."""
        critique_test_result = critique_experiment_design(test_design)
        display_critique_test_result = mo.md(critique_test_result)

    display_critique_test_result
    return


@app.cell
def _(mo):
    mo.md(
        """
    ---
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Tool 2: Load CSV

    This tool loads CSV files and makes them available in the notebook.
    It handles file paths or variable names, loads data into Polars DataFrames,
    and stores them in globals with a variable name derived from the filename.
    """
    )
    return


@app.cell
def _(Path, nodeify, pl, snakecase, tool):
    @nodeify(loopback_name="decide")
    @tool
    def load_csv(file_path: str, _globals_dict: dict = None) -> str:
        """Load a CSV file into a Polars DataFrame and store it in globals.

        The file will be loaded into a Polars DataFrame and stored in globals
        with a variable name derived from the filename.

        :param file_path: Path to CSV file or variable name in globals
        :param _globals_dict: Internal parameter - automatically injected by AgentBot
        :return: Confirmation message with variable name and basic info
        """
        # If file_path is a variable name in globals, get its value
        if _globals_dict is not None and file_path in _globals_dict:
            actual_path = _globals_dict[file_path]
            # If it's a string that looks like a path, use it
            if isinstance(actual_path, str):
                file_path = actual_path

        # Verify the file exists
        if not Path(file_path).exists():
            available_files = [
                k
                for k, v in (_globals_dict or {}).items()
                if isinstance(v, str) and Path(v).exists()
            ]
            raise FileNotFoundError(
                f"CSV file not found: {file_path}. "
                f"Available file variables in globals: {available_files}"
            )

        df = pl.read_csv(file_path)

        # Generate variable name from filename
        variable_name = snakecase(Path(file_path).stem)

        # Store in globals
        if _globals_dict is not None:
            _globals_dict[variable_name] = df

        # Generate confirmation message
        summary = f"""CSV file loaded successfully as '{variable_name}'.

    Shape: {df.shape[0]} rows × {df.shape[1]} columns
    Columns: {", ".join(df.columns)}

    Use `summarize_dataframe` to generate a detailed summary of this data."""
        return summary

    return (load_csv,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Tool 3: Summarize DataFrame

    This tool generates comprehensive summaries of dataframes that are already loaded in globals.
    It performs exploratory data analysis including statistics, data types, missing values,
    and basic insights about the data.
    """
    )
    return


@app.cell
def _(pl):
    def generate_dataframe_summary(df: pl.DataFrame) -> str:
        """Generate a comprehensive summary of a Polars DataFrame.

        :param df: Polars DataFrame to summarize
        :return: Multi-line string summary
        """
        summary_parts = []

        # Basic info
        summary_parts.append(f"## DataFrame Summary")
        summary_parts.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

        # Column info
        summary_parts.append("## Columns and Data Types")
        for col in df.columns:
            dtype = df[col].dtype
            summary_parts.append(f"- {col}: {dtype}")
        summary_parts.append("")

        # Missing values
        summary_parts.append("## Missing Values")
        null_counts = df.null_count()
        has_nulls = null_counts.sum_horizontal().item() > 0
        if has_nulls:
            for col in df.columns:
                null_count = null_counts[col].item()
                if null_count > 0:
                    pct = (null_count / df.shape[0]) * 100
                    summary_parts.append(f"- {col}: {null_count} ({pct:.1f}%)")
        else:
            summary_parts.append("- No missing values")
        summary_parts.append("")

        # Numeric columns summary statistics
        numeric_cols = [
            col
            for col in df.columns
            if df[col].dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]
        ]
        if numeric_cols:
            summary_parts.append("## Numeric Columns Summary Statistics")
            numeric_df = df.select(numeric_cols)
            stats = numeric_df.describe()
            summary_parts.append(str(stats))
            summary_parts.append("")

        # Categorical columns value counts (top 10)
        categorical_cols = [
            col
            for col in df.columns
            if df[col].dtype == pl.Utf8 and col not in numeric_cols
        ]
        if categorical_cols:
            summary_parts.append("## Categorical Columns Value Counts")
            for col in categorical_cols[:5]:  # Limit to first 5 categorical cols
                value_counts = df[col].value_counts().head(10)
                summary_parts.append(f"\n### {col}")
                summary_parts.append(str(value_counts))
            summary_parts.append("")

        # Sample data
        summary_parts.append("## Sample Data (First 5 Rows)")
        summary_parts.append(str(df.head(5)))
        summary_parts.append("")

        return "\n".join(summary_parts)

    return (generate_dataframe_summary,)


@app.cell
def _(generate_dataframe_summary, nodeify, pl, tool):
    @nodeify(loopback_name="decide")
    @tool
    def summarize_dataframe(dataframe_name: str, _globals_dict: dict = None) -> str:
        """Generate a comprehensive summary of a dataframe that exists in globals.

        Use this tool when the user asks to "understand", "tell me about", "what's in",
        or "summarize" their data. This tool performs exploratory data analysis on the
        specified dataframe, including statistics, data types, missing values, and basic insights.

        If the dataframe name is not specified, you can use partial matching or check
        available dataframes in globals. If multiple dataframes exist, use the most
        recently loaded one or ask the user to clarify.

        :param dataframe_name: Name of the dataframe variable in globals (e.g., "ic50_data_with_confounders").
            Supports partial matching (e.g., "ic50" will match "ic50_data_with_confounders")
        :param _globals_dict: Internal parameter - automatically injected by AgentBot
        :return: Comprehensive summary string of the dataframe
        """
        if _globals_dict is None:
            raise ValueError(
                "No globals_dict available. "
                "When calling directly, pass _globals_dict=globals() explicitly."
            )

        # Try to find the dataframe in globals
        # Support partial matching (e.g., "ic50" matches "ic50_data_with_confounders")
        df = None
        matched_name = None

        # First try exact match
        if dataframe_name in _globals_dict:
            value = _globals_dict[dataframe_name]
            if isinstance(value, pl.DataFrame):
                df = value
                matched_name = dataframe_name

        # If not found, try partial matching
        if df is None:
            for key, value in _globals_dict.items():
                if (
                    isinstance(value, pl.DataFrame)
                    and dataframe_name.lower() in key.lower()
                ):
                    df = value
                    matched_name = key
                    break

        if df is None:
            # List available dataframes
            available_dfs = [
                k for k, v in _globals_dict.items() if isinstance(v, pl.DataFrame)
            ]
            raise ValueError(
                f"Dataframe '{dataframe_name}' not found in globals. "
                f"Available dataframes: {available_dfs}"
            )

        # Generate summary
        summary = generate_dataframe_summary(df)

        # Add header with matched name
        result = f"""## Summary for '{matched_name}'

    {summary}

    Use `write_and_execute_code_wrapper` for more detailed analysis, visualizations, or custom operations."""
        return result

    return (summarize_dataframe,)


@app.cell
def _(mo):
    load_csv_test_button = mo.ui.run_button(label="Run Load CSV Test")
    return (load_csv_test_button,)


@app.cell(hide_code=True)
def _(load_csv_test_button, mo):
    mo.vstack(
        [
            mo.md(
                """
    ### Test: Load CSV Tool

    Test CSV loading with uploaded files (see file upload section below):
    """
            ),
            load_csv_test_button,
        ]
    )
    return


@app.cell
def _(load_csv, load_csv_test_button):
    load_csv_test_result = None
    if load_csv_test_button.value:
        # Test with a sample file path - update this to match your test file
        load_csv_test_result = load_csv(
            "ic50_data_with_confounders.csv", _globals_dict=globals()
        )
    load_csv_test_result
    return


@app.cell
def _(mo, summarize_dataframe_test_button):
    mo.vstack(
        [
            mo.md(
                """
    ### Test: Summarize DataFrame Tool

    Test dataframe summarization with loaded dataframes:
    """
            ),
            summarize_dataframe_test_button,
        ]
    )
    return


@app.cell
def _(mo):
    summarize_dataframe_test_button = mo.ui.run_button(
        label="Run Summarize DataFrame Test"
    )
    return (summarize_dataframe_test_button,)


@app.cell
def _(summarize_dataframe, summarize_dataframe_test_button):
    summarize_dataframe_test_result = None
    if summarize_dataframe_test_button.value:
        # Test with a loaded dataframe - update this to match your dataframe name
        summarize_dataframe_test_result = summarize_dataframe(
            "ic50_data_with_confounders", _globals_dict=globals()
        )
    summarize_dataframe_test_result
    return


@app.cell(hide_code=True)
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
    ## Tool 4: Fit GLM Model

    This tool fits a Generalized Linear Model (GLM) using statsmodels.
    It requires explicit specification of the dataframe, response variable, and predictor variables.
    The fitted model results are stored in globals for later interpretation.
    """
    )
    return


@app.cell
def _(lmb):
    @lmb.prompt("system")
    def glm_interpretation_sysprompt():
        """You are an expert statistician that interprets GLM model results.

        You will be provided with GLM model results from statsmodels.
        Your task is to:
        1. Summarize the model fit quality (R-squared, AIC, BIC, etc.)
        2. Identify significant effects (based on p-values)
        3. Explain the direction and magnitude of effects
        4. Compare effects across groups/categories
        5. Provide practical interpretation in natural language

        Be clear, concise, and focus on what matters for understanding the data.
        """

    return (glm_interpretation_sysprompt,)


@app.cell
def _(glm_interpretation_sysprompt, lmb):
    glm_interpretation_bot = lmb.SimpleBot(system_prompt=glm_interpretation_sysprompt())
    return (glm_interpretation_bot,)


@app.cell
def _(nodeify, pl, tool, write_and_execute_code):
    @nodeify(loopback_name="decide")
    @tool
    def fit_glm(
        dataframe_name: str,
        response_variable: str,
        predictor_variables: str,
        family: str = "gaussian",
        _globals_dict: dict = None,
    ) -> str:
        """Fit a Generalized Linear Model (GLM) using statsmodels.

        **IMPORTANT**: This tool requires explicit specification of:
        - dataframe_name: Name of the dataframe in globals
        - response_variable: Name of the response/dependent variable
        - predictor_variables: Comma-separated list of predictor/independent variables

        If you don't have this information, ask the user first before fitting the model.

        :param dataframe_name: Name of the dataframe variable in globals
        :param response_variable: Name of the response/dependent variable column
        :param predictor_variables: Comma-separated list of predictor variable names (e.g., "var1,var2,var3")
        :param family: GLM family type (default: "gaussian"). Options: "gaussian", "binomial", "poisson", "gamma", "inverse_gaussian"
        :param _globals_dict: Internal parameter - automatically injected by AgentBot
        :return: Confirmation message with variable name where GLM results are stored
        """
        if _globals_dict is None:
            raise ValueError(
                "No globals_dict available. "
                "When calling directly, pass _globals_dict=globals() explicitly."
            )

        # Find dataframe in globals (support partial matching)
        df = None
        matched_name = None

        # Try exact match first
        if dataframe_name in _globals_dict:
            value = _globals_dict[dataframe_name]
            if isinstance(value, pl.DataFrame):
                df = value
                matched_name = dataframe_name

        # Try partial matching
        if df is None:
            for key, value in _globals_dict.items():
                if (
                    isinstance(value, pl.DataFrame)
                    and dataframe_name.lower() in key.lower()
                ):
                    df = value
                    matched_name = key
                    break

        if df is None:
            available_dfs = [
                k for k, v in _globals_dict.items() if isinstance(v, pl.DataFrame)
            ]
            raise ValueError(
                f"Dataframe '{dataframe_name}' not found in globals. "
                f"Available dataframes: {available_dfs}"
            )

        # Parse predictor variables
        predictor_list = [p.strip() for p in predictor_variables.split(",")]

        # Verify columns exist
        missing_cols = []
        if response_variable not in df.columns:
            missing_cols.append(response_variable)
        for pred in predictor_list:
            if pred not in df.columns:
                missing_cols.append(pred)

        if missing_cols:
            raise ValueError(
                f"Columns not found in dataframe: {missing_cols}. "
                f"Available columns: {list(df.columns)}"
            )

        # Generate variable name for results
        results_var_name = f"glm_results_{matched_name}_{response_variable}"

        # Use write_and_execute_code to fit the GLM
        code_executor = write_and_execute_code(_globals_dict)

        # Generate code to fit GLM
        predictor_list_repr = repr(predictor_list)
        function_code = f'''def fit_glm_model():
    """Fit GLM model using statsmodels."""
    import statsmodels.api as sm

    # Get dataframe from globals
    df = _globals_dict["{matched_name}"]

    # Prepare data
    y = df["{response_variable}"].to_numpy()
    predictor_cols = {predictor_list_repr}
    X = df[predictor_cols].to_numpy()

    # Add intercept
    X = sm.add_constant(X)

    # Select family
    family_map = {{
        "gaussian": sm.families.Gaussian(),
        "binomial": sm.families.Binomial(),
        "poisson": sm.families.Poisson(),
        "gamma": sm.families.Gamma(),
        "inverse_gaussian": sm.families.InverseGaussian()
    }}
    family_obj = family_map.get("{family}", sm.families.Gaussian())

    # Fit model
    model = sm.GLM(y, X, family=family_obj)
    results = model.fit()

    # Store results in globals
    _globals_dict["{results_var_name}"] = results

    return results
    '''

        # Execute the code
        try:
            output = code_executor(function_code, {})
            result = output.get("result", None)

            if result is None:
                raise ValueError("GLM fitting failed - no result returned")

            return f"""GLM model fitted successfully.

    Model details:
    - Dataframe: {matched_name}
    - Response variable: {response_variable}
    - Predictor variables: {", ".join(predictor_list)}
    - Family: {family}
    - Results stored in: {results_var_name}

    Use `interpret_glm_results` with glm_results_variable='{results_var_name}' to get a natural language interpretation."""
        except Exception as e:
            return f"Error fitting GLM: {str(e)}"

    return (fit_glm,)


@app.cell
def _(glm_interpretation_bot, nodeify, tool):
    @nodeify(loopback_name="decide")
    @tool
    def interpret_glm_results(
        glm_results_variable: str, _globals_dict: dict = None
    ) -> str:
        """Interpret GLM model results and provide a natural language summary.

        This tool takes GLM results from statsmodels and generates a comprehensive
        interpretation including model fit, significant effects, and group comparisons.

        :param glm_results_variable: Name of the GLM results variable in globals (e.g., "glm_results_ic50_data_IC50")
        :param _globals_dict: Internal parameter - automatically injected by AgentBot
        :return: Natural language interpretation of the GLM results
        """
        if _globals_dict is None:
            raise ValueError(
                "No globals_dict available. "
                "When calling directly, pass _globals_dict=globals() explicitly."
            )

        # Find GLM results in globals
        glm_results = None
        matched_name = None

        # Try exact match first
        if glm_results_variable in _globals_dict:
            value = _globals_dict[glm_results_variable]
            # Check if it's a statsmodels GLM results object
            if hasattr(value, "summary"):
                glm_results = value
                matched_name = glm_results_variable

        # Try partial matching
        if glm_results is None:
            for key, value in _globals_dict.items():
                if (
                    hasattr(value, "summary")
                    and glm_results_variable.lower() in key.lower()
                ):
                    glm_results = value
                    matched_name = key
                    break

        if glm_results is None:
            available_results = [
                k for k, v in _globals_dict.items() if hasattr(v, "summary")
            ]
            raise ValueError(
                f"GLM results '{glm_results_variable}' not found in globals. "
                f"Available GLM results: {available_results}"
            )

        # Extract key information from GLM results
        summary_text = str(glm_results.summary())

        # Get additional details
        try:
            aic = glm_results.aic
            bic = glm_results.bic
            llf = glm_results.llf
            deviance = glm_results.deviance
        except:
            aic = None
            bic = None
            llf = None
            deviance = None

        # Create interpretation prompt
        interpretation_prompt = f"""Please interpret these GLM model results:

    {summary_text}

    Additional model statistics:
    - AIC: {aic}
    - BIC: {bic}
    - Log-likelihood: {llf}
    - Deviance: {deviance}

    Provide a clear, concise interpretation focusing on:
    1. Model fit quality
    2. Significant effects and their practical meaning
    3. Effect sizes and directions
    4. Group comparisons if applicable"""

        # Generate interpretation
        interpretation = glm_interpretation_bot(interpretation_prompt)
        return interpretation.content

    return (interpret_glm_results,)


@app.cell
def _(mo):
    mo.md(
        """
    ### Test: Fit GLM Tool

    Test GLM fitting with a loaded dataframe:
    """
    )
    return


@app.cell
def _(mo):
    fit_glm_test_button = mo.ui.run_button(label="Run Fit GLM Test")
    return (fit_glm_test_button,)


@app.cell
def _(fit_glm_test_button, mo):
    mo.vstack(
        [
            mo.md(
                """
    ### Test: Fit GLM Tool

    Test GLM fitting with a loaded dataframe:
    """
            ),
            fit_glm_test_button,
        ]
    )
    return


@app.cell
def _(fit_glm, fit_glm_test_button):
    fit_glm_test_result = None
    if fit_glm_test_button.value:
        # Test with a loaded dataframe - update parameters to match your data
        fit_glm_test_result = fit_glm(
            dataframe_name="ic50_data_with_confounders",
            response_variable="IC50",
            predictor_variables="Instrument,Operator,Temperature,pH,Passage_Number",
            family="gaussian",
            _globals_dict=globals(),
        )
    fit_glm_test_result
    return


@app.cell
def _(mo):
    mo.md(
        """
    ### Test: Interpret GLM Results Tool

    Test GLM results interpretation:
    """
    )
    return


@app.cell
def _(mo):
    interpret_glm_results_test_button = mo.ui.run_button(
        label="Run Interpret GLM Results Test"
    )
    return (interpret_glm_results_test_button,)


@app.cell
def _(interpret_glm_results_test_button, mo):
    mo.vstack(
        [
            mo.md(
                """
    ### Test: Interpret GLM Results Tool

    Test GLM results interpretation:
    """
            ),
            interpret_glm_results_test_button,
        ]
    )
    return


@app.cell
def _(interpret_glm_results, interpret_glm_results_test_button):
    interpret_glm_results_test_result = None
    if interpret_glm_results_test_button.value:
        # Test with GLM results - update variable name to match your results
        interpret_glm_results_test_result = interpret_glm_results(
            glm_results_variable="ic50_data_with_confounders_IC50",
            _globals_dict=globals(),
        )
    interpret_glm_results_test_result
    return


@app.cell(hide_code=True)
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
    ## File Upload Interface

    Upload CSV files for analysis. Files will be automatically loaded into the notebook.
    """
    )
    return


@app.cell
def _(mo):
    files = mo.ui.file(filetypes=[".csv"])
    return (files,)


@app.cell
def _(files):
    files
    return


@app.cell(hide_code=True)
def _(Path, files, io, mo, pl, snakecase):
    if files.value:
        for file in files.value:
            print(f"## Results for file: {file.name}")

            print("### Data Preview")
            df = pl.read_csv(io.BytesIO(file.contents))
            mo.ui.dataframe(df)
            variable_name = snakecase(Path(file.name).stem)
            globals()[variable_name] = df

            print(f"File loaded as variable: {variable_name}")
    return


@app.cell
def _(Path, files, snakecase):
    display_df = None
    if files.value:
        # Show the first uploaded file's data
        first_file_name = files.value[0].name
        _variable_name = snakecase(Path(first_file_name).stem)
        if _variable_name in globals():
            display_df = globals()[_variable_name]

    display_df
    return


@app.cell
def _(mo):
    mo.md(
        """
    ---
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## AgentBot Integration

    Create the AgentBot with all tools integrated, using a custom system prompt
    tailored for experiment design analysis.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    from llamabot import AgentBot

    return (AgentBot,)


@app.cell(hide_code=True)
def _(lmb):
    @lmb.prompt("system")
    def experiment_design_decision_sysprompt(
        globals_dict: dict = {}, categorized_vars: dict = None
    ) -> str:
        """Custom system prompt for experiment design agent decision-making.

        Given the chat history, pick for me one or more tools to execute
        in order to satisfy the user's query.

        **CRITICAL**: You MUST always select a tool. Never return empty tool calls.
        Every user query requires a tool to be executed.

        **CRITICAL**: After a tool executes successfully and returns results, you MUST use `respond_to_user`
        to return those results to the user. Tool execution is incomplete until you respond to the user.

        Give me just the tool name to pick.
        Use the tools judiciously to help answer the user's query.
        Query is always related to one of the tools.

        ## Error Handling and Self-Healing:

        **CRITICAL**: When you encounter an error, you MUST automatically try again with a different strategy.
        Do NOT just explain what you would do - actually execute the fix immediately. Be proactive and self-healing.

        If you see an error message in the conversation history (from `write_and_execute_code_wrapper` or any tool),
        it means the previous execution failed. You MUST:
        1. Analyze the error message carefully
        2. Identify what went wrong (e.g., wrong method name, missing import, incorrect syntax, data preprocessing needed)
        3. **IMMEDIATELY** generate corrected code using `write_and_execute_code_wrapper` again with the fixes
        4. **DO NOT** wait for user confirmation - execute the fix right away
        5. Pay special attention to library-specific differences (e.g., Polars uses `group_by` not `groupby`)

        **Common error patterns and fixes:**
        - AttributeError with "Did you mean": Use the suggested method name instead
        - Polars vs Pandas: Polars uses `group_by`, `select`, `with_columns` instead of pandas equivalents
        - NameError: Missing imports or undefined variables - add the necessary imports
        - SyntaxError: Fix the syntax issue in the code
        - TypeError: Check function arguments and their types
        - ValueError about categorical variables: Encode categorical variables as dummy/indicator variables before modeling
        - Model fitting errors: Preprocess data appropriately (encode categories, handle missing values, scale if needed)
        - Data type mismatches: Convert data types appropriately before operations

        **Trying alternative strategies:**
        - If one approach fails, try a different approach (e.g., different encoding method, different library, different algorithm)
        - If data preprocessing is needed, do it automatically - don't ask the user
        - If a tool fails, analyze why and try again with corrections
        - Build on previous attempts - use information from errors to inform your next attempt

        **Example error recovery flow:**
        1. Error occurs: "ValueError: categorical variables need encoding"
        2. **IMMEDIATELY** call `write_and_execute_code_wrapper` with code that encodes categorical variables
        3. Then retry the original operation (e.g., fit GLM) with the preprocessed data
        4. Do NOT explain what you would do - just do it

        **Remember**: The user expects you to solve problems automatically. When something fails, fix it and try again without asking for permission.

        ## Post-Execution Variable Return:

        **Understanding `write_and_execute_code_wrapper` return values:**
        When `write_and_execute_code_wrapper` executes successfully, it returns a dictionary with:
        - `"code"`: The function code that was executed
        - `"result"`: The return value from the function (this may be a simple value, DataFrame, dict, etc.)
        - `"created_variables"`: List of variable names that were created during execution
        - `"function_name"`: The name of the function that was created

        **What gets stored in globals:**
        - The function itself (e.g., `analyze_data`) is stored in globals and can be reused
        - Any variables created during execution are stored in globals
        - The result is automatically stored as `{function_name}_result` in globals

        **After code execution:**
        If you see "Code executed successfully" in the conversation history, it means code was just executed. You MUST:
        1. Extract the `"result"` field from the returned dictionary (don't return the raw dictionary)
        2. If the result is a dictionary with keys like `"summary"` and `"plot_png"`, format the summary text and mention the plot
        3. Use `respond_to_user` to present the formatted information to the user in a clear, helpful way
        4. If the user explicitly asked for an object to be returned, you can use `return_object_to_user` with `variable_name="{function_name}_result"`

        **Important**: The result variable contains the actual output of the data operation (e.g., grouped dataframe, analysis results), not the function itself. The function itself is also stored in globals and can be reused.

        Example: If you see "The result is stored in variable 'groupby_compound_id_result'",
        and the user asked for the result, use `return_object_to_user` with `variable_name="groupby_compound_id_result"`
        to return the actual grouped dataframe to the user. Otherwise, use `respond_to_user` to present the results.

        ## After Tool Execution:

        **When to use `respond_to_user`:**
        - **After gathering sufficient context**: Once you have called all necessary tools to gather information and context, use `respond_to_user` to provide a comprehensive answer
        - **After a single tool completes a simple request**: If a single tool execution fully satisfies the user's query, use `respond_to_user` immediately
        - **After the final step in a multi-step process**: Complete all necessary tool calls first, then respond with a comprehensive answer

        **When NOT to use `respond_to_user` immediately:**
        - If you need more information → Continue calling tools to gather context first
        - If the user's query requires multiple steps → Complete all steps before responding
        - If you're gathering context for a complex query → Finish gathering all needed information first

        Examples:
        - After `summarize_dataframe` returns a summary → If this fully answers the query, use `respond_to_user`. If you need to do more analysis first, continue with additional tools.
        - After `critique_experiment_design` returns a critique → Use `respond_to_user` to present the critique to the user
        - After `interpret_glm_results` returns an interpretation → Use `respond_to_user` to present the interpretation to the user
        - After `load_csv` confirms loading → If the user asked to "load and analyze", continue with analysis tools before responding. If they only asked to load, use `respond_to_user` to confirm.

        **CRITICAL**: The conversation is NOT complete until you respond to the user, but you may need to call multiple tools sequentially to gather sufficient context before responding.

        ## Multi-Step Requests and Context Gathering:

        **Sequential Tool Execution for Context:**
        You can call multiple tools sequentially to gather information and context before responding to the user.
        This is especially useful when:
        - You need to understand data structure before performing operations (e.g., `summarize_dataframe` before `fit_glm`)
        - You need to load data before analyzing it (e.g., `load_csv` then `summarize_dataframe`)
        - You need to gather multiple pieces of information to answer a complex query
        - You need to check what data/variables are available before deciding what to do

        **When to gather context first:**
        - If you don't know which dataframe to use → Use `summarize_dataframe` to see available dataframes
        - If you need to understand data structure before modeling → Use `summarize_dataframe` before `fit_glm`
        - If you need to load data before analysis → Use `load_csv` first, then analysis tools
        - If the user's query requires information you don't have → Use appropriate tools to gather that information first

        **After gathering context:**
        Once you have gathered sufficient context through one or more tool calls, THEN use `respond_to_user`
        to provide a comprehensive answer that incorporates all the information you've gathered.

        **Compound Requests:**
        If the user makes a compound request (e.g., "load and analyze", "tell me about the CSV"),
        break it down into steps:
        1. Execute the first step (e.g., load the CSV)
        2. After the first step completes, continue with subsequent steps (e.g., summarize the data)
        3. Do NOT stop after the first step - complete ALL parts of the user's request
        4. Only use `respond_to_user` after you have completed all necessary steps and gathered all needed information

        ## Tool Selection Guidelines:

        **When to use `load_csv`:**
        - The user asks to load a CSV file or mentions uploading a file
        - The user says "load the CSV" or "load the file"
        - Use this to load CSV files into the notebook environment
        - After loading, you can use `summarize_dataframe` to understand the data

        **When to use `summarize_dataframe`:**
        - The user asks to "understand" data, "tell me about" data, "what's in" data, "summarize" data
        - The user asks "Can you help me understand what's in my data?" → USE `summarize_dataframe`
        - The user asks "Tell me about the CSV file I just uploaded" → First use `load_csv`, then `summarize_dataframe`
        - The user wants an overview or summary of a dataframe
        - Use this BEFORE fitting models to understand the data structure
        - This tool provides comprehensive summaries including columns, types, missing values, statistics
        - **If you don't know which dataframe**: Check available dataframes in globals, or use the most recently loaded one
        - Examples: "tell me about my data", "what's in the CSV", "summarize the dataframe", "help me understand my data", "can you help me understand what's in my data"

        **When to use `fit_glm`:**
        - The user explicitly asks to fit a GLM, linear model, or regression model
        - The user asks to analyze relationships between variables using statistical modeling
        - **CRITICAL**: Before using `fit_glm`, you MUST have explicit information about:
          * The dataframe name
          * The response/dependent variable
          * The predictor/independent variables
        - **If you don't have this information**: Use `summarize_dataframe` first to understand the data structure,
          then ask the user to clarify which variable is the response and which are predictors.
          DO NOT guess or infer - always ask for explicit specification.
        - After fitting, use `interpret_glm_results` to provide natural language interpretation

        **When to use `interpret_glm_results`:**
        - After `fit_glm` has been executed and results are stored in globals
        - The user asks to interpret, explain, or summarize GLM model results
        - Use this to convert statistical results into natural language explanations

        **When to use `write_and_execute_code_wrapper`:**
        - The user asks to perform data operations: groupby, filter, aggregate, transform, calculate, plot, visualize, etc.
        - The user asks to manipulate, analyze, or process data
        - The user asks to create new data or perform computations
        - **CRITICAL**: If the user asks to "analyze" data (e.g., "analyze the CSV", "analyze the data", "analyze it"),
          this ALWAYS means doing exploratory data analysis with code - use `write_and_execute_code_wrapper` to:
          * Explore the dataframe structure (shape, columns, dtypes, missing values)
          * Generate summary statistics
          * Create visualizations (plots, charts, distributions)
          * Identify patterns, correlations, or anomalies
          * Perform statistical analysis (but use `fit_glm` for GLM modeling)
        - **After loading data**: If a user asks to "analyze" data that was just loaded (e.g., "load and analyze it"),
          the loading step is separate from analysis. After `load_csv` completes, you can use
          `summarize_dataframe` to get an overview, then `write_and_execute_code_wrapper` for detailed analysis.
        - Examples: "groupby compound ID", "filter by age", "calculate mean", "plot the data", "create a summary",
          "analyze the data", "analyze it", "explore the dataset"

        **How `write_and_execute_code_wrapper` stores objects in globals:**
        When you execute code using `write_and_execute_code_wrapper`, the following objects are automatically stored in globals():
        - **The function itself**: The function you define is stored in globals and can be reused in future code execution
        - **Variables created during execution**: Any variables created inside the function (if they're assigned to globals_dict) or at module level are stored in globals
        - **The execution result**: The return value is automatically stored as `{function_name}_result` in globals (e.g., if function is `analyze_data()`, result is stored as `analyze_data_result`)

        **Persistence and Reusability:**
        - All stored functions and variables persist across tool calls and are available for future use
        - You can reference previously created functions in subsequent code execution
        - You can access previously created variables in new code
        - This allows you to build up a library of reusable functions and data across multiple tool calls

        **Example workflow:**
        1. Execute code that creates `def analyze_correlations(): ...` → Function stored in globals
        2. Execute code that creates `correlation_matrix` variable → Variable stored in globals
        3. Later, execute code that calls `analyze_correlations()` or uses `correlation_matrix` → Can reference previous objects
        4. The result from step 1 is stored as `analyze_correlations_result` in globals

        **When to use `critique_experiment_design`:**
        - The user asks to critique, review, or evaluate an experiment design
        - The user provides a description of an experiment and wants feedback
        - Use this to identify potential flaws, biases, or weaknesses in experimental designs

        **When to use `return_object_to_user`:**
        - The user explicitly asks to "return", "show", "get", or "give me" a specific variable by name
        - The user asks to access an existing variable without performing operations
        - Examples: "show me the dataframe", "return ic50_data", "get the results variable"

        **When to use `respond_to_user`:**
        - **CRITICAL**: After a tool successfully executes and returns information (e.g., `summarize_dataframe`, `critique_experiment_design`, `interpret_glm_results`), you MUST use `respond_to_user` to return that information to the user. The tool execution is not complete until you respond to the user with the results.
        - You have enough information to answer the query with text AND no data operations are needed
        - The user asks a question that can be answered without executing code or returning objects
        - **After tool execution**: When you see a tool result in the conversation history (e.g., a summary from `summarize_dataframe`, critique from `critique_experiment_design`, interpretation from `interpret_glm_results`), use `respond_to_user` to present that result to the user in a clear, helpful way.
        - **IMPORTANT**: Do NOT use `respond_to_user` if the user asks to:
          * "analyze" data - use `write_and_execute_code_wrapper` instead
          * "understand" or "tell me about" data - use `summarize_dataframe` instead (but then use `respond_to_user` to return the summary)
          * "load" a file - use `load_csv` instead

        ## Variable Name Matching:
        When using `return_object_to_user`, you can match partial variable names intelligently.
        For example, if the user says "ic50" and "ic50_data_with_confounders" exists in globals,
        use the full variable name "ic50_data_with_confounders". Match variable names based on
        context and similarity to help users access their data more easily.

        **IMPORTANT**: Do NOT use `return_object_to_user` when the user asks to perform operations
        (groupby, filter, calculate, etc.). Use `write_and_execute_code_wrapper` instead.

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
    fit_glm,
    interpret_glm_results,
    load_csv,
    summarize_dataframe,
    write_and_execute_code,
):
    # Create AgentBot with custom system prompt
    experiment_design_agent = AgentBot(
        tools=[
            critique_experiment_design,
            load_csv,
            summarize_dataframe,
            fit_glm,
            interpret_glm_results,
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Chat Interface

    Chat with the experiment design critique agent.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    example_prompts = [
        "Tell me about the CSV file I just uploaded",
        "Can you help me understand what's in my data?",
        "What problems do you see with my experiment design?",
        "I'm planning an experiment - can you review it and tell me what might go wrong?",
        "Does my data look okay? What should I be worried about?",
        "Can you analyze my data and tell me what factors are important?",
        "Help me figure out if my experiment is set up correctly",
        "What are the main issues I should fix before running my experiment?",
    ]
    return (example_prompts,)


@app.cell(hide_code=True)
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
            mo.md("# Experiment Design Critique Agent"),
            mo.md(
                "Chat with a statistics agent for a first-pass critique of your experiment designs. "
                "Upload CSV files above, then ask the agent to analyze them or critique experiment designs."
            ),
            chat,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
