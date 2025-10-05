# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "anthropic==0.66.0",
#     "llamabot[all]==0.13.7",
#     "marimo",
#     "matplotlib==3.10.6",
#     "numpy==2.3.2",
#     "pandas==2.3.2",
#     "patsy==1.0.1",
#     "pydantic==2.11.7",
#     "pymc==5.25.1",
#     "seaborn==0.13.2",
# ]
#
# [tool.uv.sources]
# llamabot = { path = "../", editable = true }
#
# ///

import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import llamabot as lmb
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    return lmb, mo, pd


@app.cell
def _(mo):
    # Create a file upload widget for uploading CSV files
    file_upload = mo.ui.file(
        label="Upload CSV files",
        multiple=True,
    )
    file_upload
    return (file_upload,)


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(file_upload, mo, pd):
    # Process uploaded CSV files into DataFrames
    uploaded_dfs = {}

    if file_upload.value:
        for file_info in file_upload.value:
            # Get the filename without extension for the key
            filename = file_info.name
            df_name = filename.replace(".csv", "")

            # Use io.StringIO to convert bytes to a file-like object
            import io

            df = pd.read_csv(io.BytesIO(file_info.contents))
            uploaded_dfs[df_name] = df

            # Also add to globals for the bot to access
            globals()[df_name] = df

    # Display the DataFrames
    df_display = None
    if uploaded_dfs:
        df_display = mo.vstack(
            [
                mo.md(f"## Uploaded DataFrames ({len(uploaded_dfs)} files)"),
                *[
                    mo.vstack(
                        [
                            mo.md(
                                f"### {name} ({df.shape[0]} rows, {df.shape[1]} columns)"
                            ),
                            mo.ui.dataframe(df),
                        ]
                    )
                    for name, df in uploaded_dfs.items()
                ],
            ]
        )
    else:
        df_display = mo.md(
            "No CSV files uploaded yet. Please upload CSV files using the widget above."
        )

    df_display
    return (uploaded_dfs,)


@app.cell
def _(mo):
    api_base_input = mo.ui.text(
        label="Enter API Base URL (optional)",
        placeholder="https://api.openai.com/v1",
    )

    api_key_input = mo.ui.text(
        label="Enter your API key (optional)", placeholder="sk-..."
    )

    mo.vstack([api_base_input, api_key_input])
    return api_base_input, api_key_input


@app.cell
def _(api_base_input, api_key_input, lmb, write_and_execute_code):
    bot = lmb.ToolBot(
        system_prompt="You are an expert data analysis assistant that helps users solve their data analytics problems. Use only packages that are imported.",
        tools=[write_and_execute_code(globals())],
        model_name="gpt-4.1",
        api_base=api_base_input.value or None,
        api_key=api_key_input.value or None,
        memory=lmb.ChatMemory(),
    )
    return (bot,)


@app.cell
def _():
    def describe_imported_modules(global_dict):
        """
        Find and describe all imported modules within the global dictionary.

        Args:
            global_dict (dict): A dictionary of global variables (typically globals())

        Returns:
            str: A detailed description of imported modules with their full package names
        """
        import types

        imported_modules = [
            f"{name}: {module.__name__}"
            for name, module in global_dict.items()
            if isinstance(module, types.ModuleType)
        ]

        return f"Imported modules that you can use:\n" + "\n".join(imported_modules)

    print(describe_imported_modules(globals()))
    return (describe_imported_modules,)


@app.cell
def _():
    from llamabot.components.context_engineering import (
        describe_dataframes_in_globals,
    )

    print(describe_dataframes_in_globals(globals()))
    return (describe_dataframes_in_globals,)


@app.cell
def _(describe_dataframes_in_globals, lmb, uploaded_dfs):
    from pydantic import BaseModel, Field

    class SuggestedPrompts(BaseModel):
        prompts: list[str] = Field(
            description="A list of suggested prompts for the user to try."
        )

    prompt_suggestor_bot = lmb.StructuredBot(
        "You are an expert at generating prompts for data-related tasks. Given a dataframe schema, suggest a list of useful prompts that a user might want to try. These prompts should be relatively sophisticated and involve more than just a single dataframe operation. Given the dataframe schema, ensure that your suggestions are relevant to the data at hand, guessing at the user's potential goals based on the data.",
        SuggestedPrompts,
    )

    prompts = None
    if uploaded_dfs:
        prompts = prompt_suggestor_bot(
            describe_dataframes_in_globals(globals())
        ).prompts
    return (prompts,)


@app.cell
def _(lmb):
    import ast

    def write_and_execute_code(globals_dict: dict):
        """Write and execute code in a secure sandbox.

        :param globals_dictionary: The dictionary of global variables to use in the sandbox.
        :return: A function that can be used to execute code in the sandbox.
        """

        @lmb.tool
        def write_and_execute_code_wrapper(
            placeholder_function: str, keyword_args: dict = dict()
        ):
            """Write and execute `placeholder_function` with the passed in `keyword_args`.

            Use this tool for any task that requires custom Python code generation and execution.
            This tool has access to ALL globals in the current runtime environment (variables, dataframes, functions, etc.).
            Perfect for: data analysis, calculations, transformations, visualizations, custom algorithms.

            ## Code Generation Guidelines:

            1. **Write self-contained Python functions** with ALL imports inside the function body
            2. **Place all imports at the beginning of the function**: import statements must be the first lines inside the function
            3. **Include all required libraries**: pandas, numpy, matplotlib, etc. - import everything the function needs
            4. **Leverage existing global variables**: Can reference variables that exist in the runtime
            5. **Include proper error handling** and docstrings
            6. **Provide keyword arguments** when the function requires parameters
            7. **Make functions reusable** - they will be stored globally for future use
            8. **ALWAYS RETURN A VALUE**: Every function must explicitly return something - never just print, display, or show results without returning them. Even for plotting functions, return the figure/axes object.

            ## Function Arguments Handling:

            **CRITICAL**: You MUST always pass in keyword_args, which is a dictionary that can be empty, and match the function signature with the keyword_args:

            - **If your function takes NO parameters** (e.g., `def analyze_data():`), then pass keyword_args as an **empty dictionary**: `{}`
            - **If your function takes parameters** (e.g., `def filter_data(min_age, department):`), then pass keyword_args as a dictionary: `{"min_age": 30, "department": "Engineering"}`
            - **Never pass keyword_args that don't match the function signature** - this will cause execution errors

            ## Code Structure Example:

            ```python
            # Function with NO parameters - use empty dict {}
            def analyze_departments():
                '''Analyze department performance.'''
                import pandas as pd
                import numpy as np
                result = fake_df.groupby('department')['salary'].mean()
                return result
            # Function WITH parameters - pass matching keyword_args
            def filter_employees(min_age, department):
                '''Filter employees by criteria.'''
                import pandas as pd
                filtered = fake_df[(fake_df['age'] >= min_age) & (fake_df['department'] == department)]
                return filtered
            ```

            ## Return Value Requirements:

            - **Data analysis functions**: Return the computed results (numbers, DataFrames, lists, dictionaries)
            - **Plotting functions**: Return the figure or axes object (e.g., `return fig` or `return plt.gca()`)
            - **Filter/transformation functions**: Return the processed data
            - **Calculation functions**: Return the calculated values
            - **Utility functions**: Return relevant output (status, processed data, etc.)
            - **Never return None implicitly** - always have an explicit return statement

            ## Output Format:

            This tool returns a dictionary with two keys:
            - `"code"`: The generated Python function code as a string
            - `"result"`: The execution result of the function

            Example access pattern:
            ```python
            output = write_and_execute_code_wrapper(function_code, keyword_args)
            print("Generated code:", output["code"])
            print("Execution result:", output["result"])
            ```

            ## Code Access Capabilities:

            The generated code will have access to:

            - All global variables and dataframes in the current session
            - Any previously defined functions
            - The ability to import any standard Python libraries within the function
            - The ability to create new reusable functions that will be stored globally

            :param placeholder_function: The function to execute (complete Python function as string).
            :param keyword_args: The keyword arguments to pass to the function (dictionary matching function parameters).
            :return: The result of the function execution.
            """

            # Parse the code to extract the function name
            tree = ast.parse(placeholder_function)
            function_name = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    function_name = node.name
                    break

            if function_name is None:
                raise ValueError("No function definition found in the provided code.")

            ns = globals_dict
            compiled = compile(placeholder_function, "<llm>", "exec")
            exec(compiled, globals_dict, ns)

            result = ns[function_name](**keyword_args)
            return {"code": placeholder_function, "result": result}

        return write_and_execute_code_wrapper

    return (write_and_execute_code,)


@app.cell
def _(
    bot,
    describe_dataframes_in_globals,
    describe_imported_modules,
    mo,
    prompts,
):
    import json

    # Create state for the code output
    get_code_md, set_code_md = mo.state("# Code\n\nNo code generated yet.")

    def chat_turn(messages, config):
        # Each message has a `content` attribute, as well as a `role`
        # attribute ("user", "system", "assistant");
        user_message = messages[-1].content
        max_retries = 3
        retry_count = 0

        while retry_count <= max_retries:
            try:
                tools = bot(
                    user_message,
                    describe_dataframes_in_globals(globals()),
                    describe_imported_modules(globals()),
                )

                print(tools)
                print(tools[0].function.name)

                response = bot.name_to_tool_map[tools[0].function.name](
                    **json.loads(tools[0].function.arguments)
                )

                print(response)

                if tools[0].function.name == "write_and_execute_code_wrapper":
                    # Use the state setter instead of trying to modify code.value
                    code_content = "# Code\n\n```python\n" + response["code"] + "\n```"
                    set_code_md(code_content)
                else:
                    set_code_md("# Code")

                return response

            except Exception as e:
                retry_count += 1
                error_message = f"Error occurred: {str(e)}"
                print(f"Attempt {retry_count} failed: {error_message}")

                if retry_count > max_retries:
                    return f"Failed after {max_retries} attempts. Final error: {error_message}"

                # Update the user message to include the error for the next retry
                user_message = f"{user_message}\n\nPrevious attempt failed with error: {error_message}. Please fix the code and try again."

    chat = None
    if prompts:
        chat = mo.ui.chat(chat_turn, max_height=400, prompts=prompts)
    return chat, get_code_md


@app.cell
def _(chat, get_code_md, mo):
    app_display = None

    if chat:
        app_display = mo.vstack([chat, mo.md(get_code_md())])

    app_display
    return


if __name__ == "__main__":
    app.run()
