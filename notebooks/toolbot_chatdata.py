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

__generated_with = "0.16.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import llamabot as lmb
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from llamabot.components.tools import write_and_execute_code

    return lmb, mo, pd, write_and_execute_code


@app.cell
def _(mo):
    # Create a file upload widget for uploading CSV files
    file_upload = mo.ui.file(
        label="Upload CSV files",
        multiple=True,
    )
    file_upload
    return (file_upload,)


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

    model_name_input = mo.ui.text(
        label="Enter model name (optional)", placeholder="gpt-4.1"
    )

    mo.vstack([api_base_input, api_key_input, model_name_input])
    return api_base_input, api_key_input


@app.cell
def _(api_base_input, api_key_input, lmb, write_and_execute_code):
    from llamabot.components.chat_memory import ChatMemory

    bot = lmb.ToolBot(
        system_prompt="You are an expert data analysis assistant that helps users solve their data analytics problems. Use only packages that are imported.",
        tools=[write_and_execute_code(globals())],
        model_name="gpt-4.1",
        api_base=api_base_input.value or None,
        api_key=api_key_input.value or None,
        chat_memory=ChatMemory.threaded(),
    )
    return (bot,)


@app.cell
def _():
    from llamabot.components.context_engineering import (
        describe_dataframes_in_globals,
        describe_imported_modules,
    )

    print(describe_dataframes_in_globals(globals()))
    print(describe_imported_modules(globals()))
    return describe_dataframes_in_globals, describe_imported_modules


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
def _(
    bot,
    describe_dataframes_in_globals,
    describe_imported_modules,
    lmb,
    mo,
    prompts,
):
    import json
    import traceback

    # Create state for the code output
    get_code_md, set_code_md = mo.state("# Code\n\nNo code generated yet.")

    def chat_turn(messages, config):
        # Each message has a `content` attribute, as well as a `role`
        # attribute ("user", "system", "assistant");
        user_message = messages[-1].content
        max_retries = 3
        retry_count = 0

        message_to_send = lmb.user(
            user_message,
            describe_dataframes_in_globals(globals()),
            describe_imported_modules(globals()),
        )

        while retry_count <= max_retries:
            try:
                tools = bot(message_to_send)

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
                error_trace = traceback.format_exc()
                print(f"Attempt {retry_count} failed:\n{error_trace}")

                if retry_count > max_retries:
                    return f"Failed after {max_retries} attempts. Final error:\n{error_trace}"

                # Update the user message to include the full error trace for the next retry
                message_to_send = f"{user_message}\n\nPrevious attempt failed with error:\n{error_trace}\nPlease fix the code and try again."

    chat = None
    if prompts:
        chat = mo.ui.chat(chat_turn, prompts=prompts)
    return chat, get_code_md


@app.cell
def _(chat):
    chat
    return


@app.cell
def _(get_code_md, mo):
    mo.md(get_code_md())
    return


@app.cell
def _(bot, mo):
    mo.mermaid(bot.chat_memory.to_mermaid())
    return


if __name__ == "__main__":
    app.run()
