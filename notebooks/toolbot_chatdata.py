# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "anthropic==0.66.0",
#     "llamabot[all]==0.13.6",
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
    from llamabot.components.tools import write_and_execute_code
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    return lmb, mo, pd, write_and_execute_code


@app.cell(hide_code=True)
def _(pd):
    df = pd.read_csv("data.csv")
    df
    return


@app.cell
def _():
    return


@app.cell
def _(lmb, write_and_execute_code):
    bot = lmb.ToolBot(
        system_prompt="You are an expert data analysis assistant that helps users solve their data analytics problems. Use only packages that are imported.",
        tools=[write_and_execute_code(globals())],
        model_name="gpt-4.1",
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
def _():
    # tools = bot(
    #     describe_dataframes_in_globals(globals()),
    #     describe_imported_modules(globals()),
    # )
    return


@app.cell
def _(describe_dataframes_in_globals, lmb):
    from pydantic import BaseModel, Field

    class SuggestedPrompts(BaseModel):
        prompts: list[str] = Field(
            description="A list of suggested prompts for the user to try."
        )

    prompt_suggestor_bot = lmb.StructuredBot(
        "You are an expert at generating prompts for data-related tasks. Given a dataframe schema, suggest a list of useful prompts that a user might want to try. These prompts should be relatively sophisticated and involve more than just a single dataframe operation. Given the dataframe schema, ensure that your suggestions are relevant to the data at hand, guessing at the user's potential goals based on the data.",
        SuggestedPrompts,
    )

    prompts = prompt_suggestor_bot(describe_dataframes_in_globals(globals())).prompts
    return (prompts,)


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

    chat = mo.ui.chat(chat_turn, max_height=400, prompts=prompts)
    return chat, get_code_md


@app.cell
def _(chat, get_code_md, mo):
    mo.vstack([chat, mo.md(get_code_md())])
    return


if __name__ == "__main__":
    app.run()
