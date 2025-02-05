"""LlamaBot Notebook Explainer"""

from pathlib import Path
from tqdm import tqdm
import typer

from llamabot.bot.simplebot import SimpleBot
from llamabot.prompt_manager import prompt


@prompt(role="system")
def notebook_bot_sysprompt():
    """You are an expert at explaining Jupyter notebook code.
    You will be given a notebook code cell and the rest of the notebook.
    You will generate the necessary markdown to explain the code to a non-technical audience.
    Maximum of 3 sentences per Markdown cell.
    Be brief and concise.
    Use a conversational tone.
    """


explanation_description = """
The explanation of the code cell. Should be at most one paragraph."""


def explainer_bot(model_name: str = "o1-preview") -> SimpleBot:
    """Create a bot that explains the code in a notebook.

    :param model_name: The name of the model to use.
    :return: A bot that explains the code in a notebook.
    """
    bot = SimpleBot(
        system_prompt=notebook_bot_sysprompt(),
        model_name=model_name,
    )
    return bot


app = typer.Typer()


@prompt(role="user")
def provide_content(cell_source: str, previous_cells: list, upcoming_cells: list):
    """Here is the code context for you to work with:

    -----

    [[CELL SOURCE TO EXPLAIN BEGINS]]
    {{ cell_source }}
    [[CELL SOURCE TO EXPLAIN END]]

    -----

    Here are the previous cells in the notebook, including generated explanations:

    [[PREVIOUS CELLS START]]
    {% for cell in previous_cells %}
    {{ cell.source }}
    {% endfor %}
    [[PREVIOUS CELLS END]]

    -----

    Here are the upcoming cells in the notebook:

    [[UPCOMING CELLS START]]
    {% for cell in upcoming_cells %}
    {{ cell.source }}
    {% endfor %}
    [[UPCOMING CELLS END]]

    -----

    Please provide a Markdown explanation as instructed.
    Take into account the previously generated explanations
    and upcoming cells when writing your response.
    Examples of openers include:

    - We first start with... (At the beginning of the notebook!)
    - We're now going to...
    - Now, we will...
    - By doing..., we will see that..., so now we will...
    - Having done..., we are now going to...
    - In the following code cell, we will...
    - To understand..., we will...
    - To see why..., we will...
    - To build on that, we will...
    - Finally, we will... (At the end of the notebook!)"

    Be sure to vary the opening of the explanation
    to ensure that it doesn't sound repetitive.
    """


@app.command()
def explain(
    notebook_path: Path, overwrite: bool = False, model_name: str = "o1-preview"
):
    """
    Explain the code cells in a Jupyter notebook and create a new notebook with explanations.

    :param notebook_path: Path to the input Jupyter notebook file.
    :param overwrite: If True, overwrite the original notebook. If False, create a new file with '_explained' suffix.
    """
    try:
        import nbformat
    except ImportError:
        raise ImportError(
            "nbformat is not installed. Please install it with `pip install nbformat`."
        )

    # Read the Jupyter notebook
    with open(notebook_path, "r", encoding="utf-8") as notebook_file:
        notebook = nbformat.read(notebook_file, as_version=4)

    # Create a new notebook object to store the modified notebook
    new_notebook = nbformat.v4.new_notebook()

    # Copy the metadata from the original notebook
    new_notebook.metadata = notebook.metadata.copy()

    # Iterate through the cells of the original notebook
    for i, cell in enumerate(tqdm(notebook.cells)):
        if cell.cell_type == "code" and len(cell.source.strip()) > 0:
            # Prepare context for the current cell
            previous_cells = new_notebook.cells
            upcoming_cells = notebook.cells[i + 1 :]

            # Generate explanation for the current cell
            bot = explainer_bot(model_name)
            response = bot(provide_content(cell.source, previous_cells, upcoming_cells))

            # Create a new markdown cell with the explanation
            explanation_cell = nbformat.v4.new_markdown_cell(source=response.content)

            # Add the explanation cell to the new notebook
            new_notebook.cells.append(explanation_cell)

        # Add the original cell to the new notebook
        new_notebook.cells.append(cell)

    if overwrite:
        with open(notebook_path, "w", encoding="utf-8") as notebook_file:
            nbformat.write(new_notebook, notebook_file)
    else:
        # Write to <notebook_path>_explained.ipynb
        explained_notebook_path = f"{notebook_path}_explained.ipynb"
        with open(explained_notebook_path, "w", encoding="utf-8") as notebook_file:
            nbformat.write(new_notebook, notebook_file)
