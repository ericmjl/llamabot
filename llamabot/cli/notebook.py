"""LlamaBot Notebook Explainer"""

from pathlib import Path
import typer
import nbformat
from llamabot import SimpleBot
from llamabot.prompt_manager import prompt


bot = SimpleBot(
    "You are an expert at explaining Jupyter notebook code. "
    "You will be given a notebook code cell. "
    "You will generate the necessary markdown to explain the code to a non-technical audience. "
    "Be brief and concise. "
    "Use a conversational tone, such as 'we're going to do this and that'"
)


app = typer.Typer()


@prompt
def provide_content(cell_source, notebook):
    """Here is the code context for you to work with:

    -----

    [[CELL SOURCE TO EXPLAIN BEGINS]]
    {{ cell_source }}
    [[CELL SOURCE TO EXPLAIN END]]

    -----

    Here is the rest of the notebook to contextualize the code cell to be explained.

    [[ALL NOTEBOOK CELLS START]]
    {% for cell in notebook.cells %}
    {{ cell.source }}
    {% endfor %}
    [[ALL NOTEBOOK CELLS END]]

    -----

    Please provide a Markdown explanation as instructed.
    """


@app.command()
def explain(notebook_path: Path, overwrite: bool = False):
    """
    Explain the code cells in a Jupyter notebook and create a new notebook with explanations.

    :param notebook_path: Path to the input Jupyter notebook file.
    :param overwrite: If True, overwrite the original notebook. If False, create a new file with '_explained' suffix.
    """
    # Read the Jupyter notebook
    with open(notebook_path, "r", encoding="utf-8") as notebook_file:
        notebook = nbformat.read(notebook_file, as_version=4)

    # Create a new notebook object to store the modified notebook
    new_notebook = nbformat.v4.new_notebook()

    # Copy the metadata from the original notebook
    new_notebook.metadata = notebook.metadata.copy()

    # Iterate through the cells of the original notebook
    for cell in notebook.cells:
        # make an explanation of the cell
        if cell.cell_type == "code":
            explanation = bot(provide_content(cell.source, notebook)).content
            # Create a new markdown cell with the explanation
            explanation_cell = nbformat.v4.new_markdown_cell(source=explanation)

            new_notebook.cells.append(explanation_cell)
        new_notebook.cells.append(cell)

    if overwrite:
        with open(notebook_path, "w", encoding="utf-8") as notebook_file:
            nbformat.write(new_notebook, notebook_file)
    else:
        # Write to <notebook_path>_explained.ipynb
        explained_notebook_path = f"{notebook_path}_explained.ipynb"
        with open(explained_notebook_path, "w", encoding="utf-8") as notebook_file:
            nbformat.write(new_notebook, notebook_file)
