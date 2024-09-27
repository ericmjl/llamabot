"""LlamaBot Notebook Explainer"""

from pathlib import Path
from pydantic import model_validator, BaseModel, Field
from tqdm import tqdm
import typer
import nbformat
from llamabot import StructuredBot
from llamabot.prompt_manager import prompt


@prompt
def notebook_bot_sysprompt():
    """You are an expert at explaining Jupyter notebook code.
    You will be given a notebook code cell and the rest of the notebook.
    You will generate the necessary markdown to explain the code to a non-technical audience.
    Maximum of 3 sentences per Markdown cell.
    Be brief and concise.
    Use a conversational tone.
    """


explanation_description = """
The explanation of the code cell. Should be at most one paragraph. Examples of openers include:

- We first start with... (At the beginnig of the notebook!)
- We're now going to...
- Now, we will...
- Having done..., we are now going to...
- By doing..., we will see that..., so now we will...
- In the following code cell, we will...
- To understand..., we will...
- To see why..., we will...
- To build on that, we will...
- Finally, we will... (At the end of the notebook!)"
"""


class NotebookExplanation(BaseModel):
    """Represents an explanation for a notebook code cell."""

    explanation: str = Field(..., description=explanation_description)

    @model_validator(mode="after")
    def ensure_only_one_paragraph(self):
        """
        Validates that the explanation is at most one paragraph.

        :raises ValueError: If the explanation contains more than 3 lines.
        :return: The validated model instance.
        """
        if len(self.explanation.split("\n")) > 3:
            raise ValueError("Explanation should be at most one paragraph.")
        return self


bot = StructuredBot(
    system_prompt=notebook_bot_sysprompt(),
    pydantic_model=NotebookExplanation,
    model_name="gpt-4-turbo",
)


app = typer.Typer()


@prompt
def provide_content(cell_source: str, notebook: nbformat.NotebookNode):
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
    for cell in tqdm(notebook.cells):
        # make an explanation of the cell
        if cell.cell_type == "code":
            response = bot(provide_content(cell.source, notebook), verbose=True)
            # Create a new markdown cell with the explanation
            explanation_cell = nbformat.v4.new_markdown_cell(
                source=response.explanation
            )

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
