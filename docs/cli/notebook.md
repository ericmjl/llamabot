---
intents:
- Provide a diataxis-style tutorial on how to use the llamabot notebook CLI.
- Explain the benefits of using the llamabot notebook CLI.
linked_files:
- llamabot/cli/notebook.py
- pyproject.toml
---

# LlamaBot Notebook CLI Tutorial

Welcome to the LlamaBot Notebook CLI tutorial. This guide will walk you through how to use the LlamaBot CLI to explain Jupyter notebook code cells in a simple and understandable way for non-technical audiences.

## Benefits of Using LlamaBot Notebook CLI

The key selling point is that it lets you, the data scientist, work in flow mode without worrying too hard about documenting your code.
Once you're ready and satisfied with your notebook, you can then use the `llamabot notebook explain` command to generate explanations for your notebook.

Using the LlamaBot Notebook CLI provides several benefits:

- **Simplification of Complex Code**: Converts complex Jupyter notebook code cells into easy-to-understand markdown explanations.
- **Educational Tool**: Enhances the learning experience for students and non-technical stakeholders by providing clear explanations of what the code does.
- **Documentation Efficiency**: Saves time and effort in documentation by automatically generating explanations, which can be particularly useful for data scientists and educators.
- **Customizable Explanations**: Offers flexibility with the `overwrite` option to either replace the original notebook or create a new explained version, catering to different documentation needs.
- **Code-First Approach**: Emphasizes the importance of code quality and documentation from the start, ensuring that the code is both functional and well-documented.

## Prerequisites

Before you begin, ensure you have `llamabot` installed:

```bash
pip install llamabot
```

## Usage

To use the LlamaBot notebook CLI, follow these steps:

1. **Prepare Your Notebook**: Ensure your Jupyter notebook is ready -- just code cells, no need for markdown cells that explain the code.
2. **Run the CLI**: Use the following command to run the CLI:

```bash
llamabot notebook explain /path/to/notebook.ipynb [--overwrite]
```

### Parameters

- `notebook_path`: The path to the Jupyter notebook file you want to explain.
- `overwrite`: Optional. If set to True, the original notebook will be overwritten with the explanations. If False, a new file with the '_explained' suffix will be created.

## How It Works

When you run the command, the CLI does the following:

- Reads the Jupyter notebook from the specified path.
- Iterates through each code cell in the notebook.
- Uses the built-in `SimpleBot` to generate a markdown explanation for each code cell.
- Appends a new markdown cell with the explanation before the original code cell in the notebook.

If the `overwrite` parameter is False, the explanations are saved in a new notebook file with the '_explained' suffix, preserving the original notebook.

## Example

Here's an example command to explain a notebook without overwriting the original file:

```bash
llamabot notebook explain /path/to/notebook.ipynb
```

This will create a new file at `/path/to/notebook_explained.ipynb` in the same directory as the original.

## Recommended Usage

We recommend that your notebook comprise code cells exclusively.
That said, leave comments within your code cells
as they'll be picked up by the LLM
and can be factored into the explanation.

Any existing markdown cells will be skipped in the explanation process.

After all, the goal here is to free you up
to focus on coding and thinking about the logic of your model,
not to have to worry about documenting it!

## Conclusion

Using the LlamaBot Notebook CLI,
you can easily provide clear explanations for Jupyter notebook code cells,
making them accessible to non-technical audiences.
This tool is particularly useful for educators, data scientists,
or anyone needing to demystify code logic in a simple and engaging way.
