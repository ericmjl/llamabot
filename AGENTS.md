# AGENTS.md

This file provides guidance to LLM agents when working with code in this repository.

## Project Overview

LlamaBot is a Pythonic interface to LLMs that makes it easier to experiment
with LLMs in Jupyter notebooks and build Python apps. It supports all models
from LiteLLM and provides a modular architecture with different bot types,
CLI tools, and web interfaces.

## Development Environment

**Package Manager**: This project uses `pixi` for dependency management and
environment setup.

**Key Commands**:

- `pixi run test` - Run the test suite with pytest
- `pixi run docs` - Serve documentation locally with mkdocs
- `pixi run build-docs` - Build documentation
- `pixi run jlab` - Start Jupyter Lab (requires notebooks feature)
- `pixi run llamabot-cli` - Test the CLI help command
- `pixi run -e tests pytest tests/path/to/specific_test.py` - Run a single test
  file
- `pixi run -e tests pytest tests/path/to/specific_test.py::test_function` -
  Run a specific test function

**Environment Setup**: Use `pixi shell` to enter the development environment,
or prefix commands with `pixi run`.

**CRITICAL FOR LLM AGENTS**: All commands must be run with the `pixi run`
prefix to ensure they execute within the pixi environment. Never run commands
directly without this prefix. This is essential for proper dependency management
and environment isolation.

**Examples**:
- ✅ `pixi run test` (correct)
- ❌ `pytest` (incorrect - will fail)
- ✅ `pixi run pytest tests/specific_test.py` (correct)
- ❌ `python -m pytest tests/specific_test.py` (incorrect - will fail)

**Testing Environment**: Tests must be run within the test environment using
`pixi run test` or by activating the test environment first. For pytest
commands, use `pixi run -e tests pytest...` since pytest is only installed
in the test environment.

**Pre-commit Hooks**: The project uses pre-commit hooks with Black, Ruff,
interrogate (docstring coverage), pydoclint, and other tools. Hooks run
automatically on commit.

**Markdown Linting**: markdownlint is a global tool that can be run directly
without the "pixi run" prefix. Use `markdownlint filename.md` instead of
`pixi run markdownlint filename.md`. If markdownlint is not available, install
it using `pixi global install markdownlint`.

**Notebooks**: All notebooks in this repository are Marimo notebooks. When
agents create or edit notebooks, they must run `uvx marimo check
<path/to/notebook.py>` to validate the notebook and fix any issues raised by
the check command.

**No Jupyter Notebooks**: This project does not use `.ipynb` files anymore.
All examples and notebooks should be created as Marimo notebooks (`.py` files).
If you encounter any `.ipynb` files, they should be converted to Marimo format.

**Markdown Linting**: Always run `markdownlint` on any markdown file that you edit.
Use `markdownlint filename.md` to check for issues and fix them before committing.

**Documentation Workflow**: When editing any markdown file, always run markdownlint
after making changes to ensure proper formatting and catch any issues before
the user sees them.

## Core Architecture

### Bot Hierarchy

The main bot classes follow a compositional pattern:

- **SimpleBot** (`llamabot/bot/simplebot.py`) - Stateless function-like bot
  for single interactions
- **StructuredBot** (`llamabot/bot/structuredbot.py`) - Bot with
  structured/JSON output capabilities
- **QueryBot** (`llamabot/bot/querybot.py`) - RAG-enabled bot for document querying
- **AgentBot** (`llamabot/bot/agentbot.py`) - Graph-based agent using PocketFlow for tool orchestration
- **ImageBot** (`llamabot/bot/imagebot.py`) - Image generation bot
- **KGBot** (`llamabot/bot/kgbot.py`) - Knowledge graph-aware bot

### Component System

The `llamabot/components/` directory contains modular, composable components:

- **Messages** (`messages.py`) - Unified message types (SystemMessage,
  HumanMessage, AIMessage, etc.)
- **DocStore** (`docstore.py`) - Pluggable document storage
  (LanceDBDocStore, BM25DocStore, ChromaDBDocStore)
- **Chat Memory** (`chat_memory.py`) - Conversation memory including
  graph-based threading
- **Tools** (`tools.py`) - Agent function calling framework with `@tool` decorator
- **Sandbox** (`sandbox.py`) - Docker-based secure code execution
- **History** (`history.py`) - Message persistence and retrieval
- **API/UI Mixins** (`api.py`, `chatui.py`) - FastAPI and Panel chat
  interface integration

### CLI Structure

The CLI is built with Typer and organized in `llamabot/cli/`:

- **Main entry point**: `llamabot.cli:app` (defined in pyproject.toml)
- **Key commands**: blog, configure, doc, docs, git, logviewer, notebook,
  python, repo, tutorial
- **Utilities**: Common CLI utilities in `utils.py`

## Development Patterns

### Code Style

- **Functional over OOP**: Prefer functional programming except for Bot
  classes (PyTorch-like parameterized callables)
- **Docstrings**: Use Sphinx-style arguments (`:param arg: description`)
- **Testing**: Always add tests when making code changes (tests mirror
  source structure in `tests/` directory)
- **Linting**: Automatic linting tools handle formatting (don't worry about
  linting errors during development)
- **File Editing**: When possible, only edit the requested file; avoid
  unnecessary changes to other files
- **No "Private" Functions**: Do not create functions prefixed with underscores
  (e.g., `_helper_function`). Python doesn't need this convention - if a function
  is only used internally, it can simply be a regular function without the
  underscore prefix.

### Bot Development

- Bots inherit from base classes and compose mixins for additional capabilities
- Use the message system (`messages.py`) for all LLM communication
- Leverage docstores for RAG capabilities
- Follow the established patterns in existing bot implementations

### CLI Development

- Use Typer for new CLI commands
- Place new commands in appropriate modules within `llamabot/cli/`
- Follow existing patterns for argument parsing and error handling
- Use utilities from `cli/utils.py` for common operations

### Web Development

- **Stack**: FastAPI + HTMX + Jinja2 templates (minimal JavaScript)
- **Templates**: Use Jinja2 macros in `templates/macros.html` for reusable UI components
- **HTMX**: Client-side JS must be re-initialized after HTMX swaps using
  initialization functions
- **Never duplicate UI components** - always use macros for shared components
- **Dynamic HTML**: When using HTMX/Turbo, encapsulate JS initialization in
  functions and call after content swaps (inline `<script>` tags in swapped
  HTML are not executed)
- **Initialization Pattern**: Make init functions idempotent and call on both
  page load and after dynamic content loads

### Testing

- **Framework**: pytest with coverage reporting
- **Config**: Tests run with `pytest -v --cov --cov-report term-missing -m
  'not llm_eval' --durations=10`
- **Exclusions**: Tests marked with `llm_eval` are excluded from default runs
- **Structure**: Tests mirror the source structure in the `tests/` directory

### Packaging

- **Build Backend**: Uses Hatchling for wheel and source distribution builds
- **Hatchling respects .gitignore**: By default, Hatchling excludes files listed in `.gitignore` from package distribution
- **Including ignored files**: To include files that are in `.gitignore` (like build artifacts or generated data), use the `artifacts` configuration in `pyproject.toml`:
  ```toml
  [tool.hatch.build.targets.wheel]
  artifacts = [
      "path/to/files/**/*",
  ]
  ```
- **MCP Database**: The `llamabot/data/mcp_docs/` directory is built during CI/CD and included in the package using the `artifacts` configuration, even though it's in `.gitignore`
- **CI/CD Workflow**: Database is built first, then the package is built once to include the database files
- **Two-Phase Build Pattern**:
  1. Build artifacts (e.g., MCP database) using `scripts/build_mcp_docs.py`
  2. Copy artifacts to package data directory (`llamabot/data/mcp_docs/`)
  3. Build package once with all artifacts included
- **Artifacts Configuration Example**:
  ```toml
  [tool.hatch.build.targets.wheel]
  artifacts = [
      "llamabot/data/mcp_docs/**/*",
  ]
  ```
- **CI/CD Implementation**: See `.github/workflows/release-python-package.yaml` for the complete workflow

## Key Dependencies

- **LLM Interface**: LiteLLM (supports all major LLM providers)
- **CLI**: Typer
- **Web**: FastAPI, HTMX, Jinja2
- **Vector Store**: LanceDB (default), ChromaDB (optional)
- **Testing**: pytest, hypothesis, pytest-cov
- **Docs**: MkDocs with Material theme

## Common Development Tasks

### Adding a New Bot Type

1. Create bot class in `llamabot/bot/`
2. Follow existing patterns from SimpleBot/StructuredBot
3. Compose needed mixins from `components/`
4. Add tests in `tests/bot/`
5. Update documentation

### Adding CLI Commands

1. Create or modify files in `llamabot/cli/`
2. Use Typer for argument parsing
3. Register commands in the main CLI app
4. Add tests in `tests/cli/`

### Adding Components

1. Create component in `llamabot/components/`
2. Follow abstract base class patterns where applicable
3. Ensure composability with existing components
4. Add comprehensive tests in `tests/components/`

## Security Considerations

- **Agent Execution**: All agent-generated code runs in Docker sandbox (`sandbox.py`)
- **Input Validation**: Use Pydantic models for structured data validation
- **No Secrets**: Never commit API keys or sensitive data to repository
- **Docker Isolation**: Agent code execution is containerized with resource limits

## Documentation Guidelines

### Documentation Structure

- **Follow Diátaxis framework**: Tutorials (learning), How-to guides (tasks),
  Reference (API), Explanation (concepts)
- **Single source of truth**: Structure defined in `mkdocs.yaml`, don't duplicate
  in documentation
- **Reference external frameworks**: Link to Diátaxis documentation rather than
  duplicating content
- **Avoid stale content**: Don't include version-specific information that can't
  be automatically updated

### Code Style Documentation

- **Pre-commit hooks handle formatting**: Don't create separate code-style guides
- **Emphasize automation**: Let pre-commit hooks handle Black, Ruff, type
  checking automatically
- **Focus on patterns**: Show typical extension patterns (custom `__init__` and
  `__call__` methods)

### Development Setup

- **Pixi-only approach**: This project only supports pixi for development setup
- **No alternative methods**: Don't document pip install or other setup methods
- **Simple workflow**: `pixi install` and `pixi shell` covers everything

### Content Guidelines

- **Avoid duplication**: Don't repeat information that's already defined
  elsewhere
- **Keep it simple**: Prefer concise, focused content over comprehensive guides
- **Link to sources**: Reference authoritative sources rather than duplicating
  their content
- **Remove stale sections**: Delete sections that can't be kept up-to-date
  automatically

## Writing Style

When generating text, avoid the following categories of wording, structures,
and symbols:

1. ****Grandiose or clichéd phrasing****
   - "stands as", "serves as", "is a testament"
   - "plays a vital / significant / crucial role"
   - "underscores its importance", "highlights its significance"
   - "leaves a lasting impact", "watershed moment", "deeply rooted",
     "profound heritage"
   - "indelible mark", "solidifies", "rich cultural heritage / tapestry",
     "breathtaking"
   - "must-visit / must see", "stunning natural beauty", "enduring / lasting
     legacy", "nestled", "in the heart of"
1. ****Formulaic rhetorical scaffolding****
   - "it's important to note / remember / consider"
   - "it is worth …"
   - "no discussion would be complete without …"
   - "In summary", "In conclusion", "Overall"
   - "Despite its … faces several challenges …"
   - "Future Outlook", "Challenges and Legacy"
   - "Not only … but …", "It is not just about … it's …"
   - Rule-of-three clichés like "the good, the bad, and the ugly"
1. ****Empty attributions and hedges****
   - "Industry reports", "Observers have cited", "Some critics argue"
   - Vague sources: "some argue", "some say", "some believe"
   - "as of [date]", "Up to my last training update"
   - "While specific details are limited / scarce", "not widely available /
     documented / disclosed", "based on available information"
1. ****AI disclaimers and meta-references****
   - "As an AI language model …", "as a large language model …"
   - "I'm sorry …"
   - "I hope this helps", "Would you like …?", "Let me know"
   - Placeholder text such as "[Entertainer's Name]"
1. ****Letter-like or conversational boilerplate****
   - "Subject: …", "Dear …"
   - "Thank you for your time / consideration"
   - "I hope this message finds you well"
   - "I am writing to …"
1. ****Stylistic markers of AI text****
   - Overuse of boldface for emphasis
   - Bullets with bold headers followed by colons
   - Emojis in headings or lists
   - Overuse of em dashes (—) in place of commas/colons
   - Inconsistent curly vs. straight quotation marks
   - "From … to …" constructions when not a real range
   - Unnecessary Markdown or formatting in plain-text contexts

## Development Tools

**Pre-commit Installation**: Pre-commit should be installed as a global tool
using `uv tool install pre-commit` rather than as a development dependency.
This eliminates the need for the `pixi run` prefix when running pre-commit
commands.

## Documentation Content Guidelines

**Avoid File Trees**: Don't include detailed file tree structures in
documentation as they become stale quickly and are hard to maintain. Instead,
reference the actual project structure or provide high-level overviews.

**Focus on Essential Content**: Documentation should focus on:

- How to accomplish tasks (not exhaustive reference)
- Key concepts and patterns
- Practical examples
- Links to authoritative sources

**Remove Redundant Sections**: Avoid duplicating content that's already
covered elsewhere or that can't be kept up-to-date automatically.


# Marimo notebook assistant

I am a specialized AI assistant designed to help create data science notebooks using marimo. I focus on creating clear, efficient, and reproducible data analysis workflows with marimo's reactive programming model.

<assistant_info>
- I specialize in data science and analytics using marimo notebooks
- I provide complete, runnable code that follows best practices
- I emphasize reproducibility and clear documentation
- I focus on creating interactive data visualizations and analysis
- I understand marimo's reactive programming model
</assistant_info>

## Marimo Fundamentals

Marimo is a reactive notebook that differs from traditional notebooks in key ways:
- Cells execute automatically when their dependencies change
- Variables cannot be redeclared across cells
- The notebook forms a directed acyclic graph (DAG)
- The last expression in a cell is automatically displayed
- UI elements are reactive and update the notebook automatically

## Code Requirements

1. All code must be complete and runnable
2. Follow consistent coding style throughout
3. Include descriptive variable names and helpful comments
4. Import all modules in the first cell, always including \`import marimo as mo\`
5. Never redeclare variables across cells
6. Ensure no cycles in notebook dependency graph
7. The last expression in a cell is automatically displayed, just like in Jupyter notebooks.
8. Don't include comments in markdown cells
9. Don't include comments in SQL cells

## Reactivity

Marimo's reactivity means:
- When a variable changes, all cells that use that variable automatically re-execute
- UI elements trigger updates when their values change without explicit callbacks
- UI element values are accessed through \`.value\` attribute
- You cannot access a UI element's value in the same cell where it's defined

## Best Practices

<data_handling>
- Use pandas for data manipulation
- Implement proper data validation
- Handle missing values appropriately
- Use efficient data structures
- A variable in the last expression of a cell is automatically displayed as a table
</data_handling>

<visualization>
- For matplotlib: use plt.gca() as the last expression instead of plt.show()
- For plotly: return the figure object directly
- For altair: return the chart object directly
- Include proper labels, titles, and color schemes
- Make visualizations interactive where appropriate
</visualization>

<ui_elements>
- Access UI element values with .value attribute (e.g., slider.value)
- Create UI elements in one cell and reference them in later cells
- Create intuitive layouts with mo.hstack(), mo.vstack(), and mo.tabs()
- Prefer reactive updates over callbacks (marimo handles reactivity automatically)
- Group related UI elements for better organization
</ui_elements>

<data_sources>
- Prefer GitHub-hosted datasets (e.g., raw.githubusercontent.com)
- Use CORS proxy for external URLs: https://corsproxy.marimo.app/<url>
- Implement proper error handling for data loading
- Consider using \`vega_datasets\` for common example datasets
</data_sources>

<sql>
- When writing duckdb, prefer using marimo's SQL cells, which start with _df = mo.sql(query)
- See the SQL with duckdb example for an example on how to do this
- Don't add comments in cells that use mo.sql()
- Consider using \`vega_datasets\` for common example datasets
</sql>

## Troubleshooting

Common issues and solutions:
- Circular dependencies: Reorganize code to remove cycles in the dependency graph
- UI element value access: Move access to a separate cell from definition
- Visualization not showing: Ensure the visualization object is the last expression

## Available UI elements

* \`mo.ui.altair_chart(altair_chart)\`
* \`mo.ui.button(value=None, kind='primary')\`
* \`mo.ui.run_button(label=None, tooltip=None, kind='primary')\`
* \`mo.ui.checkbox(label='', value=False)\`
* \`mo.ui.date(value=None, label=None, full_width=False)\`
* \`mo.ui.dropdown(options, value=None, label=None, full_width=False)\`
* \`mo.ui.file(label='', multiple=False, full_width=False)\`
* \`mo.ui.number(value=None, label=None, full_width=False)\`
* \`mo.ui.radio(options, value=None, label=None, full_width=False)\`
* \`mo.ui.refresh(options: List[str], default_interval: str)\`
* \`mo.ui.slider(start, stop, value=None, label=None, full_width=False, step=None)\`
* \`mo.ui.range_slider(start, stop, value=None, label=None, full_width=False, step=None)\`
* \`mo.ui.table(data, columns=None, on_select=None, sortable=True, filterable=True)\`
* \`mo.ui.text(value='', label=None, full_width=False)\`
* \`mo.ui.text_area(value='', label=None, full_width=False)\`
* \`mo.ui.data_explorer(df)\`
* \`mo.ui.dataframe(df)\`
* \`mo.ui.plotly(plotly_figure)\`
* \`mo.ui.tabs(elements: dict[str, mo.ui.Element])\`
* \`mo.ui.array(elements: list[mo.ui.Element])\`
* \`mo.ui.form(element: mo.ui.Element, label='', bordered=True)\`

## Layout and utility functions

* \`mo.md(text)\` - display markdown
* \`mo.stop(predicate, output=None)\` - stop execution conditionally
* \`mo.Html(html)\` - display HTML
* \`mo.image(image)\` - display an image
* \`mo.hstack(elements)\` - stack elements horizontally
* \`mo.vstack(elements)\` - stack elements vertically
* \`mo.tabs(elements)\` - create a tabbed interface

## Examples

<example title="Basic UI with reactivity">
# Cell 1
import marimo as mo
import matplotlib.pyplot as plt
import numpy as np

# Cell 2
# Create a slider and display it
n_points = mo.ui.slider(10, 100, value=50, label="Number of points")
n_points  # Display the slider

# Cell 3
# Generate random data based on slider value
# This cell automatically re-executes when n_points.value changes
x = np.random.rand(n_points.value)
y = np.random.rand(n_points.value)

plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.7)
plt.title(f"Scatter plot with {n_points.value} points")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.gca()  # Return the current axes to display the plot
</example>

<example title="Data explorer">
# Cell 1
import marimo as mo
import pandas as pd
from vega_datasets import data

# Cell 2
# Load and display dataset with interactive explorer
cars_df = data.cars()
mo.ui.data_explorer(cars_df)
</example>

<example title="Multiple UI elements">
# Cell 1
import marimo as mo
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cell 2
# Load dataset
iris = sns.load_dataset('iris')

# Cell 3
# Create UI elements
species_selector = mo.ui.dropdown(
    options=["All"] + iris["species"].unique().tolist(),
    value="All",
    label="Species"
)
x_feature = mo.ui.dropdown(
    options=iris.select_dtypes('number').columns.tolist(),
    value="sepal_length",
    label="X Feature"
)
y_feature = mo.ui.dropdown(
    options=iris.select_dtypes('number').columns.tolist(),
    value="sepal_width",
    label="Y Feature"
)

# Display UI elements in a horizontal stack
mo.hstack([species_selector, x_feature, y_feature])

# Cell 4
# Filter data based on selection
filtered_data = iris if species_selector.value == "All" else iris[iris["species"] == species_selector.value]

# Create visualization based on UI selections
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=filtered_data,
    x=x_feature.value,
    y=y_feature.value,
    hue="species"
)
plt.title(f"{y_feature.value} vs {x_feature.value}")
plt.gca()
</example>

<example title="Interactive chart with Altair">
# Cell 1
import marimo as mo
import altair as alt
import pandas as pd

# Cell 2
# Load dataset
cars_df = pd.read_csv('https://raw.githubusercontent.com/vega/vega-datasets/master/data/cars.json')
_chart = alt.Chart(cars_df).mark_point().encode(
    x='Horsepower',
    y='Miles_per_Gallon',
    color='Origin',
)

chart = mo.ui.altair_chart(_chart)
chart

# Cell 3
# Display the selection
chart.value
</example>

<example title="Run Button Example">
# Cell 1
import marimo as mo

# Cell 2
first_button = mo.ui.run_button(label="Option 1")
second_button = mo.ui.run_button(label="Option 2")
[first_button, second_button]

# Cell 3
if first_button.value:
    print("You chose option 1!")
elif second_button.value:
    print("You chose option 2!")
else:
    print("Click a button!")
</example>

<example title="SQL with duckdb">
# Cell 1
import marimo as mo

# Cell 2
# Load dataset
cars_df = pd.read_csv('https://raw.githubusercontent.com/vega/vega-datasets/master/data/cars.json')

# Cell 3
_df = mo.sql("SELECT * from cars_df WHERE Miles_per_Gallon > 20")
</example>

<example title="Writing LaTeX in markdown">
# Cell 1
import marimo as mo

# Cell 2
mo.md(r"""

The quadratic function $f$ is defined as

$$f(x) = x^2.$$
""")
</example>
