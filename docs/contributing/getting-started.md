# Getting Started with Contributing

Welcome! This guide will help you set up your development environment for LlamaBot using pixi, our dependency management tool.

## Prerequisites

Before you begin, make sure you have:

- **pixi** installed ([installation instructions](https://pixi.sh/latest/))

!!! note "Python Installation"
    You don't need to install Python separately. Pixi will manage the Python version for you (this project requires Python 3.9-3.12).

## Initial Setup

### 1. Clone the Repository

```bash
git clone https://github.com/ericmjl/llamabot.git
cd llamabot
```

### 2. Install Dependencies with Pixi

This project uses [pixi](https://pixi.sh/) for dependency management. Install all dependencies:

```bash
pixi install
```

This will create a pixi environment with all the necessary dependencies, including:

- Core dependencies (defined in `[project.dependencies]`)
- Optional dependencies for notebooks, RAG, agents, and CLI tools
- Development tools (pytest, pre-commit, etc.)
- Documentation tools (mkdocs, etc.)

### 3. Activate the Development Environment

You have two options:

#### Option A: Use pixi shell (recommended)

```bash
pixi shell
```

This activates the pixi environment in your current shell.

#### Option B: Prefix commands with `pixi run`

```bash
pixi run python -c "import llamabot"
```

!!! important "Always use pixi"
    All commands must be run with the `pixi run` prefix or within a `pixi shell` session. This ensures proper dependency management and environment isolation.

## Development Workflow

### Running Tests

Run the full test suite:

```bash
pixi run test
```

Run a specific test file:

```bash
pixi run -e tests pytest tests/path/to/specific_test.py
```

Run a specific test function:

```bash
pixi run -e tests pytest tests/path/to/specific_test.py::test_function
```

!!! note "Test Environment"
    Tests run in a separate environment (`tests`). Use `pixi run -e tests` when running pytest commands directly.

### Running the CLI

Test that the CLI is working:

```bash
pixi run llamabot-cli
```

This runs `llamabot --help` to verify the CLI is properly installed.

### Working with Documentation

Serve documentation locally:

```bash
pixi run docs
```

This starts a local MkDocs server (usually at `http://127.0.0.1:8000`).

Build documentation:

```bash
pixi run build-docs
```

!!! note "Marimo Notebooks"
    This project uses [Marimo notebooks](https://marimo.io/) (`.py` files), not traditional Jupyter notebooks (`.ipynb` files). When creating or editing notebooks, always run `uvx marimo check <path/to/notebook.py` to validate them.

### Building MCP Documentation

Build the MCP documentation database:

```bash
pixi run build-mcp-docs
```

## Code Quality

### Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality. Pre-commit should be installed globally:

```bash
uv tool install pre-commit
```

Install the hooks:

```bash
pre-commit install
```

The hooks will automatically run on commit and check for:

- Code formatting (Black)
- Linting (Ruff)
- Docstring coverage (interrogate)
- Docstring style (pydoclint)
- And more

Run hooks manually:

```bash
pre-commit run --all-files
```

### Markdown Linting

Markdown files are linted using `markdownlint`. Install it globally:

```bash
pixi global install markdownlint
```

Lint a markdown file:

```bash
markdownlint filename.md
```

Always run markdownlint on any markdown file you edit.

## Project Structure

### Key Directories

- `llamabot/` - Main source code
  - `bot/` - Bot implementations (SimpleBot, QueryBot, etc.)
  - `components/` - Modular components (messages, docstore, tools, etc.)
  - `cli/` - CLI commands
  - `web/` - Web interface (FastAPI + HTMX)
- `tests/` - Test suite (mirrors source structure)
- `docs/` - Documentation
- `examples/` - Example scripts
- `notebooks/` - Marimo notebooks

### Environment Features

Pixi environments are organized by features (defined in `pyproject.toml`):

- **default** - Full development environment (tests, devtools, docs, notebooks, rag, agent, cli)
- **tests** - Testing dependencies (pytest, hypothesis, etc.)
- **docs** - Documentation tools (mkdocs, etc.)
- **notebooks** - Notebook dependencies (ollama, etc.)
- **bare** - Minimal environment (devtools only)

## Common Tasks

### Adding a New Dependency

1. Add it to the appropriate section in `pyproject.toml`:
   - `[project.dependencies]` for runtime dependencies
   - `[project.optional-dependencies.<group>]` for optional dependencies
   - `[tool.pixi.feature.<feature>.pypi-dependencies]` for feature-specific dependencies
2. Run `pixi install` to update the environment

### Running Python Code

Always use pixi:

```bash
# ✅ Correct
pixi run python script.py

# ❌ Incorrect (will fail)
python script.py
```

### Installing the Package in Editable Mode

The package is automatically installed in editable mode when you run `pixi install`. You can verify:

```bash
pixi run python -c "import llamabot; print(llamabot.__file__)"
```

## Getting Help

- Check the [AGENTS.md](../../AGENTS.md) file for detailed development patterns
- Review existing code in the repository for examples
- Open an issue on GitHub for questions or problems

## Next Steps

Once your environment is set up:

1. Explore the codebase structure
2. Read the [AGENTS.md](../../AGENTS.md) for development patterns
3. Pick an issue or feature to work on
4. Write tests for your changes
5. Submit a pull request

Happy contributing!
