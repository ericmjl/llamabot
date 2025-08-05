# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LlamaBot is a Pythonic interface to LLMs that makes it easier to experiment with LLMs in Jupyter notebooks and build Python apps. It supports all models from LiteLLM and provides a modular architecture with different bot types, CLI tools, and web interfaces.

## Development Environment

**Package Manager**: This project uses `pixi` for dependency management and environment setup.

**Key Commands**:
- `pixi run test` - Run the test suite with pytest
- `pixi run docs` - Serve documentation locally with mkdocs
- `pixi run build-docs` - Build documentation
- `pixi run jlab` - Start Jupyter Lab (requires notebooks feature)
- `pixi run llamabot-cli` - Test the CLI help command
- `pytest tests/path/to/specific_test.py` - Run a single test file
- `pytest tests/path/to/specific_test.py::test_function` - Run a specific test function

**Environment Setup**: Use `pixi shell` to enter the development environment, or prefix commands with `pixi run`.

**Pre-commit Hooks**: The project uses pre-commit hooks with Black, Ruff, interrogate (docstring coverage), pydoclint, and other tools. Hooks run automatically on commit.

## Core Architecture

### Bot Hierarchy

The main bot classes follow a compositional pattern:

- **SimpleBot** (`llamabot/bot/simplebot.py`) - Stateless function-like bot for single interactions
- **StructuredBot** (`llamabot/bot/structuredbot.py`) - Bot with structured/JSON output capabilities
- **QueryBot** (`llamabot/bot/querybot.py`) - RAG-enabled bot for document querying
- **AgentBot** (`llamabot/bot/agentbot.py`) - Tool-using agent with function calling
- **ImageBot** (`llamabot/bot/imagebot.py`) - Image generation bot
- **KGBot** (`llamabot/bot/kgbot.py`) - Knowledge graph-aware bot

### Component System

The `llamabot/components/` directory contains modular, composable components:

- **Messages** (`messages.py`) - Unified message types (SystemMessage, HumanMessage, AIMessage, etc.)
- **DocStore** (`docstore.py`) - Pluggable document storage (LanceDBDocStore, BM25DocStore, ChromaDBDocStore)
- **Chat Memory** (`chat_memory.py`) - Conversation memory including graph-based threading
- **Tools** (`tools.py`) - Agent function calling framework with `@tool` decorator
- **Sandbox** (`sandbox.py`) - Docker-based secure code execution
- **History** (`history.py`) - Message persistence and retrieval
- **API/UI Mixins** (`api.py`, `chatui.py`) - FastAPI and Panel chat interface integration

### CLI Structure

The CLI is built with Typer and organized in `llamabot/cli/`:

- **Main entry point**: `llamabot.cli:app` (defined in pyproject.toml)
- **Key commands**: blog, configure, doc, docs, git, logviewer, notebook, python, repo, tutorial
- **Utilities**: Common CLI utilities in `utils.py`

## Development Patterns

### Code Style

- **Functional over OOP**: Prefer functional programming except for Bot classes (PyTorch-like parameterized callables)
- **Docstrings**: Use Sphinx-style arguments (`:param arg: description`)
- **Testing**: Always add tests when making code changes (tests mirror source structure in `tests/` directory)
- **Linting**: Automatic linting tools handle formatting (don't worry about linting errors during development)
- **File Editing**: When possible, only edit the requested file; avoid unnecessary changes to other files

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
- **HTMX**: Client-side JS must be re-initialized after HTMX swaps using initialization functions
- **Never duplicate UI components** - always use macros for shared components
- **Dynamic HTML**: When using HTMX/Turbo, encapsulate JS initialization in functions and call after content swaps (inline `<script>` tags in swapped HTML are not executed)
- **Initialization Pattern**: Make init functions idempotent and call on both page load and after dynamic content loads

### Testing

- **Framework**: pytest with coverage reporting
- **Config**: Tests run with `pytest -v --cov --cov-report term-missing -m 'not llm_eval' --durations=10`
- **Exclusions**: Tests marked with `llm_eval` are excluded from default runs
- **Structure**: Tests mirror the source structure in the `tests/` directory

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
