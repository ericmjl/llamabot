# Building LLM Agents Made Simple

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/ericmjl/llamabot/blob/main/tutorials/pydata-boston-2025/backoffice.py)

Learn to build practical LLM agents using LlamaBot and Marimo notebooks. The most important lesson: **start with workflows, not technology**.

## Overview

This tutorial demonstrates how to build a complete back-office automation system through three agents:

- **Receipt processor** - Extracts structured data from PDF receipts
- **Invoice writer** - Generates formatted invoices from natural language
- **Coordinator** - Orchestrates both agents to handle complex workflows

This demonstrates the fundamental pattern: map your boring workflows first, build focused agents for specific tasks, then compose them.

## Files

- **`backoffice.py`** - Main tutorial notebook covering agent development with a workflow-first approach
- **`tool_calling_benchmark.py`** - Benchmark notebook testing tool calling accuracy across multiple LLM models
- **`receipt_*.pdf`** - Sample receipt PDFs for testing the receipt processor

## Key Concepts

### Workflow-First Development

Before building agents, you must:

1. **Map your workflow** - Understand the decision points and data flow
2. **Define schemas** - Create Pydantic models for structured data
3. **Build focused agents** - Each agent handles one specific task
4. **Compose agents** - Use a coordinator to orchestrate multiple agents

### Two-Step OCR Pattern

Vision models like DeepSeek-OCR excel at OCR but don't support structured outputs. The solution:

1. **OCR Step** - Extract text from images using vision models
2. **Structuring Step** - Convert unstructured text to validated Pydantic models

This pattern lets you use specialized OCR models while still getting structured, validated output.

## Prerequisites

- Python 3.13+
- Access to a Modal-hosted Ollama endpoint (configured in the notebook)
- Dependencies are managed via PEP 723 inline script metadata

## Running the Tutorial

The notebooks use PEP 723 inline script metadata for dependency management. You can run them directly with:

```bash
uv run backoffice.py
```

Or open them in Marimo:

```bash
marimo edit backoffice.py
```

## What You'll Learn

- How to map workflows before building agents
- Building specialized agents for specific tasks (receipt processing, invoice generation)
- Composing agents with a coordinator
- Using vision models for OCR with structured output validation
- Tool calling patterns and best practices

## Benchmark Notebook

The `tool_calling_benchmark.py` notebook evaluates tool calling accuracy across multiple LLM models:

- `qwen2.5:32b`
- `deepseek-r1:32b`
- `llama3.1:70b`
- `qwen2.5:72b-q4_K_M`
- `gpt-4.1` (baseline)

It tests how well each model selects the correct tool from a growing set of available tools (1-4 tools).
