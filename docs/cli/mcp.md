# MCP Server

The LlamaBot MCP (Model Context Protocol) server provides AI coding agents with
access to LlamaBot's comprehensive documentation and source code through semantic
search.

## Features

- **Semantic Search**: Find relevant documentation using natural language
  queries
- **Comprehensive Coverage**: Includes all documentation, tutorials, and
  source code docstrings
- **Fast Performance**: Pre-built database for instant startup
- **MCP Compatible**: Works with Cursor, VSCode, and other MCP-enabled tools

## Installation

The MCP server comes with a pre-built documentation database, so no additional
setup is required.

### Configure Your Coding Tool

#### Cursor

Add to your MCP settings:

```json
{
  "mcpServers": {
    "llamabot-docs": {
      "command": "uvx",
      "args": ["--with", "llamabot[all]", "llamabot", "mcp", "launch"]
    }
  }
}
```

#### VSCode

Add to your MCP configuration:

```json
{
  "mcpServers": {
    "llamabot-docs": {
      "command": "uvx",
      "args": ["--with", "llamabot[all]", "llamabot", "mcp", "launch"]
    }
  }
}
```

#### Other Tools

Use `uvx --with llamabot[all] llamabot mcp launch` as your MCP server command.

## Usage

Once configured, you can use the `docs_search` tool in your AI coding assistant:

- **Query**: Natural language search (e.g., "How do I create a SimpleBot?")
- **Results**: Returns relevant documentation snippets with metadata
- **Limit**: Control number of results (default: 5)

## Commands

### `llamabot mcp build`

Build a custom documentation database from GitHub and local source code.

**When to use:**

- You need the latest documentation (not the packaged version)
- You're developing locally and want fresh docs
- You want to customize the documentation sources

**Options:**

- Fetches latest docs from GitHub
- Extracts Python docstrings
- Creates semantic search index
- Stores in `~/.llamabot/mcp_docs/`

### `llamabot mcp launch`

Launch the MCP server for AI coding tools.

**Options:**

- `--host`: Host to bind to (default: 0.0.0.0)
- `--port`: Port to run on (default: 8765)

## Database Management

The MCP server uses a two-tier approach:

1. **User Database** (`~/.llamabot/mcp_docs/`): Built with `llamabot mcp build`
2. **Packaged Database**: Pre-built and included with the wheel

The server automatically uses the user database if it exists, otherwise falls back
to the packaged database. Most users will use the packaged database.

## Troubleshooting

### Database Not Found

If you see "LlamaBot documentation database not found":

```bash
llamabot mcp build
```

### Slow Startup

The first launch may be slow due to model loading. Subsequent launches are much faster.

### Outdated Documentation

To get the latest documentation:

```bash
llamabot mcp build
```

This rebuilds the database with the most recent content from GitHub.

## Examples

### Search for Bot Creation

Query: "How do I create a SimpleBot?"

Returns: Documentation about SimpleBot initialization, examples, and usage patterns.

### Find API Documentation

Query: "What are the parameters for StructuredBot?"

Returns: API documentation, parameter descriptions, and usage examples.

### Get Tutorial Information

Query: "Show me the tutorial for QueryBot"

Returns: Tutorial content, step-by-step instructions, and code examples.
