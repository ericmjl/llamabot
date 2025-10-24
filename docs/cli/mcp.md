# MCP Server

The LlamaBot MCP (Model Context Protocol) server provides AI coding agents with
access to LlamaBot's comprehensive documentation and source code through semantic
search.

## Quick Start

Get started with LlamaBot MCP server in 3 simple steps:

1. **Install LlamaBot** (if not already installed):

   ```bash
   pip install "llamabot[all]"
   ```

2. **Configure your AI tool** - Choose your tool:

   - **Claude Code**: Run `claude mcp add --scope user` and use command `uvx --with llamabot[all] llamabot mcp launch`
   - **Claude Desktop**: Use Settings > Extensions > Browse extensions (or manually edit config)
   - **Cursor/VSCode**: Add the server config to your MCP settings file

3. **Start using it**: Ask your AI assistant questions like "How do I create a SimpleBot?" and it will automatically search LlamaBot documentation.

See detailed configuration instructions below for your specific tool.

## Features

- **Semantic Search**: Find relevant documentation using natural language
  queries
- **Comprehensive Coverage**: Includes all documentation, tutorials, and
  source code docstrings
- **Fast Performance**: Pre-built database for instant startup
- **MCP Compatible**: Works with Claude Desktop, Claude Code, Cursor, VSCode, and other MCP-enabled tools
- **Always Up-to-Date**: Pre-built database included with every release

## Installation

The MCP server comes with a pre-built documentation database, so no additional
setup is required beyond installing LlamaBot.

## Configuration

Choose your AI coding tool below for specific setup instructions:

### Claude Desktop

Claude Desktop supports two ways to install MCP servers: the new Extensions method (recommended for simplicity) or manual JSON configuration (for advanced users).

#### Method 1: Extensions (Recommended)

The easiest way to add MCP servers to Claude Desktop:

1. Open Claude Desktop and navigate to **Settings > Extensions**
2. Click **"Browse extensions"** to view the directory
3. Search for "LlamaBot" or manually create an extension
4. Click **"Install"** and configure any settings if needed

Note: If LlamaBot is not yet available in the extensions directory, use Method 2 below.

#### Method 2: Manual JSON Configuration

Add LlamaBot to your Claude Desktop configuration file:

1. Open Claude Desktop and click the **Settings** icon
2. Navigate to the **Developer** tab
3. Click **"Edit Config"** to open `claude_desktop_config.json`

Add this configuration:

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

Configuration file locations:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

After saving, restart Claude Desktop for changes to take effect.

### Claude Code

Claude Code provides multiple ways to add MCP servers:

#### Using the CLI (Recommended)

```bash
# Add the server interactively
claude mcp add --scope user

# When prompted:
# - Server name: llamabot-docs
# - Command: uvx
# - Args: --with llamabot[all] llamabot mcp launch
```

After adding, restart Claude Code to activate the server.

#### Manual Configuration

You can also edit the configuration file directly:

1. Open `~/.claude.json` (or the appropriate config file for your OS)
2. Add the LlamaBot server configuration:

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

#### Project-Specific Configuration

For project-specific setup, create a `.mcp.json` file in your project root:

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

This makes the MCP server shareable with your team through version control.

Useful commands:

- `claude mcp list` - Show all configured MCP servers
- `claude mcp remove llamabot-docs` - Remove the server
- Restart Claude Code after configuration changes

### Cursor

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

### VSCode

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

### Other MCP-Compatible Tools

Use `uvx --with llamabot[all] llamabot mcp launch` as your MCP server command.

## Usage

Once configured, the `docs_search` tool is automatically available to your AI assistant. You don't need to explicitly call it - just ask questions naturally and your AI assistant will use it when needed.

### How It Works

The MCP server provides a `docs_search` tool that your AI assistant can call automatically. When you ask a question, the assistant decides whether to search LlamaBot documentation and uses the tool behind the scenes.

### Example Conversations

You can ask questions like:

**Getting Started Questions:**

- "How do I create a SimpleBot?"
- "What's the difference between SimpleBot and QueryBot?"
- "Show me how to use LlamaBot with Ollama"

**API and Configuration Questions:**

- "What are the parameters for StructuredBot?"
- "How do I configure LlamaBot to use different model providers?"
- "How do I add memory to a bot?"

**Advanced Usage Questions:**

- "Show me examples of using the AgentBot with tools"
- "How do I create a custom document store?"
- "What's the best way to implement RAG with LlamaBot?"

**Tutorial and How-To Questions:**

- "Walk me through creating a QueryBot"
- "How do I build a chatbot with conversation history?"
- "Show me how to implement structured outputs"

### Tool Parameters

The `docs_search` tool accepts:

- **query** (string): Natural language search query
- **limit** (integer, optional): Maximum number of results to return (default: 5)

Your AI assistant handles these parameters automatically based on your question.

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

## CI/CD and Packaging

The MCP documentation database follows a two-phase build process during CI/CD to ensure the database is included in the final package.

### Build Process

1. **Database Build**: The MCP database is built using `scripts/build_mcp_docs.py`
2. **Database Copy**: The built database is copied to `llamabot/data/mcp_docs/`
3. **Package Build**: The Python package is built once with the database included
4. **Artifacts Inclusion**: Hatchling includes the database files via `artifacts` configuration

### Build Configuration

The database directory is in `.gitignore` but still gets packaged through the `artifacts` configuration in `pyproject.toml`:

```toml
[tool.hatch.build.targets.wheel]
artifacts = [
    "llamabot/data/mcp_docs/**/*",
]
```

This pattern allows build artifacts to be included in the package even when they're not tracked in git.

### CI/CD Workflow

The process is implemented in `.github/workflows/release-python-package.yaml`:

1. Build MCP database with `pixi run python scripts/build_mcp_docs.py`
2. Verify database files exist in `llamabot/data/mcp_docs/`
3. Build package with `uv build --sdist --wheel`
4. Publish to PyPI

This ensures every release includes a fresh, up-to-date documentation database.

## Testing and Development

### Testing the MCP Server Locally

Before configuring your AI tool, you can test if the MCP server works correctly:

1. **Verify Installation**:

   ```bash
   uvx --with llamabot[all] llamabot mcp --help
   ```

   This should show the MCP CLI commands (build and launch).

2. **Check Database Exists**:

   The packaged database should be included with your installation. To verify:

   ```bash
   python -c "from llamabot.mcp_server import get_packaged_docstore_path; print(get_packaged_docstore_path())"
   ```

3. **Build Custom Database** (optional for development):

   ```bash
   llamabot mcp build
   ```

   This creates a fresh database at `~/.llamabot/mcp_docs/` with the latest documentation.

### Development Configuration

For local LlamaBot development, use the development MCP configuration:

```json
{
  "servers": {
    "llamabot-mcp-server-dev": {
      "type": "stdio",
      "command": "pixi",
      "args": ["run", "llamabot", "mcp", "launch"]
    }
  }
}
```

This configuration uses `pixi run` to execute the MCP server from your local development environment. See `.vscode/mcp.json` in the LlamaBot repository for an example.

## Troubleshooting

### Server Not Showing Up

**Problem**: The llamabot-docs server doesn't appear in your AI tool.

**Solutions**:

- Restart your AI tool (Claude Desktop, Claude Code, Cursor, etc.)
- Verify the configuration file was saved correctly
- Check that `uvx` is available in your PATH: `uvx --version`
- For Claude Code, run `claude mcp list` to see configured servers

### Database Not Found

**Problem**: Error message "LlamaBot documentation database not found"

**Solutions**:

- Verify LlamaBot is installed with all dependencies: `pip install "llamabot[all]"`
- Check if the packaged database exists (should be included with installation)
- Build a fresh database: `llamabot mcp build`
- If using a custom build, verify `~/.llamabot/mcp_docs/` exists and contains data

### Command Not Found

**Problem**: `uvx: command not found` or similar errors

**Solutions**:

- Install uv (which includes uvx): `pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Verify uvx is in your PATH: `which uvx` (Unix) or `where uvx` (Windows)
- Alternative: Use `python -m llamabot.cli.mcp` instead of the uvx command
- For development: Use `pixi run llamabot mcp launch` if working from the repository

### Slow Startup

**Problem**: The first launch takes a long time.

**Explanation**: The first launch may be slow due to model loading and database initialization.

**Solutions**:

- Subsequent launches are much faster (database is cached)
- This is normal behavior and does not indicate a problem
- Consider building the database once with `llamabot mcp build` for optimal performance

### Outdated Documentation

**Problem**: The documentation seems out of date.

**Solution**:

To get the latest documentation from GitHub:

```bash
llamabot mcp build
```

This rebuilds the database with the most recent content from the main branch. The user-built database (in `~/.llamabot/mcp_docs/`) takes precedence over the packaged database.

### Server Fails to Start

**Problem**: MCP server won't start or crashes immediately.

**Solutions**:

- Check Python version: LlamaBot requires Python 3.9+
- Verify all dependencies are installed: `pip install "llamabot[all]"`
- Look for error messages in your AI tool's logs
- Try running the server manually to see detailed errors: `uvx --with llamabot[all] llamabot mcp launch`
- Check if port 8765 is already in use (if using the --port option)

### Permission Errors

**Problem**: Permission denied when accessing the database.

**Solutions**:

- Check permissions on `~/.llamabot/mcp_docs/` directory
- Try building the database with your user account: `llamabot mcp build`
- On Windows, ensure your user has write access to the AppData directory
- On Unix systems, check directory ownership: `ls -la ~/.llamabot/`

## Frequently Asked Questions

### What data is included in the MCP server?

The MCP server indexes:

- All markdown documentation from the `docs/` directory
- Python module docstrings from the LlamaBot source code
- Tutorial and how-to guides
- API reference documentation

Release notes are excluded to keep the database focused on current functionality.

### Do I need to rebuild the database?

No, most users don't need to rebuild. The pre-built database is included with each LlamaBot release and is automatically updated when you upgrade LlamaBot.

Rebuild only if:

- You're developing LlamaBot locally
- You need documentation for unreleased features
- You want to ensure you have the absolute latest documentation from GitHub

### Can I use this with other MCP clients?

Yes! The LlamaBot MCP server follows the Model Context Protocol standard and works with any MCP-compatible client. The server uses the stdio transport, which is widely supported.

### How do I update the MCP server?

Simply update LlamaBot:

```bash
pip install --upgrade "llamabot[all]"
```

The new version will include an updated documentation database.

### Does this work offline?

Yes! The pre-built database is included with LlamaBot, so the MCP server works completely offline. You only need internet access if you want to rebuild the database with the latest documentation from GitHub.

### How large is the database?

The documentation database is relatively small (typically a few MB) and shouldn't noticeably impact installation size or performance.

### Can I customize what's included in the database?

Yes! For advanced users, you can modify `scripts/build_mcp_docs.py` to customize which files are included. After making changes, run `llamabot mcp build` to create your custom database.

### Is this different from QueryBot?

Yes. The MCP server makes LlamaBot documentation available to AI coding assistants through the Model Context Protocol. QueryBot is a LlamaBot class that lets you build RAG applications with your own documents. They serve different purposes but use similar underlying technology (LanceDB for vector search).

## Real-World Usage Examples

These examples demonstrate how the MCP server helps you work with LlamaBot more effectively.

### Example 1: Learning the Basics

**You**: "How do I create a SimpleBot in LlamaBot?"

**AI Assistant**: Uses `docs_search` to find SimpleBot documentation, then provides:

- Code example showing SimpleBot initialization
- Explanation of system prompts and model names
- Links to related documentation

**Result**: You get accurate, up-to-date information directly from LlamaBot's documentation without leaving your coding environment.

### Example 2: Exploring API Parameters

**You**: "What are all the parameters I can pass to StructuredBot?"

**AI Assistant**: Searches the documentation and returns:

- Complete parameter list with descriptions
- Type information and default values
- Usage examples and best practices

**Result**: You can configure StructuredBot correctly without searching through source code or external docs.

### Example 3: Comparing Bot Types

**You**: "What's the difference between SimpleBot and QueryBot? When should I use each?"

**AI Assistant**: Retrieves relevant documentation about both bot types and explains:

- SimpleBot: Stateless, function-like interactions
- QueryBot: RAG-enabled, document querying capabilities
- Use cases for each type

**Result**: You understand the architecture and can choose the right bot for your use case.

### Example 4: Advanced Features

**You**: "Show me how to add conversation memory to a bot"

**AI Assistant**: Finds documentation about ChatMemory and provides:

- Code examples of memory integration
- Explanation of linear vs. threaded memory
- Performance considerations

**Result**: You can implement advanced features with confidence.

### Example 5: Troubleshooting

**You**: "I'm getting an error with Ollama models in LlamaBot. How do I fix it?"

**AI Assistant**: Searches documentation for Ollama-related content:

- Correct model name format (`ollama_chat/model_name`)
- Configuration requirements
- Common issues and solutions

**Result**: You solve problems faster with access to comprehensive documentation.
