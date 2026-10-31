# MCP Server

The LlamaBot MCP (Model Context Protocol) server gives coding agents one
tool, `docs_search`, over LlamaBot's documentation and the module
docstrings in the installed package.

This page covers that documentation server (`llamabot mcp launch`).
Wiring AgentBot or ToolBot to other MCP servers is a separate feature:
[Use MCP tools with AgentBot](../how-to/agent-mcp-tools.md).

## Features

- Search with a natural-language query. `docs_search` calls
  `LanceDBDocStore.retrieve`, which runs a LanceDB `auto` query (vector
  search plus the full-text index on the document field) and reranks with
  ColBERT. The embedding model is `minishlab/potion-base-8M`.
- The index includes Markdown from `docs/`, except files under
  `docs/releases/`, and module-level docstrings from the installed
  LlamaBot package. Function and class docstrings are not indexed on
  their own.
- Release builds ship a pre-built database inside the wheel, so a normal
  install can start without a local build.
- The server speaks MCP on stdin and stdout. Cursor, VS Code, and Claude
  Desktop all start it as a local command.

## Quick Start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) so
`uvx` is on your `PATH`. The app starts the server. The command is:

```bash
uvx --with "llamabot[all]" llamabot mcp launch
```

`llamabot/cli/mcp.py` names the `all` extra in this command. That extra
installs FastMCP and LanceDB, which `launch` imports. Keep the square
brackets. Quote `llamabot[all]` in a shell so the brackets are not a glob.
Inside JSON `args`, it is a normal string.

`llamabot mcp` has two commands, `build` and `launch`. `launch` calls
`FastMCP.run(transport="stdio")`.

### Config file locations

| App | File |
| --- | --- |
| Cursor, every project | `~/.cursor/mcp.json` in your home directory. Cursor documents this `~/` path for the global file. |
| Cursor, one project | `.cursor/mcp.json` at the project root. |
| VS Code, this workspace | `.vscode/mcp.json` at the workspace root. |
| Claude Desktop on macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Claude Desktop on Windows | `%APPDATA%\Claude\claude_desktop_config.json` |
| Claude Desktop on Linux | `~/.config/Claude/claude_desktop_config.json` |

Cursor reads both of its files and merges them. If the same server name
appears in both, the project file wins. That behavior is documented in
the [Cursor MCP help](https://cursor.com/help/customization/mcp).

Cursor and Claude Desktop use a top-level `mcpServers` object. VS Code's
`.vscode/mcp.json` uses a top-level `servers` object and sets `"type":
"stdio"`. Use the block in the section for the app you are configuring.

### Confirm it is connected

Save the file, restart the app, and check that `docs_search` is listed.
Steps for each app are in [Verification](#verification).

To see whether the command itself starts, run the same `uvx` line in a
terminal. On success it prints:

```text
Ready! Use the 'docs_search' tool to query LlamaBot documentation.
```

The process then waits on stdin for MCP messages. Stop it with Ctrl-C.
That terminal process is a separate process from the one the app starts.

If the database is missing, the process raises `FileNotFoundError` and
the message contains `LlamaBot documentation database not found`. See
[Troubleshooting](#troubleshooting).

## Installation

Wheels built by the release workflow include a database at
`llamabot/data/mcp_docs/` inside the installed package. Build a local
database when you want docs newer than that copy. See
[Database management](#database-management).

### Configure your coding tool

#### Cursor

Create the global file, the project file, or both. Paths are in
[Cursor's MCP reference](https://cursor.com/docs/mcp).

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

Save the file and restart Cursor. The server is listed under Customize,
then MCPs. See [Check Cursor](#check-cursor).

#### VS Code

Create `.vscode/mcp.json` in the workspace root. It applies to that
workspace. VS Code forwards these servers to Copilot chat. Servers that
use interactive `${input:...}` variables are left out of that hand-off.
This configuration does not use input variables.

The user-profile file uses the same `servers` schema and applies to every
workspace. Open it from the Command Palette with
`MCP: Open User Configuration`. Both destinations are described in
[Add and manage MCP servers in VS Code](https://code.visualstudio.com/docs/copilot/customization/mcp-servers).
`"type": "stdio"` is required for a stdio server in the
[MCP configuration reference](https://code.visualstudio.com/docs/agents/reference/mcp-configuration).

```json
{
  "servers": {
    "llamabot-docs": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--with", "llamabot[all]", "llamabot", "mcp", "launch"]
    }
  }
}
```

The repository's `.vscode/mcp.json` uses this same `servers` shape, with
`pixi run` as the command for local development. After saving, follow
[Check VS Code](#check-vs-code).

#### Claude Desktop

Claude Desktop reads `claude_desktop_config.json`. From the app, open the
Claude menu, choose Settings, open the Developer tab, and click Edit
Config. The
[local MCP server guide](https://modelcontextprotocol.io/docs/develop/connect-local-servers)
lists the file locations:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

Edit Config creates the file when it is absent. If `mcpServers` already
has entries, add `llamabot-docs` next to them.

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

Quit Claude Desktop completely and start it again. See
[Check Claude Desktop](#check-claude-desktop).

## Verification

`docs_search` is the only tool. Its arguments are `query` (string) and
`limit` (integer, default 5).

### Check Cursor

1. Restart Cursor after changing `mcp.json`.
2. Open Customize in the sidebar and select MCPs. `llamabot-docs` is
   listed there. The toggle shows whether the server is enabled.
3. In chat, MCP tools show up with the other available tools. You can ask
   for `docs_search` by name. Cursor describes this under
   [Using MCP in chat](https://cursor.com/docs/mcp).
4. When the server is missing, open the Output panel (Cmd+Shift+U on
   macOS, Ctrl+Shift+U on Windows and Linux) and select MCP Logs.

### Check VS Code

1. Open the Command Palette and run `MCP: List Servers`.
2. Start `llamabot-docs` from that list if it is stopped, and show its
   output from the same menu.
3. In Copilot chat, ask a LlamaBot documentation question and approve the
   `docs_search` call when the client asks.
4. When the tool list was cached before the server changed, run
   `MCP: Reset Cached Tools`.

`MCP: List Servers`, show output, and `MCP: Reset Cached Tools` are the
commands in VS Code's
[MCP configuration reference](https://code.visualstudio.com/docs/agents/reference/mcp-configuration).

### Check Claude Desktop

1. Quit Claude Desktop completely and start it again, so it reloads
   `claude_desktop_config.json`. The local-server guide requires a full
   restart.
2. In the conversation input, open "Add files, connectors, and more",
   then Connectors, then Manage connectors. Select `llamabot-docs` and
   confirm `docs_search` is listed. Those labels are the ones in the
   current
   [local-server guide](https://modelcontextprotocol.io/docs/develop/connect-local-servers).
3. Ask "How do I create a SimpleBot?" and approve the `docs_search` call
   when Claude requests it.

Connection logs are `mcp.log`. Stderr from this server is
`mcp-server-llamabot-docs.log`. The guide places those files in:

- macOS: `~/Library/Logs/Claude`
- Windows: `%APPDATA%\Claude\logs`
- Linux: `~/.config/Claude/logs`

### Terminal smoke test

```bash
uvx --with "llamabot[all]" llamabot mcp launch
```

When the packaged database is selected, the process prints
`Using packaged documentation database from ...` before the Ready line.
When `~/.llamabot/mcp_docs/` exists, that directory is selected and the
packaged-database line is not printed. The command also prints
`Starting MCP server at 0.0.0.0:8765` and
`Loading documentation database...`. The host and port in that line are
the `--host` and `--port` defaults. See
[`llamabot mcp launch`](#llamabot-mcp-launch).

## Usage

Pass a natural-language `query`. `limit` is the maximum number of hits
(default 5) and is forwarded as `n_results`.

The tool returns a dictionary:

- `query`: the query string
- `total_results`: how many hits came back
- `results`: the hits

Each hit has:

- `content`: the stored document text
- `rank`: 1-based position
- `relevance_score`: `1.0 - (rank - 1) / total_results`

`relevance_score` is computed from `rank`. It is not the embedding
distance and not the ColBERT score. With the default limit of 5 and five
hits, the scores are 1.0, 0.8, 0.6, 0.4, and 0.2.

A Markdown page is stored as a heading (the file stem), a `Path:` line,
and the file body. A module docstring is stored as
`# <module> (Python Module)`, a `Path:` line, and the docstring. `Path:`
is text inside `content`.

## Commands

### llamabot mcp build

Rebuild the database at `~/.llamabot/mcp_docs/`. The command accepts no
options.

`build_docstore` does the following:

1. Collects module-level docstrings from the installed `llamabot` package.
   Modules whose module docstring is empty are skipped.
2. Downloads Markdown from the `main` branch of `ericmjl/llamabot` through
   the GitHub API, skipping `docs/releases/`.
3. If the download fails and the project root of the current working
   directory has a `docs/` directory, indexes that local tree instead,
   still skipping `docs/releases/`.
4. Writes a LanceDB table. The call is
   `LanceDBDocStore(table_name="llamabot_docs", storage_path=...)`.
   The store slugifies the table name to `llamabot-docs`.

With `uvx`:

```bash
uvx --with "llamabot[all]" llamabot mcp build
```

The database is under your home directory, so a later `launch` from
another `uvx` environment uses the same files.

### llamabot mcp launch

Start the stdio server.

| Option | Default | Effect |
| --- | --- | --- |
| `--host` | `0.0.0.0` | Included in the startup print. Not an argument to `mcp.run`. |
| `--port` | `8765` | Included in the startup print. Not an argument to `mcp.run`. |

```bash
uvx --with "llamabot[all]" llamabot mcp launch
```

Point clients at `command` and `args`. This process has no HTTP or SSE
endpoint, so a `url` entry does not connect to it.

FastMCP is constructed as `FastMCP("LlamaBot Documentation Server")`.
The key in your JSON (`llamabot-docs` in the samples) is the client-side
server name.

## Database management

`create_mcp_server` chooses the directory in this order:

1. `~/.llamabot/mcp_docs/` when that path exists.
2. `llamabot/data/mcp_docs/` inside the installed package when the user
   path is absent and the packaged path exists. The process then prints
   `Using packaged documentation database from ...`.
3. `FileNotFoundError` when neither path exists.

The existence check is `Path.exists()` on the directory. A user directory
with no table is still the selected path, so startup does not switch to
the packaged database. Run `llamabot mcp build` to populate
`~/.llamabot/mcp_docs/`, or delete that directory to use the packaged
copy.

## CI/CD and packaging

The release workflow builds the database before the wheel so the files
are in the package. The workflow file is
`.github/workflows/release-python-package.yaml`.

1. `pixi run python scripts/build_mcp_docs.py` builds
   `~/.llamabot/mcp_docs/` and copies it to `llamabot/data/mcp_docs/`.
2. The workflow lists files under `llamabot/data/mcp_docs/`.
3. `uv build --sdist --wheel` builds the package.
4. On the publish path, `uv publish` uploads to PyPI.

`llamabot/data/mcp_docs/` is gitignored. Hatchling still packages it
because `pyproject.toml` lists the directory in wheel artifacts:

```toml
[tool.hatch.build.targets.wheel]
artifacts = [
    "llamabot/data/mcp_docs/**/*",
]
```

## Troubleshooting

### MCP server not connecting

- The config file has to be the path in
  [Config file locations](#config-file-locations). VS Code's workspace
  file uses `servers`, while Cursor and Claude Desktop use `mcpServers`.
- The file has to be JSON. Trailing commas and comments are rejected.
- Cursor logs: Output panel, MCP Logs.
- VS Code logs: `MCP: List Servers`, then show output.
- Claude Desktop logs: `mcp.log` and `mcp-server-llamabot-docs.log` in the
  directory under [Check Claude Desktop](#check-claude-desktop).
- Run the `uvx` command in a terminal to read the traceback directly.

### Tool not appearing

Restart the app after editing the config. Claude Desktop needs a full
quit, as in [Check Claude Desktop](#check-claude-desktop). In VS Code,
start the server from `MCP: List Servers`, then run
`MCP: Reset Cached Tools` when the tool list was cached earlier. In
Cursor, enable the server under Customize, then MCPs.

`MCP: Reset Cached Tools` clears the cached tool list. It does not reload
document text. Restart the server after `llamabot mcp build` so the new
index is opened.

### uvx command not found

Install uv from the
[uv installation guide](https://docs.astral.sh/uv/getting-started/installation/),
then run `uvx --version`. Restart Cursor, VS Code, or Claude Desktop
after installing uv so the app process can see `uvx`. Cursor's
[MCP troubleshooting](https://cursor.com/help/customization/mcp) says to
restart Cursor after a shell profile change for the same reason.

### Database not found

Startup raises `FileNotFoundError` with this text (paths filled in):

```text
LlamaBot documentation database not found at <user path> or <packaged path>. Please run 'llamabot mcp build' first to build the database.
```

`<user path>` is `~/.llamabot/mcp_docs` expanded to an absolute path.
`<packaged path>` is `llamabot/data/mcp_docs` under the installed package.

Build a database:

```bash
uvx --with "llamabot[all]" llamabot mcp build
```

`build` calls the GitHub API and raw file host for the `main` branch. If
that request fails, it prints `Warning: Could not fetch docs from GitHub`
and uses the fallback in [`llamabot mcp build`](#llamabot-mcp-build).

### Import errors on startup

`launch` imports FastMCP and constructs a `LanceDBDocStore`. Both are
installed by the `all` extra. A missing FastMCP import looks like
`ModuleNotFoundError: No module named 'fastmcp'`. A missing LanceDB
install raises `ImportError` with the text `LanceDB is required for
LanceDBDocStore` and tells you to install `llamabot[rag]`. Use the `args`
array from the samples, including the `"llamabot[all]"` item.

### Slow first start

Opening the store loads the sentence-transformers model
`minishlab/potion-base-8M` and constructs `ColbertReranker`. The model
files are cached after that first download.

### Stale documentation

```bash
uvx --with "llamabot[all]" llamabot mcp build
```

That rewrites `~/.llamabot/mcp_docs/` from GitHub `main` and from the
module docstrings of the LlamaBot package that command installed. Restart
the MCP server in the app so it opens the new files.

## Examples

### Ask how to create a SimpleBot

In Claude Desktop, Cursor, or VS Code, ask:

```text
How do I create a SimpleBot?
```

The client can call `docs_search` with that sentence as `query`. A return
value with one hit has this shape. `content` is abbreviated here. The
real field is the full stored document.

```json
{
  "query": "How do I create a SimpleBot?",
  "total_results": 1,
  "results": [
    {
      "content": "# simplebot\n\nPath: tutorials/simplebot.md\n\n...",
      "rank": 1,
      "relevance_score": 1.0
    }
  ]
}
```

`relevance_score` is 1.0 in this illustration because there is one hit:
`1.0 - (1 - 1) / 1`.

### Narrow the question

`docs_search` takes `query` and `limit` only. Put the type or module name
in the query so the full-text index and the embedding both see it:

- `SimpleBot constructor parameters`
- `StructuredBot tutorial`
- `QueryBot document store`

Pass `limit` greater than 5 when you want more than the default five
hits, for example `10`.

### Parameters of StructuredBot

Query: `What are the parameters for StructuredBot?`

Hits are stored Markdown pages and module docstrings whose text matches
that query. The `Path:` line inside `content` names the file or module
the hit came from.
