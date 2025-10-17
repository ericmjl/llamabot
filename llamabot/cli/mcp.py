"""Launch the MCP server to expose LlamaBot documentation to AI agents."""

import typer

app = typer.Typer()


@app.command()
def build():
    """Build the LlamaBot documentation database for the MCP server.

    This command fetches the latest documentation from GitHub, extracts
    docstrings from the source code, and builds a persistent LanceDB database
    for fast semantic search. The database is stored in ~/.llamabot/mcp_docs/
    and only needs to be rebuilt when documentation changes.

    For most users, the pre-built database included with the package is sufficient.
    Use this command only if you need the latest documentation or are developing
    locally.

    After building, use "uvx --with llamabot[all] llamabot mcp launch" in your
    MCP-compatible coding tools (Cursor, VSCode, etc.) to access LlamaBot documentation.
    """
    from llamabot.mcp_server import build_docstore

    build_docstore()


@app.command()
def launch(
    host: str = typer.Option("0.0.0.0", help="Host to bind the server to"),
    port: int = typer.Option(8765, help="Port to run the server on"),
):
    """Launch the MCP server to expose LlamaBot documentation search tool.

    The server provides a single tool: docs_search
    - Searches through all LlamaBot documentation and source code
    - Uses semantic search to find relevant content
    - Returns structured results with content and metadata

    This tool can be used by MCP-compatible AI coding agents to find
    relevant LlamaBot documentation and source code information.

    Installation in coding tools:
    - Cursor/VSCode: Add to MCP settings: "uvx --with llamabot[all] llamabot mcp launch"
    - Other tools: Use "uvx --with llamabot[all] llamabot mcp launch" as the MCP server command

    The documentation database is pre-built and included with the package.

    :param host: The host to bind the server to.
    :param port: The port to run the server on.
    """
    from llamabot.mcp_server import create_mcp_server

    mcp = create_mcp_server()

    typer.echo(f"Starting MCP server at {host}:{port}")
    typer.echo("Loading documentation database...")
    typer.echo("Ready! Use the 'docs_search' tool to query LlamaBot documentation.")

    # Run the MCP server with auto-reload enabled
    mcp.run(transport="stdio")


if __name__ == "__main__":
    app()
