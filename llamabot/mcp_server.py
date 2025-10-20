"""MCP server for exposing LlamaBot documentation as resources to AI agents."""

import ast
import requests
import tempfile
from pathlib import Path
from typing import List, Dict, Optional
from fastmcp import FastMCP
from pyprojroot import here
from llamabot import LanceDBDocStore


def collect_markdown_files(docs_dir: Path) -> List[Path]:
    """Recursively collect all markdown files from the docs directory.

    Excludes release notes from docs/releases/ directory.

    :param docs_dir: Path to the documentation directory.
    :return: List of Path objects for all markdown files.
    """
    all_md_files = list(docs_dir.rglob("*.md"))
    # Filter out release notes
    return [f for f in all_md_files if "releases/" not in str(f.relative_to(docs_dir))]


def extract_module_docstring(module_path: Path) -> str:
    """Extract the module-level docstring from a Python file.

    :param module_path: Path to the Python module.
    :return: The module docstring or empty string if not found.
    """
    try:
        with open(module_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
            docstring = ast.get_docstring(tree)
            return docstring if docstring else ""
    except (SyntaxError, UnicodeDecodeError):
        return ""


def collect_python_modules(source_dir: Path) -> Dict[str, str]:
    """Collect all Python modules and their docstrings from the source directory.

    :param source_dir: Path to the source code directory.
    :return: Dictionary mapping module paths to their docstrings.
    """
    modules = {}
    for py_file in source_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        docstring = extract_module_docstring(py_file)
        if docstring:
            relative_path = py_file.relative_to(source_dir)
            modules[str(relative_path)] = docstring
    return modules


def fetch_docs_from_github() -> Optional[Path]:
    """Fetch documentation from GitHub using the tree API.

    :return: Path to temporary directory containing downloaded docs, or None if failed.
    """
    try:
        # Get the latest commit SHA for main branch
        commit_url = "https://api.github.com/repos/ericmjl/llamabot/commits/main"
        commit_response = requests.get(commit_url, timeout=10)
        commit_response.raise_for_status()
        commit_sha = commit_response.json()["sha"]

        # Get tree for the entire repository
        tree_url = (
            f"https://api.github.com/repos/ericmjl/llamabot/git/trees/{commit_sha}"
        )
        tree_response = requests.get(tree_url, params={"recursive": "1"}, timeout=10)
        tree_response.raise_for_status()

        tree_data = tree_response.json()
        temp_dir = Path(tempfile.mkdtemp())

        # Find all markdown files in docs/ directory
        for item in tree_data["tree"]:
            if (
                item["path"].startswith("docs/")
                and item["path"].endswith(".md")
                and "docs/releases/" not in item["path"]
            ):
                # Download the file
                file_url = f"https://raw.githubusercontent.com/ericmjl/llamabot/main/{item['path']}"
                file_response = requests.get(file_url, timeout=10)
                file_response.raise_for_status()
                file_content = file_response.text

                # Create local path preserving directory structure
                local_path = temp_dir / item["path"]
                local_path.parent.mkdir(parents=True, exist_ok=True)
                local_path.write_text(file_content, encoding="utf-8")

        return temp_dir

    except (requests.RequestException, KeyError, Exception) as e:
        # Log error but don't fail completely
        print(f"Warning: Could not fetch docs from GitHub: {e}")
        return None


def get_docstore_path() -> Path:
    """Get the path to the persistent LanceDB storage.

    :return: Path to the LanceDB storage directory.
    """
    return Path.home() / ".llamabot" / "mcp_docs"


def get_packaged_docstore_path() -> Path:
    """Get the path to the packaged LanceDB database.

    :return: Path to the packaged LanceDB storage directory.
    """
    import llamabot

    return Path(llamabot.__file__).parent / "data" / "mcp_docs"


def build_docstore() -> None:
    """Build the LanceDB document store with all LlamaBot documentation.

    This function fetches documentation from GitHub, collects source code docstrings,
    and builds a persistent LanceDB database for fast searching.
    """
    print("Building LlamaBot documentation database...")

    # Get local source code (always available)
    import llamabot

    source_dir = Path(llamabot.__file__).parent
    python_modules = collect_python_modules(source_dir)

    # Try to fetch docs from GitHub first
    docs_dir = fetch_docs_from_github()
    markdown_files = []

    if docs_dir and docs_dir.exists():
        markdown_files = collect_markdown_files(docs_dir)
        print(f"Found {len(markdown_files)} documentation files from GitHub")
    else:
        # Fallback: check if we're in development environment
        project_root = here()
        local_docs_dir = project_root / "docs"
        if local_docs_dir.exists():
            docs_dir = local_docs_dir
            markdown_files = collect_markdown_files(docs_dir)
            print(f"Found {len(markdown_files)} documentation files locally")
        else:
            print("No documentation found")

    # Create persistent LanceDBDocStore
    docstore_path = get_docstore_path()
    docstore_path.mkdir(parents=True, exist_ok=True)

    docstore = LanceDBDocStore(table_name="llamabot_docs", storage_path=docstore_path)

    print("Adding documentation files to database...")
    # Add all markdown files to the docstore
    for md_file in markdown_files:
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Calculate relative path for metadata
        if docs_dir:
            relative_path = md_file.relative_to(docs_dir)
        else:
            relative_path = md_file.name

        # Format content with metadata
        formatted_content = f"# {md_file.stem}\n\nPath: {relative_path}\n\n{content}"
        docstore.append(formatted_content)

    print("Adding source code docstrings to database...")
    # Add Python module docstrings to the docstore
    for module_path, docstring in python_modules.items():
        if docstring.strip():  # Only add modules with docstrings
            # Format content with metadata
            formatted_content = f"# {Path(module_path).stem} (Python Module)\n\nPath: {module_path}\n\n{docstring}"
            docstore.append(formatted_content)

    print(f"Database built successfully at {docstore_path}")
    print(
        f"Total documents: {len(markdown_files) + len([d for d in python_modules.values() if d.strip()])}"
    )


def create_mcp_server() -> FastMCP:
    """Create and configure the FastMCP server with LlamaBot documentation search tool.

    :return: Configured FastMCP server instance.
    """
    mcp = FastMCP("LlamaBot Documentation Server")

    # Check if user-built docstore exists first
    docstore_path = get_docstore_path()
    if not docstore_path.exists():
        # Fall back to packaged database
        packaged_path = get_packaged_docstore_path()
        if packaged_path.exists():
            docstore_path = packaged_path
            print(f"Using packaged documentation database from {docstore_path}")
        else:
            raise FileNotFoundError(
                f"LlamaBot documentation database not found at {docstore_path} or {packaged_path}. "
                "Please run 'llamabot mcp build' first to build the database."
            )

    # Load existing docstore
    docstore = LanceDBDocStore(table_name="llamabot_docs", storage_path=docstore_path)

    # Register the single docs_search tool
    @mcp.tool()
    def docs_search(query: str, limit: int = 5) -> dict:
        """Search through LlamaBot documentation and source code.

        :param query: The search query to find relevant documentation.
        :param limit: Maximum number of results to return (default: 5).
        :return: Dictionary containing search results with content and metadata.
        """
        # Search the docstore
        results = docstore.retrieve(query, n_results=limit)

        # Format results for the client
        formatted_results = []
        for i, result in enumerate(results):
            formatted_results.append(
                {
                    "content": result,
                    "rank": i + 1,
                    "relevance_score": 1.0 - (i / len(results)) if results else 0.0,
                }
            )

        return {
            "query": query,
            "total_results": len(formatted_results),
            "results": formatted_results,
        }

    return mcp
